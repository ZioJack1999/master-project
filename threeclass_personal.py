"""Fit a new person's three-class decoder from complete development cues.

No recorded reference/test participant is used here. The prospective validation
run belongs to the caller and must be collected after this model is frozen.
"""
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn
from scipy.signal import butter, sosfilt

from multiclass_features import (band_covariances, FeaturePreprocessor,
                                 FeatureNetwork, FeatureEnsembleRunner)
from multiclass_metrics import evaluate, cue_probabilities, trial_class_weights
from temporal_networks import TemporalEEGNet, TemporalEEGEnsemble
from threeclass_decisions import diagnostic_from_probabilities

CLASSES = ['rest', 'volar_grasp', 'finger_extension']
POLICY = dict(ema_alpha=1., dwell_updates=1, min_confidence=.5, min_margin=.05)
DEFAULTS = dict(sfreq=256, window_s=4., stride_s=.25, settle_s=.5,
                artifact_limit_v=100e-6, min_signal_std_v=1e-9,
                max_gap_s=.1, min_windows_per_cue=4,
                min_cues_per_class=8, tuning_fraction=.25,
                hidden_units=32, dropout=.25, covariance_shrinkage=.1,
                epochs=80, patience=15, batch_size=128, windows_per_cue=4,
                learning_rate=.001, weight_decay=.01,
                ensemble_seeds=[13, 42, 73], seed=42, threads=2, linear_c=None,
                arch='feature_ensemble')


class CalibrationCancelled(ValueError):
    """Fitting stopped without publishing a new model."""


def _check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise CalibrationCancelled('Three-class personal fitting was cancelled.')


def _configuration(config=None):
    c = copy.deepcopy(DEFAULTS)
    c.update(config or {})
    if c['sfreq'] != 256 or c['window_s'] not in (2., 4.) or c['stride_s'] != .25:
        raise ValueError('Three-class personal calibration requires 256 Hz, 2 or 4 s windows and .25 s stride.')
    if c['settle_s'] < .5 or c['min_cues_per_class'] < 8:
        raise ValueError('At least .5 s settling and eight clean independent cues per class are required.')
    if not 0 < c['tuning_fraction'] < .5 or not 1 <= c['windows_per_cue'] <= 4:
        raise ValueError('Invalid tuning fraction or per-cue sampling cap.')
    if c['epochs'] < 1 or c['patience'] < 1 or c['batch_size'] < 1 or not c['ensemble_seeds']:
        raise ValueError('Invalid fitting controls.')
    linear_c = c['linear_c']
    if linear_c is not None and (isinstance(linear_c, bool) or not isinstance(linear_c, (int, float))
                                 or not np.isfinite(linear_c) or linear_c <= 0):
        raise ValueError('Linear regularization C must be positive and finite.')
    if c['arch'] not in ('feature_ensemble', 'temporal_cnn') or (c['arch'] == 'temporal_cnn' and linear_c is not None):
        raise ValueError('Choose covariance features or a temporal CNN. Linear C applies to covariance features.')
    if c['max_gap_s'] <= 1/c['sfreq'] or c['artifact_limit_v'] <= 0 or c['min_signal_std_v'] <= 0:
        raise ValueError('Invalid signal quality limits.')
    return c


def _canonical(label):
    return '_'.join(str(label).casefold().replace('_', ' ').split())


def prepare_calibration(raw, timestamps, events, channels, config=None, segments=None):
    """Return causal windows and provenance; never concatenate across data gaps.

    Complete cues only are eligible. Filtering restarts at explicit segment
    boundaries and timestamp gaps. Windows crossing a restart are rejected.
    Timestamp span checks also reject windows with missing/compressed samples.
    """
    c = _configuration(config)
    x = np.asarray(raw, dtype=np.float32)
    ts = np.asarray(timestamps, dtype=float)
    channels = list(channels)
    if len(channels) != 20 or len({str(ch).strip().casefold() for ch in channels}) != 20 or any(not str(ch).strip() for ch in channels):
        raise ValueError('Personal three-class fitting requires 20 unique verified physical EEG sites.')
    n, step = round(c['window_s']*c['sfreq']), round(c['stride_s']*c['sfreq'])
    if x.ndim != 2 or x.shape != (len(channels), len(ts)) or len(ts) < n:
        raise ValueError('Expected channels x samples in volts and one timestamp per sample.')
    if not np.isfinite(x).all() or not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0):
        raise ValueError('Calibration contains nonfinite EEG or invalid/backward timestamps.')
    seg = np.zeros(len(ts), dtype=np.int64) if segments is None else np.asarray(segments)
    if seg.ndim != 1 or len(seg) != len(ts):
        raise ValueError('Segment IDs must contain one value per sample.')
    if not np.isfinite(seg).all():
        raise ValueError('Invalid calibration segment IDs.')
    starts = np.r_[0, np.flatnonzero((np.diff(ts) > c['max_gap_s']) | (seg[1:] != seg[:-1]))+1]
    ends = np.r_[starts[1:], len(ts)]
    filtered = np.empty_like(x)
    blocks = np.empty(len(ts), dtype=np.int64)
    block_first = {}
    sos = butter(4, [8, 30], fs=c['sfreq'], btype='bandpass', output='sos')
    for block, (a, b) in enumerate(zip(starts, ends)):
        filtered[:, a:b] = sosfilt(sos, x[:, a:b], axis=1).astype(np.float32)
        blocks[a:b] = block
        block_first[block] = ts[a]
    bad = (np.abs(filtered) > c['artifact_limit_v']).any(axis=0)
    windows, rows, counts, seen = [], [], Counter(), set()
    last_end = float('-inf')
    for event in events:
        counts['provided_cues'] += 1
        if not event.get('complete', False):
            counts['incomplete_cues'] += 1
            continue
        label = _canonical(event.get('label', ''))
        if label not in CLASSES:
            raise ValueError(f'Unknown three-class calibration label: {label!r}.')
        trial = str(event.get('trial', '')).strip()
        if not trial or trial in seen:
            raise ValueError('Calibration cue IDs must be nonempty and unique.')
        seen.add(trial)
        onset, end = float(event['onset']), float(event['end'])
        if not np.isfinite([onset, end]).all() or end <= onset or onset < last_end-1/c['sfreq']:
            raise ValueError('Calibration cues overlap, are unsorted or have invalid timing.')
        last_end = end
        counts['complete_cues'] += 1
        a = int(np.searchsorted(ts, onset+c['settle_s'], side='left'))
        stop = int(np.searchsorted(ts, end, side='left'))
        first = len(windows)
        if ts[0] > onset or ts[-1]+1/c['sfreq'] < end:
            counts['signal_boundary_truncated_cues'] += 1
        for start in range(a, stop-n+1, step):
            finish = start+n
            counts['candidate_windows'] += 1
            if blocks[start] != blocks[finish-1]:
                counts['gap_crossing_windows'] += 1
                continue
            if ts[start] < block_first[blocks[start]]+c['settle_s']-1e-9:
                counts['restart_settling_windows'] += 1
                continue
            expected_span = (n-1)/c['sfreq']
            if abs((ts[finish-1]-ts[start])-expected_span) > .02:
                counts['irregular_timestamp_windows'] += 1
                continue
            w = filtered[:, start:finish]
            if bad[start:finish].any():
                counts['artifact_windows'] += 1
                continue
            if np.any(w.std(axis=1) < c['min_signal_std_v']):
                counts['flat_windows'] += 1
                continue
            windows.append(w.copy())
            rows.append(dict(trial=trial, label=CLASSES.index(label), run='personal_calibration',
                             start=start, end=finish, segment=int(blocks[start]),
                             onset=onset, cue_end=end,
                             **({'calibration_block':event['calibration_block']} if 'calibration_block' in event else {})))
        if 0 < len(windows)-first < c['min_windows_per_cue']:
            counts['insufficient_clean_windows_cues'] += 1
            del windows[first:]
            del rows[first:]
        elif len(windows) > first:
            counts['clean_cues'] += 1
    counts['kept_windows'] = len(windows)
    counts['filter_segments'] = len(starts)
    counts['clean_cues_per_class'] = {
        name:len({r['trial'] for r in rows if r['label'] == i}) for i, name in enumerate(CLASSES)}
    if not windows:
        raise ValueError('No clean complete calibration cues remain after quality/gap checks.')
    return np.stack(windows), rows, dict(counts)


def split_cues(records, seed=42, min_cues_per_class=8, tuning_fraction=.25, events=None, *, reserved_trials=None):
    """Reserve later cues/blocks for tuning, retaining every overlapping copy.

    Block boundaries are determined from the ORIGINAL cue plan before quality
    filtering. Failed later tuning cues never move earlier data into tuning or
    later data into training. ``seed`` is retained for API compatibility; the
    default split is chronological. Archived recordings can explicitly reserve
    whole cues before signal quality filtering.
    """
    if min_cues_per_class < 8 or not 0 < tuning_fraction < .5:
        raise ValueError('Use at least eight clean independent cues per class and a valid tuning fraction.')
    groups = defaultdict(list)
    for i, row in enumerate(records):
        groups[row['trial']].append(i)
    labels = {}
    for trial, indices in groups.items():
        values = {records[i]['label'] for i in indices}
        if len(values) != 1 or values.pop() not in range(3):
            raise ValueError('One calibration cue has conflicting/unknown class labels.')
        labels[trial] = records[indices[0]]['label']
    for label in range(3):
        trials = sorted(t for t, y in labels.items() if y == label)
        if len(trials) < min_cues_per_class:
            raise ValueError(f'{CLASSES[label]}: need at least {min_cues_per_class} clean independent calibration cues; found {len(trials)}.')
    block_events = events is not None and any('calibration_block' in e for e in events)
    if reserved_trials is not None:
        reserved = set(reserved_trials)
        plan = {str(event['trial']) for event in (events or []) if event.get('complete')}
        if not reserved or len(reserved) != len(reserved_trials) or not reserved <= plan:
            raise ValueError('Reserve unique cue IDs from the completed recording plan.')
        if not set(groups) <= plan:
            raise ValueError('Accepted cue is absent from the completed recording plan.')
        assignments = defaultdict(set)
        for event in events:
            if 'calibration_block' in event:
                key = event.get('run_id'), event['calibration_block']
                assignments[key].add(str(event['trial']) in reserved)
        if any(len(membership) > 1 for membership in assignments.values()):
            raise ValueError('Reserve whole calibration blocks when the recording declares blocks.')
        train_trials = [trial for trial in groups if trial not in reserved]
        tuning_trials = [trial for trial in groups if trial in reserved]
    elif block_events:
        plan = {}
        for event in events:
            block = event.get('calibration_block')
            if isinstance(block, bool) or not isinstance(block, (int, np.integer)) or block < 0:
                raise ValueError('Every calibration event needs a nonnegative integer calibration block.')
            trial = str(event.get('trial', '')).strip()
            if not trial or trial in plan:
                raise ValueError('Original calibration plan has missing or duplicate cue IDs.')
            onset = float(event['onset'])
            if not np.isfinite(onset):
                raise ValueError('Original calibration cue onsets must be finite.')
            label = _canonical(event.get('label', ''))
            if label not in CLASSES:
                raise ValueError('Original calibration plan contains an unknown class.')
            plan[trial] = dict(block=int(block), onset=onset, label=CLASSES.index(label))
        ordered = sorted(plan.values(), key=lambda item:item['onset'])
        if any(a['block'] > b['block'] for a, b in zip(ordered, ordered[1:])):
            raise ValueError('Calibration blocks must follow recorded cue order.')
        blocks = sorted({item['block'] for item in plan.values()})
        for block in blocks:
            if Counter(item['label'] for item in plan.values() if item['block'] == block) != Counter({0:1, 1:1, 2:1}):
                raise ValueError('Each original calibration block must contain exactly one cue of each class.')
        n_tune_blocks = max(2, int(np.ceil(len(blocks)*tuning_fraction)))
        if n_tune_blocks >= len(blocks):
            raise ValueError('Not enough earlier calibration blocks remain for training.')
        tuning_blocks = set(blocks[-n_tune_blocks:])
        train_trials, tuning_trials = [], []
        for trial, indices in groups.items():
            if trial not in plan:
                raise ValueError('Accepted cue is absent from the original calibration plan.')
            block = plan[trial]['block']
            if labels[trial] != plan[trial]['label']:
                raise ValueError('Accepted cue label disagrees with the original calibration plan.')
            if any(records[i].get('calibration_block') != block for i in indices):
                raise ValueError('Accepted cue block disagrees with the original calibration plan.')
            (tuning_trials if block in tuning_blocks else train_trials).append(trial)
    else:
        if any('calibration_block' in r for r in records):
            raise ValueError('Original calibration events are required to reserve declared blocks before quality filtering.')
        # Legacy calibration recordings still receive one global chronological
        # cut. Do not interleave per-class later and earlier cues.
        use_onsets = all('onset' in r for r in records)
        key = 'onset' if use_onsets else 'start'
        if any(key not in r or not np.isfinite(r[key]) for r in records):
            raise ValueError('Chronological calibration splitting requires recorded cue onset/start times.')
        order = sorted(groups, key=lambda trial:(min(records[i][key] for i in groups[trial]), trial))
        target = len(order)*(1-tuning_fraction)
        possible = []
        for cut in range(1, len(order)):
            before = Counter(labels[t] for t in order[:cut])
            after = Counter(labels[t] for t in order[cut:])
            if all(before[i] >= 4 and after[i] >= 2 for i in range(3)):
                possible.append(cut)
        if not possible:
            raise ValueError('No chronological split has enough earlier training and later tuning cues for all classes; collect balanced cue blocks.')
        cut = min(possible, key=lambda value:(abs(value-target), value))
        train_trials, tuning_trials = order[:cut], order[cut:]
    train_counts = Counter(labels[t] for t in train_trials)
    tuning_counts = Counter(labels[t] for t in tuning_trials)
    if not all(train_counts[i] >= 4 and tuning_counts[i] >= 2 for i in range(3)):
        raise ValueError('Reserved calibration blocks need at least four clean training and two clean tuning cues per class; collect a new calibration.')
    train = [i for trial in train_trials for i in groups[trial]]
    tuning = [i for trial in tuning_trials for i in groups[trial]]
    if reserved_trials is None and all('onset' in r and 'cue_end' in r for r in records):
        if max(records[i]['cue_end'] for i in train) > min(records[i]['onset'] for i in tuning)+1/256:
            raise ValueError('Training and tuning cues overlap the chronological boundary.')
    train, tuning = np.array(sorted(train), dtype=int), np.array(sorted(tuning), dtype=int)
    if {records[i]['trial'] for i in train} & {records[i]['trial'] for i in tuning}:
        raise ValueError('Whole-cue calibration split leaked.')
    return train, tuning


def _probabilities(model, features):
    model.eval()
    device = next(model.parameters()).device
    with torch.inference_mode():
        return np.concatenate([torch.softmax(model(torch.from_numpy(features[a:a+256]).to(device)), 1).cpu().numpy()
                               for a in range(0, len(features), 256)])


def _metrics(probabilities, rows):
    y = np.array([r['label'] for r in rows], dtype=np.int64)
    result = evaluate(probabilities, y, rows, CLASSES)
    truth, p, _ = cue_probabilities(probabilities, rows)
    result['cue_nll'] = float(-np.log(np.maximum(p[np.arange(len(truth)), truth], 1e-12)).mean())
    result['deployment_policy'] = diagnostic_from_probabilities(probabilities, rows, CLASSES, POLICY)
    return result


def _fit_member(tx, tr, vx, vr, pre, c, seed, cancel, progress, device, *, initial_state=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    temporal = c['arch'] == 'temporal_cnn'
    model = (TemporalEEGNet(tx.shape[1], 3, c.get('window_standardization', False)) if temporal else
             FeatureNetwork('log_covariance', 20, 3, pre.parameters, c['hidden_units'], c['dropout']))
    if initial_state is not None:
        model.load_state_dict(initial_state)
        model.head[-1].reset_parameters()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['learning_rate'], weight_decay=c['weight_decay'])
    groups = defaultdict(list)
    for i, row in enumerate(tr):
        groups[row['trial']].append(i)
    def balance_key(row):
        return (row['subject'], row['label']) if c.get('balance_subjects') else row['label']
    class_cues = Counter(balance_key(tr[g[0]]) for g in groups.values())
    y = np.array([r['label'] for r in tr], dtype=np.int64)
    best, state, best_epoch, wait, history = None, None, 0, 0, []
    for epoch in range(1, c['epochs']+1):
        _check_cancel(cancel)
        chosen, weights = [], []
        for trial in sorted(groups):
            g = groups[trial]
            draw = rng.choice(g, min(c['windows_per_cue'], len(g)), replace=False)
            chosen.extend(draw)
            weights.extend([1/(class_cues[balance_key(tr[g[0]])]*len(draw))]*len(draw))
        chosen, weights = np.array(chosen), np.array(weights, dtype=np.float32)
        order = rng.permutation(len(chosen))
        model.train()
        loss_sum, weight_sum = 0., 0.
        for a in range(0, len(order), c['batch_size']):
            _check_cancel(cancel)
            positions = order[a:a+c['batch_size']]
            idx = chosen[positions]
            w = torch.from_numpy(weights[positions]).to(device)
            inputs = torch.from_numpy(tx[idx]).to(device)
            if temporal:
                gain = torch.empty_like(inputs[:, :, :1]).uniform_(.9, 1.1)
                inputs = inputs * gain
            targets = torch.from_numpy(y[idx]).to(device)
            optimizer.zero_grad()
            losses = nn.functional.cross_entropy(model(inputs), targets, reduction='none')
            loss = (losses*w).sum()/w.sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            loss_sum += float((losses.detach()*w).sum())
            weight_sum += float(w.sum())
        m = _metrics(_probabilities(model, vx), vr)
        cue = m['deployment_policy']
        score = ((-m['cue_nll'],) if temporal else
                 (cue['balanced_accuracy'], min(cue['recall'].values()), -m['cue_nll']))
        row = dict(epoch=epoch, sampled_window_loss=loss_sum/weight_sum,
                   tuning_cue_balanced_accuracy=m['cue_trials']['balanced_accuracy'],
                   tuning_deployment_balanced_accuracy=cue['balanced_accuracy'],
                   tuning_deployment_coverage=cue['coverage'], tuning_cue_nll=m['cue_nll'])
        if epoch == 1 or epoch % 10 == 0:
            tm = _metrics(_probabilities(model, tx), tr)
            row.update(train_cue_balanced_accuracy=tm['cue_trials']['balanced_accuracy'],
                       train_deployment_balanced_accuracy=tm['deployment_policy']['balanced_accuracy'],
                       train_deployment_coverage=tm['deployment_policy']['coverage'], train_cue_nll=tm['cue_nll'])
        history.append(row)
        if best is None or score > best:
            state = {name: tensor.detach().to('cpu', copy=True) for name, tensor in model.state_dict().items()}
            best, best_epoch, wait = score, epoch, 0
        else:
            wait += 1
        if progress and (epoch == 1 or epoch % 20 == 0):
            progress(f"Fitting three-class seed {seed}, epoch {epoch}. Tuning cue loss {m['cue_nll']:.3f}.")
        if wait >= c['patience']:
            break
    model.load_state_dict(state)
    return state, dict(seed=seed, selected_epoch=best_epoch, epochs=history)


def fit_personal(raw, timestamps, events, channels, output_dir, config=None,
                 cancel=None, progress=None, segments=None, device='cpu', *, reserved_trials=None):
    """Create a new model from designated development cues and return its diagnostics.

    The directory must be new or empty. Cancellation and fitting failures never
    publish a partial model. For archived recordings, pass the reserved tuning
    cue IDs explicitly and keep the final test recording outside these inputs.
    """
    c = _configuration(config)
    _check_cancel(cancel)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Personal model output directory is nonempty; use a new model version.')
    if progress:
        progress('Checking complete calibration cues, gaps and EEG quality.')
    X, rows, counts = prepare_calibration(raw, timestamps, events, channels, c, segments)
    train, tuning = split_cues(rows, c['seed'], c['min_cues_per_class'], c['tuning_fraction'], events=events,
                              reserved_trials=reserved_trials)
    tr, vr = [rows[i] for i in train], [rows[i] for i in tuning]
    y = np.array([r['label'] for r in tr], dtype=np.int64)
    weights = trial_class_weights(y, tr)
    _check_cancel(cancel)
    temporal = c['arch'] == 'temporal_cnn'
    if temporal:
        x_train = X[train]
        mean = np.average(x_train.mean(axis=2), axis=0, weights=weights).astype(np.float32)
        variance = np.average(np.mean((x_train-mean[:, None])**2, axis=2), axis=0, weights=weights)
        std = np.maximum(np.sqrt(variance), 1e-9).astype(np.float32)
        tx = (x_train-mean[:, None])/(std[:, None]+1e-8)
        vx = (X[tuning]-mean[:, None])/(std[:, None]+1e-8)
        pre = None
    else:
        cov = band_covariances(X, c['sfreq'])
        pre = FeaturePreprocessor.fit(cov[train], weights, 'log_covariance', c['covariance_shrinkage'])
        tx, vx = pre.transform(cov[train]), pre.transform(cov[tuning])
        mean, std = np.zeros(20, dtype=np.float32), np.ones(20, dtype=np.float32)
    torch.set_num_threads(int(c['threads']))
    states, histories = [], []
    hidden_units = c['hidden_units']
    if c['linear_c'] is not None:
        from sklearn.linear_model import LogisticRegression
        if progress:
            progress('Fitting regularized linear classifier on training cues.')
        n_cues = len({r['trial'] for r in tr})
        model = LogisticRegression(C=c['linear_c'], max_iter=500)
        model.fit(tx, y, sample_weight=weights * (n_cues / weights.sum()))
        states.append({'head.weight': torch.tensor(model.coef_, dtype=torch.float32),
                       'head.bias': torch.tensor(model.intercept_, dtype=torch.float32)})
        hidden_units = 0
    else:
        for seed in c['ensemble_seeds']:
            _check_cancel(cancel)
            state, history = _fit_member(tx, tr, vx, vr, pre, c, seed, cancel, progress, device)
            states.append(state)
            histories.append(history)
    payload = dict(arch=c['arch'], n_classes=3, n_channels=20, n_times=X.shape[-1],
                   member_states=states, train_mean=mean, train_std=std)
    if temporal:
        runner = TemporalEEGEnsemble(20, 3, states).to(device)
        train_prob, tuning_prob = _probabilities(runner, tx), _probabilities(runner, vx)
    else:
        payload.update(family='log_covariance', hidden_units=hidden_units, dropout=c['dropout'],
                       preprocessing=pre.parameters, post_logit_bias=np.zeros(3, dtype=np.float32))
        runner = FeatureEnsembleRunner(payload, device=device)
        train_prob, tuning_prob = runner.from_covariances(cov[train]), runner.from_covariances(cov[tuning])
    cue_manifest = [dict(trial=str(e['trial']), label=_canonical(e['label']),
                         onset=float(e['onset']), end=float(e['end']),
                         **({'calibration_block':int(e['calibration_block'])} if 'calibration_block' in e else {}))
                    for e in events if e.get('complete', False)]
    input_hashes = dict(
        raw_volts_float32_sha256=hashlib.sha256(np.asarray(raw, dtype=np.float32).tobytes()).hexdigest(),
        timestamps_float64_sha256=hashlib.sha256(np.asarray(timestamps, dtype=np.float64).tobytes()).hexdigest(),
        completed_cue_manifest_sha256=hashlib.sha256(json.dumps(cue_manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest())
    if segments is not None:
        input_hashes['segments_sha256'] = hashlib.sha256(np.asarray(segments, dtype=np.int64).tobytes()).hexdigest()
    diagnostics = dict(calibration_only=True, final_validation_performed=False, counts=counts,
        training=_metrics(train_prob, tr), tuning=_metrics(tuning_prob, vr), members=histories,
        train_cues=sorted({r['trial'] for r in tr}), tuning_cues=sorted({r['trial'] for r in vr}),
        train_cues_per_class={name:len({r['trial'] for r in tr if r['label'] == i}) for i, name in enumerate(CLASSES)},
        tuning_cues_per_class={name:len({r['trial'] for r in vr if r['label'] == i}) for i, name in enumerate(CLASSES)},
        config=c, input_hashes=input_hashes, completed_cue_manifest=cue_manifest,
        warning='Calibration tuning is development evidence. Collect NEW frozen-model validation cues; no accuracy is fabricated or inferred for them.')
    declared_blocks = any('calibration_block' in e for e in events)
    if reserved_trials is not None:
        diagnostics['split_scope'] = 'Explicit whole blocks reserved from archived recordings before quality filtering.'
    elif declared_blocks:
        diagnostics['split_scope'] = 'Later original calibration blocks reserved before quality filtering.'
    else:
        diagnostics['split_scope'] = 'One chronological cut of complete accepted calibration cues; no interleaving.'
    diagnostics['train_tuning_gap'] = (diagnostics['training']['deployment_policy']['balanced_accuracy']-
                                       diagnostics['tuning']['deployment_policy']['balanced_accuracy'])
    diagnostics['raw_train_tuning_gap'] = (diagnostics['training']['cue_trials']['balanced_accuracy']-
                                           diagnostics['tuning']['cue_trials']['balanced_accuracy'])
    diagnostics['tuning_abstention_rate'] = 1-diagnostics['tuning']['deployment_policy']['coverage']
    diagnostics['split_details'] = dict(
        latest_training_cue_end=max(r['cue_end'] for r in tr),
        earliest_tuning_cue_onset=min(r['onset'] for r in vr),
        chronology=('Whole cues held out across archived recordings.' if reserved_trials is not None else
                    'Every training cue precedes every tuning cue.'))
    if reserved_trials is not None:
        diagnostics['split_details'].update(reserved_cue_ids=sorted(reserved_trials), reserved_before_quality_filtering=True)
    elif declared_blocks:
        declared_blocks = sorted({int(e['calibration_block']) for e in events})
        n_reserved = max(2, int(np.ceil(len(declared_blocks)*c['tuning_fraction'])))
        diagnostics['split_details'].update(
            original_block_ids=declared_blocks,
            training_block_ids=declared_blocks[:-n_reserved],
            reserved_tuning_block_ids=declared_blocks[-n_reserved:],
            reserved_before_quality_filtering=True)
    classifier = 'regularized_linear' if hidden_units == 0 else 'neural_ensemble'
    early_stopping = ('Fixed L2 regularization. Tuning cues are held out.' if hidden_units == 0 else
                      'Tuning deployed-cue BA, minimum class recall, then raw cue NLL.')
    if temporal:
        classifier = c['arch']
        early_stopping = 'Lowest cue log loss on the reserved tuning blocks.'
    contract = dict(schema_version=2, task_mode='multiclass', arch=c['arch'],
        role='personal_threeclass_model', classes=CLASSES, channels=list(channels), sfreq=256,
        n_times=payload['n_times'], normalization='fixed' if temporal else 'feature_train_only',
        filter=dict(mode='causal', fmin=8., fmax=30., order=4),
        artifact_limit_v=c['artifact_limit_v'], min_signal_std_v=c['min_signal_std_v'],
        policy=dict(POLICY), verified=True,
        verified_meaning='Personal calibration provenance/preprocessing documented; prospective accuracy unvalidated.',
        initialization='scratch', ensemble_seeds=[] if hidden_units == 0 else list(c['ensemble_seeds']),
        classifier=classifier,
        participant=str(c.get('participant', 'self')),
        calibration_run_id=str(c.get('calibration_run_id', '')),
        calibration_input_hashes=input_hashes,
        calibration_split=diagnostics['split_scope'],
        early_stopping=early_stopping,
        final_validation_performed=False, training_cues=diagnostics['train_cues'],
        tuning_cues=diagnostics['tuning_cues'], class_bias_calibration='None: zero offsets for this new person.')
    _check_cancel(cancel)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output/'personal_threeclass.pt'
    contract_path = Path(str(checkpoint)+'.json')
    with tempfile.TemporaryDirectory(prefix='.fit-', dir=output) as temporary:
        temp = Path(temporary)
        torch.save(payload, temp/'model.pt')
        contract['sha256'] = hashlib.sha256((temp/'model.pt').read_bytes()).hexdigest()
        (temp/'model.pt.json').write_text(json.dumps(contract, indent=2), encoding='utf-8')
        (temp/'training.json').write_text(json.dumps(diagnostics, indent=2), encoding='utf-8')
        _check_cancel(cancel)
        # The contract is installed last: incomplete exports cannot be loaded.
        (temp/'model.pt').replace(checkpoint)
        (temp/'training.json').replace(output/'training.json')
        (temp/'model.pt.json').replace(contract_path)
    if progress:
        progress('Personal three-class model ready. Collect new cues to validate the frozen model.')
    return dict(checkpoint=str(checkpoint.resolve()), contract=str(contract_path.resolve()),
                diagnostics=diagnostics)
