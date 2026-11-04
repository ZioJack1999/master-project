"""Predeclared source-training/tuning-only comparison of low-rank neural heads."""
import argparse
from collections import defaultdict, Counter
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from multiclass_cohort_data import load_cohort
from multiclass_data import digest
from multiclass_features import FeaturePreprocessor, FeatureNetwork, FeatureEnsembleRunner
from multiclass_metrics import evaluate, cue_probabilities, trial_class_weights
from .lowrank import fit_projection, project, makehead, fold_state_dict

ROOT = Path(__file__).resolve().parents[2]


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def development_indices(records, subject, classes, train_sessions, validation_sessions):
    if not train_sessions or not validation_sessions or set(train_sessions)&set(validation_sessions):
        raise ValueError('Training and source tuning need nonempty disjoint sessions.')
    mapped = np.array([classes.index(r['label_name']) if r['label_name'] in classes else -1 for r in records])
    result = {}
    for name, sessions in [('train', train_sessions), ('validation', validation_sessions)]:
        result[name] = np.array([i for i, r in enumerate(records) if
                                r['subject'] == subject and r['session'] in sessions and mapped[i] >= 0], dtype=int)
        if set(mapped[result[name]]) != set(range(len(classes))):
            raise ValueError(f'{name} lacks at least one declared class.')
    if {records[i]['trial'] for i in result['train']} & {records[i]['trial'] for i in result['validation']}:
        raise ValueError('A cue leaked across development partitions.')
    return result, mapped


class SourceCovarianceGuard:
    """Disallow indexing any covariance outside declared source development."""
    def __init__(self, values, allowed):
        self.values, self.allowed = values, set(int(i) for i in allowed)
        self.read_indices = set()

    def __getitem__(self, indices):
        requested = np.asarray(indices)
        if requested.ndim != 1 or not np.issubdtype(requested.dtype, np.integer):
            raise ValueError('Only explicit source index arrays may read covariances.')
        selected = set(int(i) for i in requested)
        if not selected <= self.allowed:
            raise ValueError('Historical test/undeclared covariance access was blocked.')
        self.read_indices |= selected
        return self.values[requested]


def rows_for(records, indices, labels):
    return [dict(records[i], label=int(labels[i])) for i in indices]


def probabilities(model, features):
    model.eval()
    with torch.inference_mode():
        return np.concatenate([torch.softmax(model(torch.from_numpy(features[a:a+256])), 1).numpy()
                               for a in range(0, len(features), 256)])


def metrics(p, records, classes):
    y = np.array([r['label'] for r in records], dtype=np.int64)
    result = evaluate(p, y, records, classes)
    truth, cp, _ = cue_probabilities(p, records)
    result['cue_nll'] = float(-np.log(np.maximum(cp[np.arange(len(truth)), truth], 1e-12)).mean())
    return result


def fit_member(x, train_rows, vx, val_rows, classes, candidate, config, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = makehead(x.shape[1], candidate['hidden'], len(classes), candidate['dropout'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=candidate['weight_decay'])
    groups = defaultdict(list)
    for i, r in enumerate(train_rows):
        groups[r['trial']].append(i)
    counts = Counter(train_rows[g[0]]['label'] for g in groups.values())
    y = np.array([r['label'] for r in train_rows], dtype=np.int64)
    best, state, selected_epoch, wait, history = None, None, 0, 0, []
    for epoch in range(1, config['epochs']+1):
        chosen, weights = [], []
        for trial in sorted(groups):
            group = groups[trial]
            draw = rng.choice(group, min(len(group), config['windows_per_cue']), replace=False)
            chosen.extend(draw)
            weights.extend([1/(counts[train_rows[group[0]]['label']]*len(draw))]*len(draw))
        chosen, weights = np.asarray(chosen), np.asarray(weights, dtype=np.float32)
        order = rng.permutation(len(chosen))
        model.train()
        total_loss, total_weight = 0., 0.
        for a in range(0, len(order), config['batch_size']):
            positions = order[a:a+config['batch_size']]
            indices = chosen[positions]
            w = torch.from_numpy(weights[positions])
            optimizer.zero_grad()
            losses = nn.functional.cross_entropy(model(torch.from_numpy(x[indices])), torch.from_numpy(y[indices]), reduction='none')
            loss = (losses*w).sum()/w.sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total_loss += float((losses.detach()*w).sum())
            total_weight += float(w.sum())
        vm = metrics(probabilities(model, vx), val_rows, classes)
        cue = vm['cue_trials']
        score = (cue['balanced_accuracy'], min(cue['recall'].values()), -vm['cue_nll'])
        row = dict(epoch=epoch, sampled_window_loss=total_loss/total_weight,
                   validation_cue_ba=score[0], validation_cue_nll=vm['cue_nll'])
        if epoch == 1 or epoch % 10 == 0:
            tm = metrics(probabilities(model, x), train_rows, classes)
            row.update(train_cue_ba=tm['cue_trials']['balanced_accuracy'], train_cue_nll=tm['cue_nll'])
        history.append(row)
        if best is None or score > best:
            best, state, selected_epoch, wait = score, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            wait += 1
        if wait >= config['patience']:
            break
    model.load_state_dict(state)
    return model, dict(seed=seed, selected_epoch=selected_epoch, epochs=history)


def fit_case(case, candidate, cov, records, indices, labels, classes, channels, config, folder):
    result_path = folder/'result.json'
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        if saved['model_sha256'] != digest(folder/'model.pt'):
            raise ValueError('A frozen source-only model changed.')
        return saved
    folder.mkdir(parents=True, exist_ok=True)
    tr, va = indices['train'], indices['validation']
    train_cov, val_cov = np.asarray(cov[tr]), np.asarray(cov[va])
    tr_rows, va_rows = rows_for(records, tr, labels), rows_for(records, va, labels)
    weights = trial_class_weights(labels[tr], tr_rows)
    pre = FeaturePreprocessor.fit(train_cov, weights, 'log_covariance', config['covariance_shrinkage'])
    features, val_features = pre.transform(train_cov), pre.transform(val_cov)
    projection = fit_projection(features, tr_rows, candidate['components']) if candidate['components'] else None
    x, vx = project(features, projection), project(val_features, projection)
    states, histories, equivalence = [], [], []
    for seed in config['seeds']:
        model, history = fit_member(x, tr_rows, vx, va_rows, classes, candidate, config, seed)
        state = fold_state_dict(model.state_dict(), projection)
        deployed = FeatureNetwork('log_covariance', 20, len(classes), pre.parameters,
                                  candidate['hidden'], candidate['dropout'])
        deployed.load_state_dict(state)
        before, after = probabilities(model, vx), probabilities(deployed, val_features)
        delta = float(np.max(abs(before-after)))
        if not np.allclose(before, after, rtol=2e-5, atol=2e-6):
            raise ValueError('PCA folding did not preserve inference probabilities.')
        equivalence.append(delta)
        states.append(state)
        histories.append(history)
    payload = dict(arch='feature_ensemble', family='log_covariance', n_classes=len(classes),
        n_channels=20, n_times=1024, hidden_units=candidate['hidden'], dropout=candidate['dropout'],
        preprocessing=pre.parameters, member_states=states, train_mean=np.zeros(20, np.float32),
        train_std=np.ones(20, np.float32), post_logit_bias=np.zeros(len(classes), np.float32),
        source_only_lowrank_projection=projection, study_id=config['experiment_id'])
    model_path = folder/'model.pt'
    torch.save(payload, model_path)
    runner = FeatureEnsembleRunner(payload)
    training = metrics(runner.from_covariances(train_cov), tr_rows, classes)
    validation = metrics(runner.from_covariances(val_cov), va_rows, classes)
    contract = dict(schema_version=2, task_mode='multiclass', arch='feature_ensemble',
        classes=classes, channels=channels, sfreq=256, n_times=1024, normalization='feature_train_only',
        filter=dict(mode='causal', fmin=8., fmax=30., order=4), artifact_limit_v=100e-6,
        min_signal_std_v=1e-9, role='source_training_tuning_only_reference', verified=False,
        sha256=digest(model_path), study_id=config['experiment_id'], source_case=case['name'],
        target_test_scored=False, policy_status='No live policy calibration or prospective validation.')
    dump(str(model_path)+'.json', contract)
    source_hashes = dict(train_covariance_sha256=hashlib.sha256(train_cov.tobytes()).hexdigest(),
                         tuning_covariance_sha256=hashlib.sha256(val_cov.tobytes()).hexdigest())
    result = dict(case=case['name'], subject=case['subject'], classes=classes, candidate=candidate,
        train_cues=len({r['trial'] for r in tr_rows}), tuning_cues=len({r['trial'] for r in va_rows}),
        training=training, validation=validation,
        train_validation_gap=training['cue_trials']['balanced_accuracy']-validation['cue_trials']['balanced_accuracy'],
        projection={k:v for k, v in projection.items() if k not in ('center','components')} if projection else None,
        folding_max_probability_errors=equivalence, members=histories, model_sha256=digest(model_path),
        contract_sha256=digest(str(model_path)+'.json'), source_hashes=source_hashes,
        test_scored=False, no_test_covariance_access=True)
    dump(result_path, result)
    print(case['name'], candidate['name'], 'train %.3f tune %.3f' %
          (training['cue_trials']['balanced_accuracy'], validation['cue_trials']['balanced_accuracy']), flush=True)
    return result


def run(config_path):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    torch.set_num_threads(config['threads'])
    work = ROOT/config['output']
    work.mkdir(parents=True, exist_ok=True)
    source_files = [Path(__file__), Path(__file__).with_name('lowrank.py'), config_path]
    declaration = {p.name:digest(p) for p in source_files}
    frozen_path = work/'declaration_frozen.json'
    if frozen_path.exists():
        if json.loads(frozen_path.read_text()) != declaration:
            raise ValueError('The declared source-only experiment changed; preserve its artifacts.')
    else:
        dump(frozen_path, declaration)
    if (work/'study_results.json').exists():
        print('Reusing finalized source-only evidence.', flush=True)
        return json.loads((work/'study_results.json').read_text())
    raw_cov, records, manifest = load_cohort(ROOT/config['cohort'])
    cases, allowed = [], set()
    for case in config['cases']:
        classes = manifest['config']['class_sets'][case['class_set']]
        indices, labels = development_indices(records, case['subject'], classes, case['train'], case['validation'])
        allowed.update(indices['train']); allowed.update(indices['validation'])
        cases.append((case, classes, indices, labels))
    cov = SourceCovarianceGuard(raw_cov, allowed)
    results = []
    start = time.monotonic()
    for case, classes, indices, labels in cases:
        for candidate in config['candidates']:
            results.append(fit_case(case, candidate, cov, records, indices, labels, classes,
                manifest['channels'], config, work/case['name']/candidate['name']))
    summary = []
    for candidate in config['candidates']:
        matching = [r for r in results if r['candidate']['name'] == candidate['name']]
        summary.append(dict(candidate=candidate['name'],
            mean_validation_cue_ba=float(np.mean([r['validation']['cue_trials']['balanced_accuracy'] for r in matching])),
            worst_class_recall=float(min(v for r in matching for v in r['validation']['cue_trials']['recall'].values())),
            mean_validation_cue_nll=float(np.mean([r['validation']['cue_nll'] for r in matching])),
            mean_training_cue_ba=float(np.mean([r['training']['cue_trials']['balanced_accuracy'] for r in matching])),
            cases=len(matching)))
    selected = max(summary, key=lambda r:(r['mean_validation_cue_ba'],r['worst_class_recall'],-r['mean_validation_cue_nll']))
    report = dict(config=config, declaration_hashes=declaration, results=results, summary=summary,
                  selected_candidate=selected['candidate'], target_test_scored=False,
                  accessed_covariance_rows=len(cov.read_indices), allowed_source_rows=len(allowed),
                  total_cache_rows=len(records), duration_s=time.monotonic()-start,
                  warning='Source-only method development on previously exposed data. No test predictions, prospective accuracy, reliable improvement or population claim.')
    dump(work/'study_results.json', report)
    # Public report keeps every case/candidate outcome and selected-seed curves;
    # full three-seed histories remain private beside the frozen checkpoints.
    public = copy.deepcopy(report)
    for row in public['results']:
        row['members'] = [m for m in row['members'] if m['seed'] == 42]
    dump(ROOT/config['aggregate'], public)
    print(json.dumps(dict(summary=summary,selected_candidate=selected['candidate'],target_test_scored=False),indent=2),flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('experiment.json'))
    run(parser.parse_args().config)
