"""Prospective personal calibration must keep independent cues and gaps intact."""
import copy
import hashlib
import json
import threading

import numpy as np
import pytest
import torch
from scipy.signal import butter, sosfilt

from threeclass_personal import (CLASSES, POLICY, CalibrationCancelled,
    prepare_calibration, split_cues, fit_personal, _metrics)
from threeclass_decisions import cue_decision, diagnostic_from_probabilities
from multiclass_features import FeaturePreprocessor, FeatureEnsembleRunner, band_covariances
from multiclass_decoder import MulticlassDecoder

CHANNELS = ['Fp1','Fp2','F7','F3','Fz','F4','F8','T3','C3','Cz',
            'C4','T4','T5','P3','Pz','P4','T6','O1','POz','O2']


@pytest.fixture(scope='module')
def calibration():
    rng = np.random.default_rng(23)
    n = 8*256
    raw = []
    events = []
    for cue in range(24):
        label = cue % 3
        t = np.arange(n)/256
        wave = (4e-6*np.sin(2*np.pi*(10+label*6)*t)[None, :]*
                np.linspace(.6,1.4,20)[:,None]+rng.normal(0,2e-6,(20,n)))
        raw.append(wave.astype(np.float32))
        events.append(dict(trial=f'cal-{cue:02d}', label=CLASSES[label],
                           onset=100.+cue*8, end=108.+cue*8, complete=True))
    raw = np.concatenate(raw, axis=1)
    return raw, 100.+np.arange(raw.shape[1])/256, events


def test_whole_cue_split_excludes_incomplete_and_overlapping_copies(calibration):
    raw, ts, events = calibration
    X, rows, counts = prepare_calibration(raw, ts, events, CHANNELS)
    train, tune = split_cues(rows)
    train_cues = {rows[i]['trial'] for i in train}
    tune_cues = {rows[i]['trial'] for i in tune}
    assert not train_cues & tune_cues
    assert len(train_cues) == 18 and len(tune_cues) == 6
    assert max(rows[i]['cue_end'] for i in train) <= min(rows[i]['onset'] for i in tune)
    assert counts['clean_cues_per_class'] == {c:8 for c in CLASSES}
    for cue in train_cues:
        assert all(i in train for i,r in enumerate(rows) if r['trial'] == cue)
    incomplete = copy.deepcopy(events)
    incomplete[0]['complete'] = False
    _, remaining, qa = prepare_calibration(raw, ts, incomplete, CHANNELS)
    assert not any(r['trial'] == incomplete[0]['trial'] for r in remaining)
    assert qa['incomplete_cues'] == 1
    with pytest.raises(ValueError, match='eight|at least 8'):
        split_cues(remaining)


def test_filter_and_windows_restart_at_segments(calibration):
    raw, ts, events = calibration
    boundary = 10*256  # two seconds into the second cue
    segments = np.zeros(len(ts), dtype=int)
    segments[boundary:] = 1
    X, rows, qa = prepare_calibration(raw, ts, events, CHANNELS, segments=segments)
    assert qa['gap_crossing_windows'] > 0 and qa['filter_segments'] == 2
    for r in rows:
        assert not r['start'] < boundary < r['end']
    expected = sosfilt(butter(4,[8,30],fs=256,btype='bandpass',output='sos'), raw[:,boundary:],axis=1)
    after = next(i for i,r in enumerate(rows) if r['trial'] == events[1]['trial'])
    row = rows[after]
    assert row['start'] >= boundary+128
    np.testing.assert_allclose(X[after], expected[:,row['start']-boundary:row['end']-boundary],atol=1e-11)
    changed = raw.copy()
    changed[:,:boundary] *= 50
    X2, rows2, _ = prepare_calibration(changed,ts,events,CHANNELS,segments=segments)
    match = next(i for i,r in enumerate(rows2) if r['start']==row['start'])
    np.testing.assert_array_equal(X[after],X2[match])


def test_nonfinite_backward_timestamps_and_label_conflicts_fail(calibration):
    raw, ts, events = calibration
    bad = raw.copy();bad[0,10] = np.nan
    with pytest.raises(ValueError,match='nonfinite'):
        prepare_calibration(bad,ts,events,CHANNELS)
    backwards = ts.copy();backwards[10] = backwards[9]
    with pytest.raises(ValueError,match='timestamps'):
        prepare_calibration(raw,backwards,events,CHANNELS)
    bad_events = copy.deepcopy(events);bad_events[0]['label']='move'
    with pytest.raises(ValueError,match='Unknown'):
        prepare_calibration(raw,ts,bad_events,CHANNELS)


@pytest.mark.parametrize('device', [
    'cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')),
])
def test_fitting_uses_train_only_statistics_and_round_trips(calibration,tmp_path,monkeypatch,device):
    raw, ts, events = calibration
    X, rows, _ = prepare_calibration(raw,ts,events,CHANNELS)
    tr, va = split_cues(rows)
    expected_cov = band_covariances(X[tr])
    original_fit = FeaturePreprocessor.fit
    observed = []
    def guarded(cov,weights,*a,**kw):
        np.testing.assert_array_equal(cov,expected_cov)
        observed.append(len(cov))
        return original_fit(cov,weights,*a,**kw)
    monkeypatch.setattr(FeaturePreprocessor,'fit',guarded)
    config = dict(epochs=2,patience=2,ensemble_seeds=[42],threads=1,
                  participant='synthetic_fixture',calibration_run_id='fixture_calibration')
    result = fit_personal(raw,ts,events,CHANNELS,tmp_path/'model',config,device=device)
    assert observed == [len(tr)]
    diagnostics = result['diagnostics']
    assert diagnostics['split_details']['latest_training_cue_end'] <= diagnostics['split_details']['earliest_tuning_cue_onset']
    assert 'deployment_policy' in diagnostics['training'] and 'deployment_policy' in diagnostics['tuning']
    assert diagnostics['train_tuning_gap'] == pytest.approx(
        diagnostics['training']['deployment_policy']['balanced_accuracy']-diagnostics['tuning']['deployment_policy']['balanced_accuracy'])
    assert 'tuning_deployment_balanced_accuracy' in diagnostics['members'][0]['epochs'][0]
    contract = json.loads(open(result['contract']).read())
    assert contract['classes'] == CLASSES and contract['policy'] == POLICY
    assert contract['participant'] == 'synthetic_fixture' and not contract['final_validation_performed']
    assert contract['calibration_run_id'] == 'fixture_calibration'
    assert not set(contract['training_cues']) & set(contract['tuning_cues'])
    assert contract['sha256'] == hashlib.sha256(open(result['checkpoint'],'rb').read()).hexdigest()
    payload = torch.load(result['checkpoint'],weights_only=False)
    assert all(tensor.device.type == 'cpu' for state in payload['member_states'] for tensor in state.values())
    expected = FeatureEnsembleRunner(payload).predict(X[:1])[0]
    decoder = MulticlassDecoder(result['checkpoint'],device=device)
    assert all(param.device == torch.device(device) for member in decoder.feature_runner.members for param in member.parameters())
    # The trained filter has continuous state; use the exact causal window's raw
    # prefix and select the corresponding decoder sample endpoint.
    raw_end = rows[0]['end']
    outputs = decoder.process_chunk_all(raw[:,:raw_end])
    # First prepared window begins .5s after acquisition/cue onset, aligned to
    # decoder .25s stride, so its ending corresponds to an actual streamed update.
    actual = next(o for o in outputs if o['sample_end'] == raw_end)
    np.testing.assert_allclose(actual['probabilities_raw'],expected,atol=1e-6)
    with pytest.raises(ValueError,match='nonempty'):
        fit_personal(raw,ts,events,CHANNELS,tmp_path/'model',config)


def test_cancellation_publishes_no_model(calibration,tmp_path):
    raw,ts,events = calibration
    cancel = threading.Event();cancel.set()
    with pytest.raises(CalibrationCancelled):
        fit_personal(raw,ts,events,CHANNELS,tmp_path/'cancelled',cancel=cancel)
    assert not (tmp_path/'cancelled').exists()
    cancel.clear()
    def progress(message):
        if 'epoch' in message:
            cancel.set()
    with pytest.raises(CalibrationCancelled):
        fit_personal(raw,ts,events,CHANNELS,tmp_path/'during_fit',
                     dict(epochs=3,patience=3,ensemble_seeds=[42],threads=1),cancel=cancel,progress=progress)
    assert not list(tmp_path.rglob('*.pt'))


def block_records(n_blocks=12):
    events, rows = [], []
    for block in range(n_blocks):
        for label, name in enumerate(CLASSES):
            trial = f'block-{block}-class-{label}'
            onset = 100.+(block*3+label)*8.
            events.append(dict(trial=trial, label=name, onset=onset, end=onset+8,
                               complete=True, calibration_block=block))
            for window in range(4):
                rows.append(dict(trial=trial, label=label, start=int((onset-100)*256)+window*64,
                    end=int((onset-100)*256)+window*64+1024, onset=onset,
                    cue_end=onset+8, calibration_block=block, run='synthetic_calibration'))
    return events, rows


def test_declared_blocks_reserve_later_tuning_before_any_quality_exclusions():
    events, rows = block_records()
    train, tune = split_cues(rows, events=events)
    assert {rows[i]['calibration_block'] for i in train} == set(range(9))
    assert {rows[i]['calibration_block'] for i in tune} == {9, 10, 11}
    assert max(rows[i]['cue_end'] for i in train) <= min(rows[i]['onset'] for i in tune)
    # Losing one later tuning cue leaves two clean cues for that class.
    remaining = [r for r in rows if not (r['calibration_block'] == 11 and r['label'] == 2)]
    train2, tune2 = split_cues(remaining, events=events)
    assert {remaining[i]['calibration_block'] for i in train2} == set(range(9))
    assert len({remaining[i]['trial'] for i in tune2 if remaining[i]['label'] == 2}) == 2
    # Losing a second later cue must fail rather than silently move block8
    # from training to tuning, despite ten clean cues of the class remaining.
    insufficient = [r for r in remaining if not (r['calibration_block'] == 10 and r['label'] == 2)]
    with pytest.raises(ValueError, match='Reserved calibration blocks'):
        split_cues(insufficient, events=events)
    with pytest.raises(ValueError, match='Original calibration events'):
        split_cues(rows)


def test_block_provenance_reaches_prepared_windows(calibration):
    raw, ts, original = calibration
    events = [dict(e, calibration_block=i//3) for i, e in enumerate(original)]
    _, rows, _ = prepare_calibration(raw, ts, events, CHANNELS)
    assert {r['calibration_block'] for r in rows} == set(range(8))
    train, tune = split_cues(rows, events=events)
    assert {rows[i]['calibration_block'] for i in train} == set(range(6))
    assert {rows[i]['calibration_block'] for i in tune} == {6, 7}


def test_archived_fit_reserves_complete_blocks_without_requiring_later_tuning():
    events, rows = block_records()
    reserved = [e['trial'] for e in events if e['calibration_block'] in {1, 2}]
    train, tune = split_cues(rows, events=events, reserved_trials=reserved)
    assert {rows[i]['trial'] for i in tune} == set(reserved)
    assert {rows[i]['trial'] for i in train} == {e['trial'] for e in events} - set(reserved)
    with pytest.raises(ValueError, match='whole calibration blocks'):
        split_cues(rows, events=events, reserved_trials=reserved[:-1])
    with pytest.raises(ValueError, match='completed recording plan'):
        split_cues(rows, events=events, reserved_trials=reserved + ['missing'])


def test_legacy_split_refuses_to_interleave_segregated_class_times():
    records = []
    for label in range(3):
        for cue in range(8):
            records.append(dict(trial=f'{label}-{cue}', label=label,
                                onset=(label*8+cue)*8., cue_end=(label*8+cue+1)*8., start=(label*8+cue)*2048))
    with pytest.raises(ValueError, match='No chronological split'):
        split_cues(records)


def test_cue_decision_requires_clean_majority_and_rejects_exact_ties():
    rest = [dict(class_id=0, quality='ok') for _ in range(4)]
    assert cue_decision(rest) == dict(class_id=0, eligible_updates=4, accepted_updates=4)
    assert cue_decision(rest, eligible_updates=9)['class_id'] == -1
    grasp = [dict(class_id=1, quality='ok') for _ in range(4)]
    assert cue_decision(rest+grasp)['class_id'] == -1
    bad = [dict(class_id=1, quality='artifact') for _ in range(4)]
    assert cue_decision(rest+bad)['class_id'] == 0
    with pytest.raises(ValueError, match='Eligible update'):
        cue_decision(rest, eligible_updates=3)


def test_low_confidence_raw_argmax_success_is_not_deployed_success():
    probabilities, rows = [], []
    for label in range(3):
        for window in range(4):
            p = np.full(3, .3);p[label] = .4
            probabilities.append(p)
            rows.append(dict(trial=f'cue{label}', label=label, run='synthetic', start=window*64))
    p = np.array(probabilities)
    metrics = _metrics(p, rows)
    assert metrics['cue_trials']['balanced_accuracy'] == 1.
    deployed = metrics['deployment_policy']
    assert deployed['balanced_accuracy'] == 0 and deployed['coverage'] == 0
    assert deployed['n_cues'] == deployed['n_rejected'] == 3
    assert np.array(deployed['confusion_matrix'])[:,3].sum() == 3
    assert 'not full continuous validation' in deployed['scope']


def test_probability_averaging_cannot_replace_deployed_majority_votes():
    probabilities, rows = [], []
    for label in range(3):
        for window in range(7):
            p = np.full(3, .005)
            p[label] = .99
            if window >= 3:
                p = np.full(3, .01);p[label] = .46;p[(label+1)%3] = .53
            probabilities.append(p)
            rows.append(dict(trial=f'cue{label}', label=label, run='synthetic', start=window*64))
    p = np.array(probabilities)
    assert _metrics(p, rows)['cue_trials']['balanced_accuracy'] == 1.
    deployed = diagnostic_from_probabilities(p, rows, CLASSES)
    assert deployed['coverage'] == 1. and deployed['balanced_accuracy'] == 0.
