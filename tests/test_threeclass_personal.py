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
    prepare_calibration, split_cues, fit_personal)
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


def test_fitting_uses_train_only_statistics_and_round_trips(calibration,tmp_path,monkeypatch):
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
    result = fit_personal(raw,ts,events,CHANNELS,tmp_path/'model',config)
    assert observed == [len(tr)]
    contract = json.loads(open(result['contract']).read())
    assert contract['classes'] == CLASSES and contract['policy'] == POLICY
    assert contract['participant'] == 'synthetic_fixture' and not contract['final_validation_performed']
    assert contract['calibration_run_id'] == 'fixture_calibration'
    assert not set(contract['training_cues']) & set(contract['tuning_cues'])
    assert contract['sha256'] == hashlib.sha256(open(result['checkpoint'],'rb').read()).hexdigest()
    payload = torch.load(result['checkpoint'],weights_only=False)
    expected = FeatureEnsembleRunner(payload).predict(X[:1])[0]
    decoder = MulticlassDecoder(result['checkpoint'])
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
