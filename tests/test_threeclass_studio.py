"""Three-class lab controls, fresh validation and real transport integration."""
import copy
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import threeclass_studio as app
from threeclass_studio import ThreeClassProtocol, ThreeClassStudio, score_validation
from model_utils import digest
from test_threeclass_personal import calibration

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


@pytest.fixture
def studio(tmp_path):
    config = json.loads((ROOT/'threeclass_config.json').read_text())
    config.update(checkpoint=str(ROOT/config['checkpoint']), participant='synthetic_fixture',
                  validation_cues_per_class=1, epochs=2, patience=2,
                  ensemble_seeds=[42], threads=1)
    shutil.copy2(ROOT/'threeclass_studio.html', tmp_path/'threeclass_studio.html')
    now = [100.]
    instance = ThreeClassStudio(config, root=tmp_path, clock=lambda:now[0])
    return instance, now


def packet(label, quality='ok'):
    p = [0., 0., 0.]
    if label >= 0:
        p[label] = 1.
    return dict(class_id=label, quality=quality, probabilities_raw=p,
                probabilities_ema=p, class_name=app.CLASSES[label] if label >= 0 else 'uncertain')


def completed_protocol(reject_label=None):
    p = ThreeClassProtocol('validation', 100., 1)
    for event_label in list(p.labels):
        start = p.events[-1]['onset']
        for offset in [4.5, 4.75, 5., 5.25]:
            p.add(packet(-1 if event_label == reject_label else int(event_label)), start+offset)
        p.advance(start+8.)
    return p


def test_protocol_never_mixes_cues_or_invents_delayed_transitions():
    p = ThreeClassProtocol('validation', 100., 1)
    p.add(packet(0), 104.25)
    assert not p.predictions
    p.add(packet(0), 104.5)
    p.add(packet(0), 108.01)
    assert len(p.predictions) == 1
    p.advance(130.)
    assert p.index == 1 and not p.done
    assert p.events[0]['end'] == 130. and p.events[1]['onset'] == 130.
    assert all(int(np.sum(p.labels == c)) == 1 for c in range(3))


def test_primary_score_counts_rejections_as_errors_and_refuses_incomplete():
    p = completed_protocol(reject_label=1)
    result = score_validation(p, permutations=100)
    m = result['metrics']
    assert m['balanced_accuracy'] == pytest.approx(2/3)
    assert m['coverage'] == pytest.approx(2/3) and m['n_rejected'] == 1
    assert m['recall']['volar_grasp'] == 0 and m['confusion_matrix'][1][3] == 1
    assert len(m['confusion_matrix']) == 3 and len(m['confusion_matrix'][0]) == 4
    with pytest.raises(ValueError, match='complete fresh'):
        score_validation(ThreeClassProtocol('validation', 1., 1), permutations=10)


def test_reference_demo_and_new_participant_cannot_validate(studio):
    s, _ = studio
    with pytest.raises(ValueError, match='Reconnect EEG with the new participant'):
        s.action('/api/start', dict(participant='unapplied_id'))
    assert s.state['model_role'] == 'recorded_reference' and not s.state['model_ready']
    s.state['connected'] = True
    with pytest.raises(ValueError, match='personal three-class calibration'):
        s.action('/api/validate')
    s.action('/api/demo')
    s.state['model_ready'] = True
    with pytest.raises(ValueError, match='fresh live'):
        s.action('/api/calibrate')
    with pytest.raises(ValueError, match='fresh live'):
        s.action('/api/validate')
    s.results = [dict(kind='old_person')]
    s.action('/api/connect', dict(participant='different_person'))
    assert not s.state['model_ready'] and not s.results
    assert s.state['participant'] == 'different_person'


def test_hidden_display_timeout_and_stop_abort_without_fitting(studio):
    s, now = studio
    s.state['connected'] = True
    s.action('/api/calibrate')
    assert s.protocol is not None
    now[0] += 4.
    s._refresh(now[0])
    assert s.protocol is None and not s.state['streaming'] and not s.state['training']
    assert s.results[-1]['status'] == 'aborted'
    s.action('/api/calibrate')
    s.action('/api/cue-presented', dict(run_id=s.protocol.run_id, trial=1))
    s.action('/api/stop', dict(reason='Cue display hidden.'))
    assert s.events[-1]['complete'] is False and s.protocol is None
    assert s.results[-1]['reason'] == 'Cue display hidden.'


def test_model_contract_and_policy_are_frozen_for_validation(studio, tmp_path):
    s, now = studio
    local = tmp_path/'private_reference.pt'
    shutil.copy2(s.checkpoint, local)
    shutil.copy2(str(s.checkpoint)+'.json', str(local)+'.json')
    s.checkpoint = local
    p = completed_protocol()
    p.frozen = dict(model_sha256=digest(local), contract_sha256=digest(str(local)+'.json'),
                    policy=dict(s.decoder.policy.config))
    s.protocol = p
    s.state.update(model_ready=True, connected=True, protocol_kind='validation')
    s.last_ui_seen = now[0]
    s.decoder.policy.config['min_confidence'] = .01
    s._refresh(now[0])
    assert s.results[-1]['status'] == 'failed' and 'decision policy changed' in s.state['error']
    s.decoder.policy.config = dict(p.frozen['policy'])
    s.protocol = completed_protocol()
    s.protocol.frozen = p.frozen
    Path(str(local)+'.json').write_text('{}')
    s._refresh(now[0])
    assert s.results[-1]['status'] == 'failed' and 'model/contract changed' in s.state['error']


def test_personal_fit_then_new_frozen_validation_and_private_save(studio, calibration):
    s, now = studio
    raw, ts, events = calibration
    s.raw, s.times = [raw.copy()], [ts.copy()]
    s.segment_ids = [np.full(len(ts), s.segment)]
    s.events = copy.deepcopy(events)
    s._fit(SimpleNamespace(events=events, run_id='synthetic_calibration_run'))
    s.training_thread.join(45.)
    assert not s.training_thread.is_alive()
    assert s.state['model_ready'], s.state['error']
    assert s.state['model_role'] == 'personal'
    model_hash = s.state['model_sha256']
    # Restarting the application keeps a verified model available, but only an
    # explicit same-participant selection marks it ready for another test.
    restarted = ThreeClassStudio(s.config, root=s.root, clock=lambda:now[0])
    assert not restarted.state['model_ready']
    candidate = restarted.state['model_choices'][0]
    restarted.action('/api/load-model', dict(participant='synthetic_fixture', model_id=candidate['model_id']))
    assert restarted.state['model_ready'] and restarted.state['model_sha256'] == model_hash
    now[0] = 300.
    s.state['connected'] = True
    s.action('/api/validate')
    labels = list(s.protocol.labels)
    rng = np.random.default_rng(7)
    for label in labels:
        s.action('/api/cue-presented', dict(run_id=s.protocol.run_id, trial=s.protocol.index+1))
        onset = s.protocol.events[-1]['onset']
        for a in range(0, 8*256, 64):
            stamps = onset+np.arange(a, a+64)/256
            signal = rng.normal(0, 3e-6, (20, 64)).astype(np.float32)
            s.ingest(signal, stamps)
            now[0] = onset+(a+64)/256
            s.last_ui_seen = now[0]
            s._refresh(now[0])
    assert s.protocol is None and not s.state['streaming']
    assert s.state['result']['metrics']['n_cues'] == 3
    assert s.state['result']['source'] == 'live'
    assert s.state['result']['frozen']['model_sha256'] == model_hash
    assert digest(s.checkpoint) == model_hash
    folder = Path(s.state['last_saved'])
    saved = np.load(folder/'eeg.npz')
    assert saved['raw'].shape == (20, 24*256)
    assert saved['unit'] == 'V'
    metadata = json.loads((folder/'metadata.json').read_text())
    assert metadata['model_role'] == 'personal'
    assert metadata['results'][-1]['metrics']['n_cues'] == 3
    # Synthetic pipeline success does not establish participant accuracy.


def test_render_acknowledgement_precedes_labels_and_timeout_aborts(studio):
    s, now = studio
    s.state['connected'] = True
    result = s.action('/api/calibrate')
    assert result['awaiting_cue_ack'] and not s.protocol.events
    with pytest.raises(ValueError, match='Stale or unexpected'):
        s.action('/api/cue-presented', dict(run_id='old', trial=1))
    now[0] += .3
    result = s.action('/api/cue-presented', dict(run_id=s.protocol.run_id, trial=1))
    assert not result['awaiting_cue_ack'] and s.protocol.events[0]['onset'] == now[0]
    assert s.protocol.events[0]['display_acknowledged']
    now[0] += 8.
    s.last_ui_seen = now[0]
    s._refresh(now[0])
    assert s.state['awaiting_cue_ack'] and len(s.protocol.events) == 1
    now[0] += 3.1
    s.last_ui_seen = now[0]  # Polling is alive, but the next cue was never painted.
    s._refresh(now[0])
    assert s.protocol is None and s.results[-1]['status'] == 'aborted'
    assert 'acknowledgement timed out' in s.state['error']
    assert s.events[0]['complete']  # Completed preceding cue remains intact.


def test_all_site_quality_identifies_a_channel_outside_the_three_displayed_traces(tmp_path):
    c = json.loads((ROOT/'threeclass_config.custom20.example.json').read_text())
    c['checkpoint'] = str(ROOT/c['checkpoint'])
    s = ThreeClassStudio(c, root=tmp_path, clock=lambda:100.)
    assert s.decoder is None  # Physical custom20 sites differ from the reference.
    s.state['streaming'] = True
    raw = np.random.default_rng(29).normal(0, 3e-6, (20, 1280)).astype('float32')
    raw[s.channels.index('F1'), 1000] = .001
    s.ingest(raw, 100.+np.arange(1280)/256)
    assert s.state['quality'] == 'artifact'
    assert 'F1' in s.state['quality_details']['artifact_channels']
    assert s.state['quality_details']['peak_uV']['F1'] > 100
    assert len(s.state['quality_details']['peak_uV']) == 20
    assert s.state['decision'] == 'uncertain'
    s._reset_segment()
    assert not s.state['quality_details'] and s.state['quality'] == 'warming_up'


def test_electrode_spelling_case_does_not_change_reference_or_blank_the_scope(tmp_path):
    c = json.loads((ROOT/'threeclass_config.json').read_text())
    c.update(checkpoint=str(ROOT/c['checkpoint']), channels=[ch.lower() for ch in c['channels']])
    s = ThreeClassStudio(c, root=tmp_path, clock=lambda:100.)
    assert s.decoder is not None and s.state['model_role'] == 'recorded_reference'
    s.state['streaming'] = True
    s.ingest(np.random.default_rng(3).normal(0, 3e-6, (20, 1280)).astype('float32'),
             100.+np.arange(1280)/256)
    for channel in ['C3', 'Cz', 'C4']:
        assert len(s.waves[channel]) == 1024 and all(v is not None for v in s.waves[channel])


def test_pending_connection_is_discarded_after_source_switch(studio, monkeypatch):
    s, _ = studio
    entered, release = threading.Event(), threading.Event()
    inlet = SimpleNamespace(closed=False)
    inlet.close_stream = lambda:setattr(inlet, 'closed', True)
    def connect(*_):
        entered.set()
        assert release.wait(5.)
        return inlet, SimpleNamespace(metadata=dict(name='obsolete'))
    monkeypatch.setattr(app, 'connect', connect)
    s.action('/api/connect')
    worker = threading.Thread(target=s.worker)
    worker.start()
    try:
        assert entered.wait(5.)
        s.action('/api/demo')
        release.set()
        deadline = time.monotonic()+5.
        while not inlet.closed and time.monotonic() < deadline:
            time.sleep(.01)
        assert inlet.closed and s.inlet is None and s.state['mode'] == 'demo'
    finally:
        release.set()
        s.stop_event.set()
        worker.join(5.)


def test_demo_sample_clock_survives_a_scheduling_pause(studio):
    s, now = studio
    s.action('/api/demo')
    s.action('/api/start')
    worker = threading.Thread(target=s.worker)
    worker.start()
    try:
        deadline = time.monotonic()+3.
        while len(s.times) < 1 and time.monotonic() < deadline:
            time.sleep(.01)
        assert s.times
        now[0] += 2.
        deadline = time.monotonic()+3.
        while len(s.times) < 2 and time.monotonic() < deadline:
            time.sleep(.01)
        assert len(s.times) >= 2 and worker.is_alive()
        np.testing.assert_allclose(np.diff(np.concatenate(s.times)), 1/256, atol=1e-10)
    finally:
        s.stop_event.set()
        worker.join(5.)


def test_http_console_and_demo_gates(studio):
    s, _ = studio
    handler = type('TestHandler', (app.Handler,), dict(studio=s))
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        with urllib.request.urlopen(base+'/') as r:
            assert b'Finger Extension' in r.read()
        req = urllib.request.Request(base+'/api/demo', data=b'{}', headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req) as r:
            assert json.load(r)['mode'] == 'demo'
        req = urllib.request.Request(base+'/api/validate', data=b'{}')
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req)
        assert exc.value.code == 409
        with urllib.request.urlopen(base+'/api/state') as r:
            assert not json.load(r)['model_ready']
    finally:
        server.shutdown(); server.server_close(); thread.join(5.)
