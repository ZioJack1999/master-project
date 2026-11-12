"""Real liblsl→30-field adapter→three-class runtime→HTTP/disconnect check."""
import hashlib
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer
import urllib.error
import urllib.request
import numpy as np
import pytest
from threeclass_studio import ThreeClassStudio, Handler, CLASSES

ROOT = Path(__file__).resolve().parents[1]


def test_real_lsl_threeclass_http_warmup_and_partial_calibration_disconnect(tmp_path):
    """Test transport/serialization, never assert synthetic classification accuracy."""
    config = json.loads((ROOT/'threeclass_config.json').read_text())
    name = 'ThreeClass_Transport_'+str(time.time_ns())
    config.update(stream_name=name, source_id=name, participant='transport_test',
                  reports_dir=str(tmp_path/'private_reports'))
    reference = ROOT/config['checkpoint']
    before_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
    sender = subprocess.Popen([sys.executable, str(Path(__file__).with_name('mock_threeclass_lsl.py')), name],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    other = subprocess.Popen([sys.executable, str(Path(__file__).with_name('mock_threeclass_lsl.py')), 'Unrelated_'+name],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    ready = queue.Queue()
    threading.Thread(target=lambda:ready.put(sender.stdout.readline().strip()), daemon=True).start()
    studio = ThreeClassStudio(config, root=ROOT)
    handler = type('TransportHandler', (Handler,), dict(studio=studio))
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    worker = threading.Thread(target=studio.worker, daemon=True)
    http = threading.Thread(target=server.serve_forever, daemon=True)
    base = f'http://127.0.0.1:{server.server_port}'

    def api(path='/api/state?cue_display=1', body=None):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(base+path, data=data,
            method='POST' if body is not None else 'GET', headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request, timeout=10.) as response:
            return json.load(response)

    try:
        assert ready.get(timeout=20.) == 'READY'
        worker.start()
        http.start()
        api('/api/connect', dict(participant='transport_test'))
        deadline = time.monotonic()+20.
        retry_at = time.monotonic()+2.
        state = api()
        connection_errors = []
        while not state['connected'] and time.monotonic() < deadline:
            if state.get('error') and not state.get('connecting') and time.monotonic() >= retry_at:
                connection_errors.append(state.get('error'))
                api('/api/connect', dict(participant='transport_test'))
                retry_at = time.monotonic()+2.
            time.sleep(.05)
            state = api()
        assert state['connected'], (state['error'], connection_errors)
        assert state['mode'] == 'live' and state['stream_name'] == name
        assert studio.adapter.indices == list(range(25,5,-1))
        np.testing.assert_array_equal(studio.adapter.scales, np.full(20, 1e-6))
        # The shipped participant08 model is explicitly not a personal fit.
        assert not state['model_ready'] and state['model_role'] == 'recorded_reference'
        with pytest.raises(urllib.error.HTTPError) as blocked:
            api('/api/validate', {})
        assert blocked.value.code == 409

        api('/api/start', {})
        deadline = time.monotonic()+8.
        state = api()
        while not studio.predictions and time.monotonic() < deadline:
            time.sleep(.05)
            state = api()
        assert studio.predictions, state
        state = api()
        assert studio.predictions[0]['sample_end'] == 1024
        assert state['quality'] == 'ok'
        probabilities = np.array(state['raw_probabilities'])
        assert probabilities.shape == (3,) and np.isfinite(probabilities).all()
        assert probabilities.sum() == pytest.approx(1., abs=1e-5)
        assert state['result'] is None
        assert not any(r.get('kind') == 'fresh_frozen_personal_validation' for r in studio.results)
        assert np.max(abs(np.concatenate(studio.raw, axis=1))) < 100e-6

        # Begin a real cue collection, then remove the outlet before it finishes.
        state = api('/api/calibrate', {})
        assert state['protocol_kind'] == 'calibration' and state['cue'] in CLASSES
        assert state['awaiting_cue_ack'] and not studio.protocol.events
        state = api('/api/cue-presented', dict(run_id=state['cue_run_id'], trial=state['cue_index']))
        assert not state['awaiting_cue_ack']
        deadline = time.monotonic()+2.
        while not studio.raw and time.monotonic() < deadline:
            time.sleep(.05)
            api()  # Keep visible-cue heartbeat alive while this test consumes it.
        assert studio.raw
        sender.terminate()
        sender.wait(timeout=5.)
        deadline = time.monotonic()+4.
        state = api()
        while (state['streaming'] or state['connected']) and time.monotonic() < deadline:
            time.sleep(.05)
            state = api()
        assert not state['streaming'] and not state['connected'], state
        assert not state['training'] and not state['model_ready']
        assert studio.protocol is None and state['decision'] == 'uncertain'
        assert state['result']['status'] == 'aborted'
        assert state['result']['kind'] == 'calibration'
        assert studio.events and not studio.events[-1]['complete']

        saved = Path(state['last_saved'])
        metadata = json.loads((saved/'metadata.json').read_text())
        events = json.loads((saved/'events.json').read_text())
        with np.load(saved/'eeg.npz', allow_pickle=False) as eeg:
            assert eeg['raw'].shape[0] == 20 and eeg['raw'].shape[1] > 0
            assert str(eeg['unit']) == 'V' and float(eeg['sfreq']) == 256.
            assert list(eeg['channels']) == config['channels']
            assert np.isfinite(eeg['raw']).all() and np.max(abs(eeg['raw'])) < 100e-6
        assert metadata['source'] == 'live' and metadata['unit'] == 'V'
        assert not events[-1]['complete']
        assert not any(r.get('kind') == 'personal_calibration_fit' for r in metadata['results'])
        assert not list((tmp_path/'private_reports').rglob('model_*'))
        assert hashlib.sha256(reference.read_bytes()).hexdigest() == before_hash
    finally:
        studio.cancel_fit.set()
        studio.stop_event.set()
        if worker.is_alive():
            worker.join(timeout=5.)
        if http.is_alive():
            server.shutdown()
            http.join(timeout=5.)
        server.server_close()
        if sender.poll() is None:
            sender.terminate()
            sender.wait(timeout=5.)
        sender.stdout.close()
        if other.poll() is None:
            other.terminate()
            other.wait(timeout=5.)
