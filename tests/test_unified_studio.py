import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer
import threading
import time

import pytest

from multiclass_lab_bundle import CLASS_SETS
from threeclass_studio import Handler
from unified_studio import UnifiedStudio


ROOT = Path(__file__).resolve().parents[1]


def test_switch_profiles_over_http_and_keep_recordings_separate(tmp_path):
    def config(name, reports):
        settings = json.loads((ROOT/name).read_text(encoding='utf-8-sig'))
        settings.update(checkpoint=str(ROOT/settings['checkpoint']),
                        reports_dir=str(tmp_path/reports), udp_port=0)
        return settings

    studio = UnifiedStudio(config('threeclass_config.json', 'threeclass'),
                           config('lab_config.json', 'binary'), root=tmp_path, demo=True)
    handler = type('TestHandler', (Handler,), {'studio': studio})
    with ThreadingHTTPServer(('127.0.0.1', 0), handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'

        def request(path, body=None):
            data = None if body is None else json.dumps(body).encode()
            with urlopen(Request(base+path, data=data, headers={'Content-Type': 'application/json'}),
                         timeout=10) as response:
                return json.load(response)

        try:
            initial = request('/api/state')
            assert initial['profile'] == 'threeclass' and initial['model_role'] == 'recorded_reference'
            assert list(initial['profiles']) == ['threeclass', 'binary', '5class', '7class', '9class']
            assert request('/api/start', {})['streaming']
            deadline = time.monotonic() + 3
            while not studio.engine.raw and time.monotonic() < deadline:
                time.sleep(.01)
            assert studio.engine.raw
            with pytest.raises(HTTPError, match='409'):
                request('/api/profile', {'profile': 'binary'})
            request('/api/stop', {})
            recorded = studio.engine
            previous_worker = studio.worker
            binary = request('/api/profile', {'profile': 'binary'})
            assert not previous_worker.is_alive()
            assert binary['classes'] == ['rest', 'right_hand_grasp']
            assert binary['mode'] == 'demo' and not binary['streaming']
            assert studio.engine.reports != recorded.reports
            metadata = json.loads((Path(recorded.state['last_saved'])/'metadata.json').read_text())
            assert metadata['classes'] == initial['classes'] and metadata['source'] == 'demo'
            with pytest.raises(HTTPError, match='409'):
                request('/api/start', {'profile_revision': initial['profile_revision']})

            for count, classes in CLASS_SETS.items():
                state = request('/api/profile', {'profile': f'{count}class'})
                assert state['classes'] == classes and not state['model_ready']
                assert studio.engine.reference_checkpoint is None
                assert studio.engine.reports == tmp_path/'session_reports'/f'{count}class'
            returned = request('/api/profile', {'profile': 'threeclass'})
            assert returned['model_sha256'] == initial['model_sha256']
            assert not returned['streaming'] and not studio.engine.raw
        finally:
            server.shutdown()
            thread.join(timeout=5)
            studio.close()
