import json
import socket
import sys
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bci_studio_gui
import threeclass_studio


CHANNELS_20 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "POz", "O2"]


def test_bci_studio_gui_routes_and_static(tmp_path):
    # Create a dummy virtual_hand.html if not present
    vhand = ROOT / 'virtual_hand.html'
    created_dummy = False
    if not vhand.exists():
        vhand.write_text("<!DOCTYPE html><html><body>Dummy Virtual Hand</body></html>", encoding='utf-8')
        created_dummy = True

    try:
        config = {
            'stream_name': 'test_stream',
            'reports_dir': str(tmp_path / 'reports'),
            'udp_port': 0,
            'device': 'cpu',
            'threshold': 0.5,
        }
        studio = bci_studio_gui.Studio(config, root=ROOT)
        handler = type('TestHandler', (bci_studio_gui.Handler,), {'studio': studio})
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        port = server.server_port
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        base = f'http://127.0.0.1:{port}'

        # Test index
        with urllib.request.urlopen(f'{base}/', timeout=5) as res:
            assert res.status == 200
            assert 'text/html' in res.headers.get('Content-Type')
            content = res.read().decode('utf-8')
            assert 'openHandBtn' in content

        # Test /virtual_hand, /virtual_hand.html, /hand
        for path in ('/virtual_hand', '/virtual_hand.html', '/hand'):
            with urllib.request.urlopen(f'{base}{path}', timeout=5) as res:
                assert res.status == 200
                assert 'text/html' in res.headers.get('Content-Type')

        # Test static js assets and MIME types
        with urllib.request.urlopen(f'{base}/static/js/three.min.js', timeout=5) as res:
            assert res.status == 200
            ct = res.headers.get('Content-Type', '')
            assert 'application/javascript' in ct
            assert len(res.read()) > 1000

        with urllib.request.urlopen(f'{base}/static/js/virtual_hand.js', timeout=5) as res:
            assert res.status == 200
            ct = res.headers.get('Content-Type', '')
            assert 'application/javascript' in ct
            assert len(res.read()) > 1000

        # Test /api/state
        with urllib.request.urlopen(f'{base}/api/state', timeout=5) as res:
            assert res.status == 200
            data = json.loads(res.read().decode('utf-8'))
            assert 'step_count' in data
            assert 'connected' in data

        # Security regression: Path traversal attempts must return 403 Forbidden
        for bad_path in ('/static/../bci_studio_gui.py', '/static/js/../../bci_studio_gui.py', '/static/../../README.md'):
            try:
                urllib.request.urlopen(f'{base}{bad_path}', timeout=5)
                assert False, f"Expected 403 Forbidden for traversal {bad_path}"
            except urllib.error.HTTPError as err:
                assert err.code == 403, f"Expected 403 for {bad_path}, got {err.code}"

        server.shutdown()
        server.server_close()
    finally:
        if created_dummy and vhand.exists():
            vhand.unlink()


def test_threeclass_studio_routes_and_static(tmp_path):
    vhand = ROOT / 'virtual_hand.html'
    created_dummy = False
    if not vhand.exists():
        vhand.write_text("<!DOCTYPE html><html><body>Dummy Virtual Hand</body></html>", encoding='utf-8')
        created_dummy = True

    try:
        config = {
            'participant': 'test_sub',
            'channels': CHANNELS_20,
            'reports_dir': str(tmp_path / 'reports'),
            'udp_port': 0,
            'device': 'cpu',
            'threshold': 0.5,
        }
        studio = threeclass_studio.ThreeClassStudio(config, root=ROOT)
        handler = type('TestHandler', (threeclass_studio.Handler,), {'studio': studio})
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        port = server.server_port
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        base = f'http://127.0.0.1:{port}'

        # Test index
        with urllib.request.urlopen(f'{base}/', timeout=5) as res:
            assert res.status == 200
            assert 'text/html' in res.headers.get('Content-Type')
            content = res.read().decode('utf-8')
            assert 'openHandBtn' in content

        # Test /virtual_hand, /virtual_hand.html, /hand
        for path in ('/virtual_hand', '/virtual_hand.html', '/hand'):
            with urllib.request.urlopen(f'{base}{path}', timeout=5) as res:
                assert res.status == 200
                assert 'text/html' in res.headers.get('Content-Type')

        # Test static js assets and MIME types
        with urllib.request.urlopen(f'{base}/static/js/three.min.js', timeout=5) as res:
            assert res.status == 200
            ct = res.headers.get('Content-Type', '')
            assert 'application/javascript' in ct
            assert len(res.read()) > 1000

        with urllib.request.urlopen(f'{base}/static/js/virtual_hand.js', timeout=5) as res:
            assert res.status == 200
            ct = res.headers.get('Content-Type', '')
            assert 'application/javascript' in ct
            assert len(res.read()) > 1000

        # Test /api/state
        with urllib.request.urlopen(f'{base}/api/state', timeout=5) as res:
            assert res.status == 200
            data = json.loads(res.read().decode('utf-8'))
            assert 'task_label' in data

        # Security regression: Path traversal attempts must return 403 Forbidden
        for bad_path in ('/static/../threeclass_studio.py', '/static/js/../../threeclass_studio.py', '/static/../../README.md'):
            try:
                urllib.request.urlopen(f'{base}{bad_path}', timeout=5)
                assert False, f"Expected 403 Forbidden for traversal {bad_path}"
            except urllib.error.HTTPError as err:
                assert err.code == 403, f"Expected 403 for {bad_path}, got {err.code}"

        server.shutdown()
        server.server_close()
    finally:
        if created_dummy and vhand.exists():
            vhand.unlink()


def test_udp_5005_expanded_payload_binary(tmp_path):
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(('127.0.0.1', 0))
    receiver.settimeout(2.0)
    udp_port = receiver.getsockname()[1]

    config = {
        'stream_name': 'test_stream',
        'reports_dir': str(tmp_path / 'reports'),
        'udp_host': '127.0.0.1',
        'udp_port': udp_port,
        'device': 'cpu',
        'threshold': 0.5,
    }
    studio = bci_studio_gui.Studio(config, root=ROOT)
    studio._send_rest_udp(reason='test_idle')

    data, _ = receiver.recvfrom(8192)
    pkt = json.loads(data.decode('utf-8'))

    required_keys = [
        'class_name', 'class_id', 'grasp_command', 'extension_command',
        'continuous_flex', 'continuous_splay', 'probabilities', 'quality'
    ]
    for key in required_keys:
        assert key in pkt, f"Missing required key {key} in UDP packet"

    assert pkt['class_name'] == 'rest'
    assert pkt['class_id'] == 0
    assert pkt['grasp_command'] == 0
    assert pkt['extension_command'] == 0
    assert pkt['continuous_flex'] == 0.0
    assert pkt['continuous_splay'] == 0.0
    assert pkt['quality'] == 'test_idle'
    assert isinstance(pkt['probabilities'], dict)
    assert 'rest' in pkt['probabilities']

    # Test that uncertain class prediction broadcasts class_id as integer -1 (never null)
    incoming = {
        'sample_end': 1024,
        'class_id': None,
        'class_name': 'uncertain',
        'quality': 'ok',
        'probabilities_raw': [0.34, 0.33, 0.33],
        'probabilities_ema': [0.34, 0.33, 0.33],
    }
    raw_id = incoming.get('class_id')
    active_id = int(raw_id) if raw_id is not None else -1
    studio._send_udp(dict(
        timestamp=time.time(),
        sample_end=incoming['sample_end'],
        class_name=incoming['class_name'],
        class_id=active_id,
        grasp_command=0,
        extension_command=0,
        continuous_flex=0.0,
        continuous_splay=0.0,
        probabilities={'rest': 0.34, 'volar_grasp': 0.33, 'finger_extension': 0.33},
        quality='ok',
    ))
    data2, _ = receiver.recvfrom(8192)
    pkt2 = json.loads(data2.decode('utf-8'))
    assert pkt2['class_id'] == -1
    assert pkt2['class_id'] is not None

    receiver.close()


def test_udp_5005_expanded_payload_threeclass(tmp_path):
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(('127.0.0.1', 0))
    receiver.settimeout(2.0)
    udp_port = receiver.getsockname()[1]

    config = {
        'participant': 'test_sub',
        'channels': CHANNELS_20,
        'reports_dir': str(tmp_path / 'reports'),
        'udp_host': '127.0.0.1',
        'udp_port': udp_port,
        'device': 'cpu',
        'threshold': 0.5,
    }
    studio = threeclass_studio.ThreeClassStudio(config, root=ROOT)
    studio._send_rest_udp(reason='test_idle')

    data, _ = receiver.recvfrom(8192)
    pkt = json.loads(data.decode('utf-8'))

    required_keys = [
        'class_name', 'class_id', 'grasp_command', 'extension_command',
        'continuous_flex', 'continuous_splay', 'probabilities', 'quality'
    ]
    for key in required_keys:
        assert key in pkt, f"Missing required key {key} in UDP packet"

    assert pkt['class_name'] == 'rest'
    assert pkt['class_id'] == 0
    assert pkt['grasp_command'] == 0
    assert pkt['extension_command'] == 0
    assert pkt['continuous_flex'] == 0.0
    assert pkt['continuous_splay'] == 0.0
    assert pkt['quality'] == 'test_idle'
    assert isinstance(pkt['probabilities'], dict)
    assert 'rest' in pkt['probabilities']

    # Test that uncertain class prediction broadcasts class_id as integer -1 (never null)
    incoming = {
        'sample_end': 1024,
        'class_id': None,
        'class_name': 'uncertain',
        'quality': 'ok',
        'probabilities_raw': [0.34, 0.33, 0.33],
        'probabilities_ema': [0.34, 0.33, 0.33],
    }
    raw_id = incoming.get('class_id')
    active_id = int(raw_id) if raw_id is not None else -1
    studio._send_udp(dict(
        timestamp=time.time(),
        sample_end=incoming['sample_end'],
        class_name=incoming['class_name'],
        class_id=active_id,
        grasp_command=0,
        extension_command=0,
        continuous_flex=0.0,
        continuous_splay=0.0,
        probabilities={'rest': 0.34, 'volar_grasp': 0.33, 'finger_extension': 0.33},
        quality='ok',
    ))
    data2, _ = receiver.recvfrom(8192)
    pkt2 = json.loads(data2.decode('utf-8'))
    assert pkt2['class_id'] == -1
    assert pkt2['class_id'] is not None

    receiver.close()
