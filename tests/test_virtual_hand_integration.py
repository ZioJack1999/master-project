"""Integration tests for the 3D Virtual Hand standalone application and server routing."""
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import threading
import urllib.request
import urllib.error
import pytest

import bci_studio_gui as binary_app
import threeclass_studio as threeclass_app

ROOT = Path(__file__).resolve().parents[1]


def test_virtual_hand_files_exist():
    """Verify virtual_hand.html, three.min.js, and virtual_hand.js exist and are valid."""
    html_file = ROOT / 'virtual_hand.html'
    three_file = ROOT / 'static' / 'js' / 'three.min.js'
    hand_js_file = ROOT / 'static' / 'js' / 'virtual_hand.js'

    assert html_file.is_file(), "virtual_hand.html must exist in the root repository."
    assert three_file.is_file(), "static/js/three.min.js must exist."
    assert hand_js_file.is_file(), "static/js/virtual_hand.js must exist."

    html_content = html_file.read_text(encoding='utf-8')
    assert 'static/js/three.min.js' in html_content
    assert 'static/js/virtual_hand.js' in html_content
    assert 'VirtualHand3D' in html_content
    assert 'target-pill' in html_content
    assert 'conn-badge' in html_content or 'conn-dot' in html_content
    assert 'resetCamBtn' in html_content
    assert 'wireframeBtn' in html_content
    assert 'themeSelect' in html_content
    assert '/api/state' in html_content

    hand_js_content = hand_js_file.read_text(encoding='utf-8')
    assert 'class VirtualHand3D' in hand_js_content
    assert 'setTargetState' in hand_js_content
    assert 'resetCamera' in hand_js_content
    assert 'toggleWireframe' in hand_js_content
    assert 'setTheme' in hand_js_content
    assert 'volar_grasp' in hand_js_content
    assert 'finger_extension' in hand_js_content


def test_virtual_hand_served_by_binary_studio(tmp_path):
    """Verify bci_studio_gui HTTP server routes virtual_hand.html and static assets."""
    config = json.loads((ROOT / 'lab_config.json').read_text(encoding='utf-8-sig'))
    config['reports_dir'] = str(tmp_path / 'reports')
    studio = binary_app.Studio(config)
    handler = type('TestHandler', (binary_app.Handler,), {'studio': studio})
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base = f'http://127.0.0.1:{server.server_port}'
    try:
        # 1. virtual_hand.html route
        with urllib.request.urlopen(f'{base}/virtual_hand.html') as resp:
            assert resp.status == 200
            assert 'text/html' in resp.headers.get('Content-Type', '')
            content = resp.read().decode('utf-8')
            assert 'VirtualHand3D' in content
            assert resp.headers.get('Access-Control-Allow-Origin') == '*'

        # 2. virtual_hand clean alias
        with urllib.request.urlopen(f'{base}/virtual_hand') as resp:
            assert resp.status == 200
            assert 'text/html' in resp.headers.get('Content-Type', '')

        # 3. static/js/virtual_hand.js asset
        with urllib.request.urlopen(f'{base}/static/js/virtual_hand.js') as resp:
            assert resp.status == 200
            assert 'javascript' in resp.headers.get('Content-Type', '')
            js_content = resp.read().decode('utf-8')
            assert 'VirtualHand3D' in js_content

        # 4. static/js/three.min.js asset
        with urllib.request.urlopen(f'{base}/static/js/three.min.js') as resp:
            assert resp.status == 200
            assert 'javascript' in resp.headers.get('Content-Type', '')

        # 5. api/state with CORS
        with urllib.request.urlopen(f'{base}/api/state') as resp:
            assert resp.status == 200
            assert resp.headers.get('Access-Control-Allow-Origin') == '*'
            state = json.loads(resp.read().decode('utf-8'))
            assert 'classes' in state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
        studio.stop.set()


def test_virtual_hand_served_by_threeclass_studio(tmp_path):
    """Verify threeclass_studio HTTP server routes virtual_hand.html and static assets."""
    config = json.loads((ROOT / 'threeclass_config.json').read_text(encoding='utf-8-sig'))
    config['checkpoint'] = str(ROOT / config['checkpoint'])
    config['validation_cues_per_class'] = 1
    studio = threeclass_app.ThreeClassStudio(config, root=ROOT)
    handler = type('TestHandler', (threeclass_app.Handler,), {'studio': studio})
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base = f'http://127.0.0.1:{server.server_port}'
    try:
        # 1. virtual_hand.html route
        with urllib.request.urlopen(f'{base}/virtual_hand.html') as resp:
            assert resp.status == 200
            assert 'text/html' in resp.headers.get('Content-Type', '')
            content = resp.read().decode('utf-8')
            assert 'VirtualHand3D' in content
            assert resp.headers.get('Access-Control-Allow-Origin') == '*'

        # 2. virtual_hand clean alias
        with urllib.request.urlopen(f'{base}/virtual_hand') as resp:
            assert resp.status == 200
            assert 'text/html' in resp.headers.get('Content-Type', '')

        # 3. static/js/virtual_hand.js asset
        with urllib.request.urlopen(f'{base}/static/js/virtual_hand.js') as resp:
            assert resp.status == 200
            assert 'javascript' in resp.headers.get('Content-Type', '')
            js_content = resp.read().decode('utf-8')
            assert 'VirtualHand3D' in js_content

        # 4. api/state with CORS
        with urllib.request.urlopen(f'{base}/api/state') as resp:
            assert resp.status == 200
            assert resp.headers.get('Access-Control-Allow-Origin') == '*'
            state = json.loads(resp.read().decode('utf-8'))
            assert 'classes' in state
            assert state['classes'] == ['rest', 'volar_grasp', 'finger_extension']
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
        studio.stop_event.set()
