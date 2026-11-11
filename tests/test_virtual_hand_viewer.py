"""
Tests for virtual_hand_viewer.py and batch launcher shortcuts.
"""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import virtual_hand_viewer as vhv


def test_cli_help():
    """Verify that virtual_hand_viewer.py --help runs cleanly with returncode 0."""
    res = subprocess.run(
        [sys.executable, str(REPO_ROOT / "virtual_hand_viewer.py"), "--help"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "3D Virtual Hand Desktop Viewer" in res.stdout
    assert "--port" in res.stdout
    assert "--binary" in res.stdout
    assert "--threeclass" in res.stdout
    assert "--standalone" in res.stdout


def test_cli_dry_run_binary():
    """Verify --binary selects port 8052."""
    res = subprocess.run(
        [sys.executable, str(REPO_ROOT / "virtual_hand_viewer.py"), "--binary", "--dry-run"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "8052" in res.stdout
    assert "http://127.0.0.1:8052/virtual_hand" in res.stdout


def test_cli_dry_run_threeclass():
    """Verify --threeclass selects port 8051."""
    res = subprocess.run(
        [sys.executable, str(REPO_ROOT / "virtual_hand_viewer.py"), "--threeclass", "--dry-run"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "8051" in res.stdout
    assert "http://127.0.0.1:8051/virtual_hand" in res.stdout


def test_cli_custom_port():
    """Verify --port 8999 uses the specified port."""
    res = subprocess.run(
        [sys.executable, str(REPO_ROOT / "virtual_hand_viewer.py"), "--port", "8999", "--dry-run"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "8999" in res.stdout
    assert "http://127.0.0.1:8999/virtual_hand" in res.stdout


def test_port_detection_logic():
    """Verify port detection helper functions."""
    port, desc = vhv.detect_active_port("127.0.0.1", preferred_mode="binary")
    assert port == 8052

    port, desc = vhv.detect_active_port("127.0.0.1", preferred_mode="threeclass")
    assert port == 8051


def test_find_browser_executables():
    """Verify browser discovery executes without exceptions."""
    browsers = vhv.find_browser_executables()
    assert isinstance(browsers, dict)
    # On Windows, at least one browser (e.g. edge or zen or chrome) should be found
    if sys.platform == "win32":
        assert len(browsers) > 0


def test_launch_viewer_window_dry_run():
    """Verify launch_viewer_window executes cleanly in dry_run mode."""
    success = vhv.launch_viewer_window(
        url="http://127.0.0.1:8052/virtual_hand",
        browser_pref="auto",
        app_mode=True,
        dry_run=True,
    )
    assert success is True


def test_standalone_udp_bridge():
    """Verify StandaloneUDPBridge binds UDP and serves /api/state over HTTP."""
    bridge_http_port = 8098
    bridge_udp_port = 5098

    bridge = vhv.StandaloneUDPBridge(
        http_port=bridge_http_port,
        udp_port=bridge_udp_port,
        demo=False,
    )
    bridge.run()
    try:
        # Give server thread a moment to bind
        time.sleep(0.3)

        # Test initial /api/state
        req = urllib.request.Request(f"http://127.0.0.1:{bridge_http_port}/api/state")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["class_name"] == "rest"

        # Send test UDP datagram
        udp_sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        packet = {
            "grasp_command": 1,
            "extension_command": 0,
            "continuous_flex": 0.85,
            "continuous_splay": 0.0,
            "class_name": "volar_grasp",
            "quality": "good",
        }
        udp_sender.sendto(json.dumps(packet).encode("utf-8"), ("127.0.0.1", bridge_udp_port))
        udp_sender.close()

        # Check /api/state updated
        time.sleep(0.2)
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data["grasp_command"] == 1
            assert data["class_name"] == "volar_grasp"
            assert pytest.approx(data["continuous_flex"], 0.01) == 0.85

    finally:
        bridge.stop()


def test_studio_launcher_forwards_viewer_options(tmp_path):
    """Open viewer help through the shared launcher without a browser."""
    if os.name == 'nt':
        command = ['cmd.exe', '/d', '/c', 'call', str(REPO_ROOT / 'Launch_Studio.bat')]
    else:
        command = [sys.executable, str(REPO_ROOT / 'launch_studio.py')]
    result = subprocess.run(
        [*command, 'hand', '--help'], cwd=tmp_path,
        env=dict(os.environ, BCI_STUDIO_PYTHON=sys.executable),
        capture_output=True, text=True, timeout=30, check=True,
    )
    assert '3D Virtual Hand Desktop Viewer' in result.stdout
    assert '--standalone' in result.stdout
