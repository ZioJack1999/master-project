#!/usr/bin/env python3
"""
Virtual Hand Viewer - Standalone 3D Biomechanical Hand Launcher

Launches the 3D Virtual Hand client in a dedicated, distraction-free app window
using Chrome / Edge app mode (e.g., msedge --app=http://127.0.0.1:8052/virtual_hand)
with fallback to the system default browser.

Supports:
  - Port 8052: Binary BCI Studio (Rest vs. Right Hand Grasp)
  - Port 8051: Three-Class BCI Studio (Rest vs. Volar Grasp vs. Finger Extension)
  - Auto-detection: scans active ports to automatically bind to the running studio
  - Standalone UDP bridge: when run without a full BCI web server, spins up a
    lightweight local HTTP server and UDP 5005 receiver to animate the hand
    directly from external decoders, Unity, or Python test scripts.
"""

import argparse
import glob
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


DEFAULT_BINARY_PORT = 8052
DEFAULT_THREECLASS_PORT = 8051
DEFAULT_BRIDGE_PORT = 8055
DEFAULT_UDP_PORT = 5005
REPO_ROOT = Path(__file__).resolve().parent


def is_port_listening(host: str, port: int, timeout: float = 0.4) -> bool:
    """Check if a TCP port is open and accepting connections."""
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except (OSError, socket.error):
        return False


def query_studio_status(host: str, port: int, timeout: float = 0.8) -> dict:
    """Query /api/state from a running studio instance to inspect task metadata."""
    url = f"http://{host}:{port}/api/state"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "VirtualHandViewer/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def detect_active_port(host: str, preferred_mode: str = "auto") -> tuple[int, str]:
    """
    Detect the active BCI Studio port.
    Returns (port, description).
    """
    if preferred_mode == "binary":
        return DEFAULT_BINARY_PORT, "Binary BCI Studio (forced)"
    if preferred_mode == "threeclass":
        return DEFAULT_THREECLASS_PORT, "Three-Class BCI Studio (forced)"

    binary_active = is_port_listening(host, DEFAULT_BINARY_PORT)
    threeclass_active = is_port_listening(host, DEFAULT_THREECLASS_PORT)

    if binary_active and not threeclass_active:
        return DEFAULT_BINARY_PORT, "Binary BCI Studio (port 8052 active)"
    if threeclass_active and not binary_active:
        return DEFAULT_THREECLASS_PORT, "Three-Class BCI Studio (port 8051 active)"
    if binary_active and threeclass_active:
        # Both active: query state to inspect if one is running cues
        state_3c = query_studio_status(host, DEFAULT_THREECLASS_PORT)
        state_bin = query_studio_status(host, DEFAULT_BINARY_PORT)
        if state_3c.get("streaming") or state_3c.get("protocol"):
            return DEFAULT_THREECLASS_PORT, "Three-Class BCI Studio (port 8051 active session)"
        if state_bin.get("streaming") or state_bin.get("protocol"):
            return DEFAULT_BINARY_PORT, "Binary BCI Studio (port 8052 active session)"
        # Default to binary if ambiguous
        return DEFAULT_BINARY_PORT, "Binary BCI Studio (port 8052, both active)"

    # Neither active yet; default to binary 8052
    return DEFAULT_BINARY_PORT, "Binary BCI Studio (default, port 8052)"


def find_browser_executables() -> dict[str, str]:
    """Find installed browser executables supporting app mode or clean windows."""
    executables = {}

    # 1. Edge executables (exclude headless EdgeCore)
    edge_candidates = [
        shutil.which("msedge"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe"),
    ]
    for cand in edge_candidates:
        if cand and os.path.isfile(cand) and "EdgeCore" not in cand:
            executables["edge"] = cand
            break

    # 2. Chrome executables
    chrome_candidates = [
        shutil.which("chrome"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%PROGRAMFILES%\Google\Chrome\Application\chrome.exe"),
    ]
    for cand in chrome_candidates:
        if cand and os.path.isfile(cand):
            executables["chrome"] = cand
            break

    # 3. Brave executables
    brave_candidates = [
        shutil.which("brave"),
        r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\BraveSoftware\Brave-Browser\Application\brave.exe"),
    ]
    for cand in brave_candidates:
        if cand and os.path.isfile(cand):
            executables["brave"] = cand
            break

    # 4. Zen Browser / Firefox
    zen_candidates = [
        r"C:\Program Files\Zen Browser\zen.exe",
        shutil.which("zen"),
        shutil.which("firefox"),
        r"C:\Program Files\Mozilla Firefox\firefox.exe",
    ]
    for cand in zen_candidates:
        if cand and os.path.isfile(cand):
            executables["zen"] = cand
            break

    return executables


def launch_viewer_window(
    url: str,
    browser_pref: str = "auto",
    app_mode: bool = True,
    width: int = 820,
    height: int = 760,
    dry_run: bool = False,
) -> bool:
    """
    Launch the 3D Virtual Hand client in an app window or browser tab.
    """
    browsers = find_browser_executables()
    chosen_browser = None
    chosen_path = None

    if browser_pref != "auto" and browser_pref in browsers:
        chosen_browser = browser_pref
        chosen_path = browsers[browser_pref]
    elif browser_pref == "auto":
        # Prefer edge or chrome for seamless --app mode on Windows
        for b in ["edge", "chrome", "brave", "zen"]:
            if b in browsers:
                chosen_browser = b
                chosen_path = browsers[b]
                break

    if chosen_path and app_mode and chosen_browser in ("edge", "chrome", "brave"):
        cmd = [
            chosen_path,
            f"--app={url}",
            f"--window-size={width},{height}",
            "--new-window",
            "--disable-extensions",
        ]
        print(f"[VirtualHandViewer] Launching dedicated app window via {chosen_browser}:")
        print(f"  Command: {' '.join(cmd)}")
        if dry_run:
            return True
        try:
            subprocess.Popen(cmd, close_fds=sys.platform != "win32")
            return True
        except Exception as e:
            print(f"[VirtualHandViewer] App mode launch failed ({e}), falling back to webbrowser...")

    if chosen_path and chosen_browser == "zen":
        cmd = [chosen_path, "-new-window", url]
        print(f"[VirtualHandViewer] Launching window via {chosen_browser}: {' '.join(cmd)}")
        if dry_run:
            return True
        try:
            subprocess.Popen(cmd, close_fds=sys.platform != "win32")
            return True
        except Exception as e:
            print(f"[VirtualHandViewer] Browser launch failed ({e}), falling back...")

    # Fallback to python standard webbrowser / os.startfile
    print(f"[VirtualHandViewer] Opening via system default browser: {url}")
    if dry_run:
        return True
    try:
        if sys.platform == "win32":
            os.startfile(url)
            return True
        else:
            webbrowser.open(url, new=1, autoraise=True)
            return True
    except Exception as e:
        print(f"[VirtualHandViewer] Failed to open browser: {e}")
        return False


class StandaloneUDPBridge:
    """
    A lightweight standalone HTTP and UDP bridge server.
    Allows running the 3D Virtual Hand independently of BCI Studio,
    receiving live packets on UDP 5005 and serving /api/state.
    """

    def __init__(self, http_port: int = DEFAULT_BRIDGE_PORT, udp_port: int = DEFAULT_UDP_PORT, demo: bool = False):
        self.http_port = http_port
        self.udp_port = udp_port
        self.demo = demo
        self.running = False
        self.state = {
            "target": "rest",
            "class_name": "rest",
            "class_id": 0,
            "grasp_command": 0,
            "extension_command": 0,
            "continuous_flex": 0.12,
            "continuous_splay": 0.0,
            "p_smooth": 0.0,
            "probabilities": {"rest": 0.95, "volar_grasp": 0.03, "finger_extension": 0.02},
            "confidence": 0.95,
            "quality": "good",
            "task_label": "Standalone UDP Bridge",
            "timestamp": time.time(),
        }
        self.lock = threading.Lock()
        self.http_server = None
        self.udp_sock = None

    def start_udp_listener(self):
        try:
            self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_sock.bind(("0.0.0.0", self.udp_port))
            self.udp_sock.settimeout(0.5)
            print(f"[StandaloneBridge] UDP 5005 receiver bound to 0.0.0.0:{self.udp_port}")
        except Exception as e:
            print(f"[StandaloneBridge] Warning: UDP bind failed on {self.udp_port}: {e}")
            self.udp_sock = None
            return

        def _udp_loop():
            while self.running:
                try:
                    data, _ = self.udp_sock.recvfrom(4096)
                    parsed = json.loads(data.decode("utf-8"))
                    with self.lock:
                        if "grasp_command" in parsed:
                            self.state["grasp_command"] = int(parsed["grasp_command"])
                        if "extension_command" in parsed:
                            self.state["extension_command"] = int(parsed["extension_command"])
                        if "continuous_flex" in parsed:
                            self.state["continuous_flex"] = float(parsed["continuous_flex"])
                        if "continuous_splay" in parsed:
                            self.state["continuous_splay"] = float(parsed["continuous_splay"])
                        if "class_name" in parsed:
                            self.state["class_name"] = str(parsed["class_name"])
                            self.state["target"] = str(parsed["class_name"])
                        if "probabilities" in parsed and isinstance(parsed["probabilities"], dict):
                            self.state["probabilities"] = parsed["probabilities"]
                        if "quality" in parsed:
                            self.state["quality"] = str(parsed["quality"])
                        self.state["timestamp"] = time.time()
                except socket.timeout:
                    continue
                except Exception:
                    continue

        t = threading.Thread(target=_udp_loop, daemon=True)
        t.start()

    def start_demo_worker(self):
        """Synthetic animation loop when demo mode is requested or no UDP is received."""
        def _demo_loop():
            step = 0
            while self.running:
                time.sleep(0.04)
                step += 1
                if not self.demo:
                    continue
                cycle = (step % 200) / 200.0
                with self.lock:
                    if cycle < 0.35:
                        # Rest phase
                        self.state["class_name"] = "rest"
                        self.state["target"] = "rest"
                        self.state["grasp_command"] = 0
                        self.state["extension_command"] = 0
                        self.state["continuous_flex"] = 0.12
                        self.state["continuous_splay"] = 0.0
                        self.state["probabilities"] = {"rest": 0.88, "volar_grasp": 0.07, "finger_extension": 0.05}
                    elif cycle < 0.70:
                        # Volar Grasp phase
                        prog = (cycle - 0.35) / 0.35
                        self.state["class_name"] = "volar_grasp"
                        self.state["target"] = "volar_grasp"
                        self.state["grasp_command"] = 1
                        self.state["extension_command"] = 0
                        self.state["continuous_flex"] = 0.12 + 0.88 * prog
                        self.state["continuous_splay"] = 0.0
                        self.state["probabilities"] = {"rest": 0.08, "volar_grasp": 0.86, "finger_extension": 0.06}
                    else:
                        # Finger Extension phase
                        prog = (cycle - 0.70) / 0.30
                        self.state["class_name"] = "finger_extension"
                        self.state["target"] = "finger_extension"
                        self.state["grasp_command"] = 0
                        self.state["extension_command"] = 1
                        self.state["continuous_flex"] = 0.0
                        self.state["continuous_splay"] = 0.85 * prog
                        self.state["probabilities"] = {"rest": 0.05, "volar_grasp": 0.05, "finger_extension": 0.90}
                    self.state["timestamp"] = time.time()

        t = threading.Thread(target=_demo_loop, daemon=True)
        t.start()

    def start_http_server(self):
        bridge = self
        web_root = REPO_ROOT

        class BridgeHandler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(web_root), **kwargs)

            def log_message(self, format, *args):
                pass  # Keep console quiet

            def do_GET(self):
                clean_path = self.path.split("?")[0]
                if clean_path in ("/", "/virtual_hand", "/virtual_hand.html", "/hand"):
                    html_file = web_root / "virtual_hand.html"
                    if html_file.exists():
                        data = html_file.read_bytes()
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html; charset=utf-8")
                        self.send_header("Content-Length", str(len(data)))
                        self.end_headers()
                        self.wfile.write(data)
                        return
                elif clean_path == "/api/state":
                    with bridge.lock:
                        payload = json.dumps(bridge.state).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return

                super().do_GET()

            def guess_type(self, path):
                if str(path).endswith(".js"):
                    return "application/javascript"
                return super().guess_type(path)

        self.http_server = ThreadingHTTPServer(("127.0.0.1", self.http_port), BridgeHandler)
        print(f"[StandaloneBridge] HTTP server serving {web_root} at http://127.0.0.1:{self.http_port}/virtual_hand")

        t = threading.Thread(target=self.http_server.serve_forever, daemon=True)
        t.start()

    def run(self):
        self.running = True
        self.start_udp_listener()
        if self.demo:
            self.start_demo_worker()
        self.start_http_server()

    def stop(self):
        self.running = False
        if self.http_server:
            self.http_server.shutdown()
            self.http_server.server_close()
        if self.udp_sock:
            self.udp_sock.close()


def main():
    parser = argparse.ArgumentParser(
        description="3D Virtual Hand Desktop Viewer - distraction-free app window launcher.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect running BCI Studio (Binary on 8052 or Three-Class on 8051) and open:
  python virtual_hand_viewer.py

  # Explicitly connect to Binary BCI Studio (port 8052):
  python virtual_hand_viewer.py --port 8052
  python virtual_hand_viewer.py --binary

  # Explicitly connect to Three-Class BCI Studio (port 8051):
  python virtual_hand_viewer.py --port 8051
  python virtual_hand_viewer.py --threeclass

  # Standalone mode: listen on UDP 5005 and display 3D hand without full BCI Studio:
  python virtual_hand_viewer.py --standalone
  python virtual_hand_viewer.py --standalone --demo

  # Regular browser tab instead of app window:
  python virtual_hand_viewer.py --no-app
        """,
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--port",
        type=int,
        help="HTTP port of the target BCI Studio server (8052 for Binary, 8051 for Three-Class).",
    )
    mode_group.add_argument(
        "--binary",
        action="store_true",
        help="Target Binary BCI Studio on port 8052.",
    )
    mode_group.add_argument(
        "--threeclass",
        action="store_true",
        help="Target Three-Class BCI Studio on port 8051.",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address of BCI Studio (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="/virtual_hand",
        help="URL path to the 3D Virtual Hand (default: /virtual_hand).",
    )
    parser.add_argument(
        "--browser",
        type=str,
        choices=["auto", "edge", "chrome", "brave", "zen", "default"],
        default="auto",
        help="Preferred browser for launching the app window (default: auto).",
    )
    parser.add_argument(
        "--no-app",
        action="store_true",
        help="Open in a standard browser tab instead of a dedicated distraction-free app window.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=820,
        help="Window width in pixels (default: 820).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=760,
        help="Window height in pixels (default: 760).",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=0.0,
        help="Max seconds to wait for server socket to become available before launching (default: 0).",
    )
    parser.add_argument(
        "--standalone",
        "--bridge",
        dest="standalone",
        action="store_true",
        help="Spin up standalone UDP 5005 receiver bridge and local HTTP server.",
    )
    parser.add_argument(
        "--udp-port",
        type=int,
        default=DEFAULT_UDP_PORT,
        help=f"UDP port for standalone bridge mode (default: {DEFAULT_UDP_PORT}).",
    )
    parser.add_argument(
        "--bridge-port",
        type=int,
        default=DEFAULT_BRIDGE_PORT,
        help=f"Local HTTP port for standalone bridge mode (default: {DEFAULT_BRIDGE_PORT}).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="In standalone bridge mode, continuously animate synthetic grasp and extension cues.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved target URL and browser command without opening window.",
    )

    args = parser.parse_args()

    # Determine mode / port
    if args.standalone:
        print("=" * 70)
        print("   3D VIRTUAL HAND DESKTOP VIEWER (STANDALONE UDP BRIDGE)")
        print("=" * 70)
        bridge = StandaloneUDPBridge(
            http_port=args.bridge_port,
            udp_port=args.udp_port,
            demo=args.demo,
        )
        bridge.run()
        target_url = f"http://{args.host}:{args.bridge_port}{args.path}"
        time.sleep(0.3)
        launch_viewer_window(
            url=target_url,
            browser_pref=args.browser,
            app_mode=not args.no_app,
            width=args.width,
            height=args.height,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            bridge.stop()
            return 0
        print("[StandaloneBridge] Running. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n[StandaloneBridge] Shutting down.")
            bridge.stop()
        return 0

    target_mode = "auto"
    target_port = args.port
    if args.binary:
        target_mode = "binary"
        target_port = DEFAULT_BINARY_PORT
    elif args.threeclass:
        target_mode = "threeclass"
        target_port = DEFAULT_THREECLASS_PORT

    if target_port is None:
        target_port, desc = detect_active_port(args.host, preferred_mode=target_mode)
    else:
        desc = f"Specified port {target_port}"

    # If no studio server is running and user did not force a port, start standalone demo mode
    if not is_port_listening(args.host, target_port) and not args.binary and not args.threeclass and args.port is None:
        print("=" * 70)
        print("   3D VIRTUAL HAND DESKTOP VIEWER (STANDALONE DEMO MODE)")
        print("=" * 70)
        print(f"[VirtualHandViewer] No active BCI Studio detected on port {target_port}.")
        print("[VirtualHandViewer] Launching standalone 3D Virtual Hand in demo mode...")
        bridge = StandaloneUDPBridge(
            http_port=args.bridge_port,
            udp_port=args.udp_port,
            demo=True,
        )
        bridge.run()
        target_url = f"http://{args.host}:{args.bridge_port}{args.path}"
        time.sleep(0.3)
        launch_viewer_window(
            url=target_url,
            browser_pref=args.browser,
            app_mode=not args.no_app,
            width=args.width,
            height=args.height,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            bridge.stop()
            return 0
        print("[StandaloneBridge] Running live 3D Virtual Hand. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n[StandaloneBridge] Shutting down.")
            bridge.stop()
        return 0

    # Waiting logic if requested
    if args.wait > 0 and not is_port_listening(args.host, target_port):
        print(f"[VirtualHandViewer] Waiting up to {args.wait}s for {args.host}:{target_port} to start...")
        t0 = time.time()
        while time.time() - t0 < args.wait:
            if is_port_listening(args.host, target_port):
                print(f"[VirtualHandViewer] Connected to {args.host}:{target_port} after {time.time()-t0:.1f}s.")
                break
            time.sleep(0.25)

    target_url = f"http://{args.host}:{target_port}{args.path}"

    print("=" * 70)
    print("   3D VIRTUAL HAND DESKTOP VIEWER")
    print("=" * 70)
    print(f"  Target Studio : {desc}")
    print(f"  URL           : {target_url}")
    print(f"  App Window    : {'Enabled (Distraction-Free)' if not args.no_app else 'Disabled (Browser Tab)'}")
    print(f"  Window Size   : {args.width}x{args.height}")
    print("=" * 70)

    success = launch_viewer_window(
        url=target_url,
        browser_pref=args.browser,
        app_mode=not args.no_app,
        width=args.width,
        height=args.height,
        dry_run=args.dry_run,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
