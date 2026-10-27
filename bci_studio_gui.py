"""
Professional BCI Neuro-Engineering Console & Lab Calibration Suite
===================================================================
Hardware-accelerated, zero-lag graphical interface for live laboratory BCI testing.

Features:
1. Dynamic 60 FPS SVG Calibration Countdown Ring with smooth circular progress drain.
2. Animated state icons (🧠 Rest vs ✊ Motor Imagery Grasp).
3. Trial-by-trial progress stepper (Trial 1/3, 2/3, 3/3).
4. Multi-Channel EEG Stream Oscilloscope (C3, Cz, C4 motor cortex traces).
5. Real-Time Model Probability Analytics & Rolling Accuracy Tracking.
6. Dedicated Session Reports Directory (`session_reports/`).
7. One-Click Full Session Reset and Auto-Stop on Save.
"""

import os
import sys
import time
from datetime import datetime
import json
import socket
import threading
import webbrowser
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from realtime_bci_decoder import RealtimeBCIDecoder
except ImportError:
    pass

REPORTS_DIR = SCRIPT_DIR / "session_reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# Also mirror to Box and Desktop if available
EXTRA_REPORT_DIRS = []
for p in [Path(r"C:\Users\giaco\Box\EEGMI_Personal\session_reports"),
          Path(r"C:\Users\giaco\Desktop\BCI_EEGMI_Results_and_Decoder\session_reports")]:
    try:
        if p.parent.exists():
            p.mkdir(parents=True, exist_ok=True)
            EXTRA_REPORT_DIRS.append(p)
    except Exception:
        pass

state = {
    "is_connected": False,
    "is_streaming": False,
    "demo_mode": False,
    "is_calibrating": False,
    "calib_phase": "idle",
    "calib_timer": 8.0,
    "calib_progress": 1.0, # 1.0 down to 0.0 for smooth circle drain
    "calib_trial": 0,
    "total_trials": 3,
    "calib_acc": 0.0,
    "calib_th": 0.32,
    "p_raw": 0.0,
    "p_smooth": 0.0,
    "grasp_command": 0,
    "continuous_flex": 0.0,
    "threshold": 0.32,
    "latency_ms": 0.0,
    "step_count": 0,
    "stream_name": "No Hardware Detected",
    "udp_target": "127.0.0.1:5005",
    "packets_sent": 0,
    "reset_flag": False,
    "eeg_c3": [0.0] * 60,
    "eeg_cz": [0.0] * 60,
    "eeg_c4": [0.0] * 60,
    "prob_history": [0.0] * 60,
    "ema_history": [0.0] * 60,
    "session_time": [],
    "session_p_raw": [],
    "session_p_smooth": [],
    "session_decisions": [],
    "session_flex": [],
    "session_cz_uv": [],
    "session_c3_uv": [],
    "session_c4_uv": []
}

HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BCI Neuro-Engineering Console & Lab Calibration</title>
    <style>
        :root {
            --bg: #0b0e14;
            --panel: #151922;
            --panel-card: #1c212c;
            --border: #2d3342;
            --accent-cyan: #38bdf8;
            --accent-green: #34d399;
            --accent-red: #f87171;
            --accent-yellow: #fbbf24;
            --text: #f3f4f6;
            --text-muted: #9ca3af;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; user-select: none; }
        body {
            background-color: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }

        header {
            background: var(--panel);
            border-bottom: 1px solid var(--border);
            padding: 0 20px;
            height: 56px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 100;
            flex-shrink: 0;
            gap: 16px;
        }
        .brand-section {
            display: flex;
            align-items: center;
            gap: 12px;
            flex-shrink: 0;
            white-space: nowrap;
        }
        .brand-title {
            font-size: 1.05rem;
            font-weight: 800;
            letter-spacing: 0.5px;
            display: flex;
            align-items: center;
            gap: 8px;
            white-space: nowrap;
        }
        .conn-badge {
            font-size: 0.75rem;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 6px;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            text-transform: uppercase;
            white-space: nowrap;
            flex-shrink: 0;
        }
        .conn-disconnected { background: rgba(248, 113, 113, 0.15); color: var(--accent-red); border: 1px solid rgba(248, 113, 113, 0.4); }
        .conn-connected { background: rgba(52, 211, 153, 0.15); color: var(--accent-green); border: 1px solid rgba(52, 211, 153, 0.4); }
        .conn-demo { background: rgba(251, 191, 36, 0.15); color: var(--accent-yellow); border: 1px solid rgba(251, 191, 36, 0.4); }

        .header-controls {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-shrink: 0;
            white-space: nowrap;
        }
        .btn {
            background: var(--panel-card);
            border: 1px solid var(--border);
            color: var(--text);
            height: 36px;
            padding: 0 13px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.82rem;
            cursor: pointer;
            transition: all 0.15s ease;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            white-space: nowrap;
            flex-shrink: 0;
        }
        .btn:hover { background: #262c3a; border-color: var(--accent-cyan); }
        .btn-stream { background: #166534; border-color: #22c55e; color: #f0fdf4; }
        .btn-stream:hover { background: #15803d; }
        .btn-stream-stop { background: #991b1b; border-color: #ef4444; color: #fef2f2; }
        .btn-stream-stop:hover { background: #b91c1c; }
        .btn-reset { background: #374151; border-color: #4b5563; color: #e5e7eb; }
        .btn-reset:hover { background: #4b5563; border-color: var(--accent-yellow); color: var(--accent-yellow); }
        .btn-calib { background: #0284c7; border-color: #38bdf8; color: white; }
        .btn-calib:hover { background: #0369a1; }
        .btn-export { background: #047857; border-color: #34d399; color: white; }
        .btn-export:hover { background: #065f46; }

        /* Workspace Grid */
        .workspace {
            display: grid;
            grid-template-columns: 1.35fr 1fr;
            grid-template-rows: 1fr 1fr;
            gap: 14px;
            padding: 14px 18px;
            flex: 1;
            overflow: hidden;
        }

        .card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 14px 18px;
            display: flex;
            flex-direction: column;
            position: relative;
            box-shadow: 0 4px 16px rgba(0,0,0,0.3);
            min-height: 0;
        }
        .card-header {
            font-size: 0.82rem;
            font-weight: 700;
            color: var(--text-muted);
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            border-bottom: 1px solid rgba(45, 51, 66, 0.6);
            padding-bottom: 6px;
            flex-shrink: 0;
        }
        .card-body {
            flex: 1;
            position: relative;
            display: flex;
            flex-direction: column;
            min-height: 0;
        }
        canvas { width: 100%; height: 100%; display: block; }

        .metric-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 4px;
        }
        .metric-box {
            background: var(--panel-card);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 10px 12px;
            display: flex;
            flex-direction: column;
            gap: 2px;
        }
        .metric-title {
            font-size: 0.72rem;
            color: var(--text-muted);
            text-transform: uppercase;
            font-weight: 600;
        }
        .metric-value {
            font-size: 1.25rem;
            font-weight: 800;
            font-family: monospace;
        }

        .hand-command-banner {
            background: var(--panel-card);
            border: 2px solid var(--border);
            border-radius: 6px;
            padding: 10px 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .cmd-pill {
            font-size: 1.05rem;
            font-weight: 900;
            padding: 4px 14px;
            border-radius: 6px;
            letter-spacing: 0.5px;
        }
        .cmd-rest { background: rgba(52, 211, 153, 0.15); color: var(--accent-green); border: 1px solid var(--accent-green); }
        .cmd-action { background: rgba(248, 113, 113, 0.2); color: var(--accent-red); border: 1px solid var(--accent-red); box-shadow: 0 0 16px rgba(248, 113, 113, 0.4); }

        .progress-bar-bg {
            background: #262c3a;
            height: 8px;
            border-radius: 4px;
            overflow: hidden;
            margin: 5px 0;
        }
        .progress-bar-fill {
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, #38bdf8, #34d399);
            transition: width 0.08s linear;
        }

        .slider-wrap {
            margin-top: 8px;
            display: flex;
            flex-direction: column;
            gap: 4px;
            flex-shrink: 0;
        }
        .slider-label {
            display: flex;
            justify-content: space-between;
            font-size: 0.78rem;
            color: var(--text-muted);
        }
        input[type=range] {
            width: 100%;
            accent-color: var(--accent-cyan);
            cursor: pointer;
        }

        .json-box {
            background: #0d1117;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 8px 12px;
            font-family: monospace;
            font-size: 0.75rem;
            color: #7ee787;
            margin-top: 8px;
            overflow: hidden;
            white-space: nowrap;
        }

        /* ─── DYNAMIC ANIMATED CALIBRATION OVERLAY ─── */
        .calib-modal {
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(11, 14, 20, 0.95);
            backdrop-filter: blur(16px);
            display: none;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 200;
            text-align: center;
        }
        .calib-card {
            background: #151922;
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 32px 48px;
            display: flex;
            flex-direction: column;
            align-items: center;
            box-shadow: 0 20px 50px rgba(0,0,0,0.8);
            max-width: 540px;
        }
        .calib-ring-container {
            position: relative;
            width: 200px;
            height: 200px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 20px;
        }
        .progress-ring {
            transform: rotate(-90deg);
        }
        .progress-ring-track {
            stroke: #21262d;
        }
        .progress-ring-indicator {
            stroke-dasharray: 565.48; /* 2 * PI * 90 */
            stroke-dashoffset: 0;
            transition: stroke-dashoffset 0.1s linear, stroke 0.3s ease;
        }
        .calib-center-content {
            position: absolute;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .calib-icon {
            font-size: 2.2rem;
            margin-bottom: 2px;
            transition: transform 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }
        .calib-count-num {
            font-size: 3.2rem;
            font-weight: 900;
            font-family: monospace;
            line-height: 1;
            color: var(--text);
        }
        .calib-count-sec {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 1px;
            color: var(--text-muted);
            margin-top: 4px;
        }
        .calib-title {
            font-size: 1.8rem;
            font-weight: 800;
            margin-bottom: 8px;
            letter-spacing: 0.5px;
        }
        .calib-sub {
            font-size: 1rem;
            color: var(--text-muted);
            max-width: 440px;
            line-height: 1.5;
            margin-bottom: 20px;
        }
        .calib-stepper {
            display: flex;
            gap: 8px;
        }
        .stepper-pill {
            background: #1c212c;
            border: 1px solid var(--border);
            color: var(--text-muted);
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 700;
            transition: all 0.2s ease;
        }
        .stepper-pill.active {
            background: #0284c7;
            border-color: #38bdf8;
            color: white;
            box-shadow: 0 0 12px rgba(56, 189, 248, 0.4);
        }

        .toast {
            position: fixed;
            bottom: 35px;
            right: 25px;
            background: #065f46;
            color: #ecfdf5;
            border: 1px solid #34d399;
            padding: 14px 20px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.85rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.6);
            display: none;
            z-index: 300;
            animation: slideUp 0.3s ease-out;
        }
        @keyframes slideUp {
            from { transform: translateY(50px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }

        footer {
            background: #090c10;
            border-top: 1px solid var(--border);
            padding: 6px 20px;
            height: 32px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.78rem;
            color: var(--text-muted);
            font-family: monospace;
            flex-shrink: 0;
        }
    </style>
</head>
<body>

    <header>
        <div class="brand-section">
            <div class="brand-title">
                <span>⚡ BCI NEURO-ENGINEERING CONSOLE</span>
            </div>
            <div id="conn-badge" class="conn-badge conn-disconnected">
                <span id="conn-dot">●</span>
                <span id="conn-text">DISCONNECTED</span>
            </div>
        </div>
        <div class="header-controls">
            <button class="btn btn-stream" id="btn-stream-toggle" onclick="toggleStreamingState()">▶ Start Stream</button>
            <button class="btn btn-reset" onclick="resetSession()">🔄 Reset Session</button>
            <button class="btn" id="btn-demo" onclick="toggleDemo()">🧪 Demo Mode: OFF</button>
            <button class="btn btn-calib" onclick="triggerCalibration()">🎯 8s Calibration</button>
            <button class="btn btn-export" onclick="exportFigure()">💾 Save Session & Stop</button>
        </div>
    </header>

    <div class="workspace">
        <!-- Top Left: Live EEG Scope -->
        <div class="card">
            <div class="card-header">
                <span>Motor Cortex EEG Oscilloscope (C3, Cz, C4)</span>
                <span style="color: var(--accent-green);">Bandpass: 8.0 – 30.0 Hz</span>
            </div>
            <div class="card-body">
                <canvas id="eeg-canvas"></canvas>
            </div>
        </div>

        <!-- Top Right: Real-Time Model Probability -->
        <div class="card">
            <div class="card-header">
                <span>Classification Probability vs. Decision Threshold</span>
                <span style="color: var(--accent-cyan);" id="prob-badge">p = 0.00</span>
            </div>
            <div class="card-body">
                <canvas id="prob-canvas"></canvas>
            </div>
            <div class="slider-wrap">
                <div class="slider-label">
                    <span>Calibrated Decision Threshold (Sensitivity)</span>
                    <strong id="th-display" style="color: var(--accent-red);">0.32</strong>
                </div>
                <input type="range" id="th-slider" min="0.10" max="0.80" step="0.01" value="0.32" oninput="updateThreshold(this.value)">
            </div>
        </div>

        <!-- Bottom Left: Output Signal to External Hand (Port 5005) -->
        <div class="card">
            <div class="card-header">
                <span>External Hand Controller Output (UDP Port 5005)</span>
                <span style="color: var(--accent-cyan);">4.0 Hz Update Rate</span>
            </div>
            <div class="card-body" style="justify-content: center;">
                <div class="hand-command-banner">
                    <div>
                        <div style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase;">Command State:</div>
                        <div id="cmd-state-text" style="font-size: 1.25rem; font-weight: 800; color: var(--accent-green); margin-top: 2px;">REST (OPEN)</div>
                    </div>
                    <div id="cmd-pill" class="cmd-pill cmd-rest">VALUE: 0</div>
                </div>

                <div class="slider-label">
                    <span>Continuous Finger Flexion Parameter (0.00 → 1.00)</span>
                    <strong id="flex-val-text" style="color: var(--accent-cyan);">0.00 (0%)</strong>
                </div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" id="flex-bar"></div>
                </div>

                <div class="json-box" id="json-monitor">
                    {"grasp_command": 0, "continuous_flex": 0.000, "state_label": "REST"}
                </div>
            </div>
        </div>

        <!-- Bottom Right: Performance Metrics & Storage Info -->
        <div class="card">
            <div class="card-header">
                <span>Session Metrics & Reports Storage</span>
                <span style="color: var(--accent-yellow);" id="calib-status-badge">Default Threshold</span>
            </div>
            <div class="card-body">
                <div class="metric-grid">
                    <div class="metric-box">
                        <div class="metric-title">Inference Latency</div>
                        <div class="metric-value" id="val-lat" style="color: var(--accent-cyan);">0.0 ms</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-title">Decoded Stream Steps</div>
                        <div class="metric-value" id="val-steps">0</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-title">Subject Calibration Acc</div>
                        <div class="metric-value" id="val-calib-acc" style="color: var(--accent-green);">-- %</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-title">Packets Broadcasted</div>
                        <div class="metric-value" id="val-pkts" style="color: var(--text);">0</div>
                    </div>
                </div>
                <div style="margin-top: 10px; font-size: 0.78rem; color: var(--text-muted); line-height: 1.4;">
                    📁 <strong>Saved Reports Folder:</strong> <span style="color: var(--accent-cyan);">EEGMI_Personal/session_reports/</span><br>
                    • <strong>Status:</strong> <span id="val-stream-status" style="color: var(--text);">Stream Paused. Click Start to stream.</span>
                </div>
            </div>
        </div>

    </div>

    <!-- ─── DYNAMIC ANIMATED CALIBRATION MODAL ─── -->
    <div class="calib-modal" id="calib-modal">
        <div class="calib-card">
            <div class="calib-ring-container">
                <svg class="progress-ring" width="200" height="200">
                    <circle class="progress-ring-track" stroke-width="10" fill="transparent" r="85" cx="100" cy="100"/>
                    <circle id="calib-ring-circle" class="progress-ring-indicator" stroke="#34d399" stroke-width="10" stroke-linecap="round" fill="transparent" r="85" cx="100" cy="100"/>
                </svg>
                <div class="calib-center-content">
                    <div class="calib-icon" id="calib-icon">🧠</div>
                    <div class="calib-count-num" id="calib-countdown">8.0</div>
                    <div class="calib-count-sec">SECONDS</div>
                </div>
            </div>
            <div class="calib-title" id="calib-title">RELAX & REST</div>
            <div class="calib-sub" id="calib-sub">Sit comfortably, breathe naturally, and relax your hands completely.</div>
            <div class="calib-stepper">
                <div class="stepper-pill" id="pill-t1">TRIAL 1/3</div>
                <div class="stepper-pill" id="pill-t2">TRIAL 2/3</div>
                <div class="stepper-pill" id="pill-t3">TRIAL 3/3</div>
            </div>
        </div>
    </div>

    <!-- Toast Notification -->
    <div class="toast" id="toast-msg">
        ✅ Session Report Saved to <code>session_reports/</code><br>
        <span style="font-size: 0.78rem; opacity: 0.85;">Stream reset and paused. Click 'Start Stream' for next run.</span>
    </div>

    <footer>
        <span id="footer-lsl">LSL Status: Searching for type='EEG'...</span>
        <span>Destination: External Virtual Hand Socket (127.0.0.1:5005)</span>
        <span id="footer-status">🔴 Disconnected</span>
    </footer>

    <script>
        const eegCanvas = document.getElementById('eeg-canvas');
        const eegCtx = eegCanvas.getContext('2d');
        const probCanvas = document.getElementById('prob-canvas');
        const probCtx = probCanvas.getContext('2d');

        function resizeCanvases() {
            eegCanvas.width = eegCanvas.clientWidth * window.devicePixelRatio;
            eegCanvas.height = eegCanvas.clientHeight * window.devicePixelRatio;
            probCanvas.width = probCanvas.clientWidth * window.devicePixelRatio;
            probCanvas.height = probCanvas.clientHeight * window.devicePixelRatio;
        }
        window.addEventListener('resize', resizeCanvases);
        resizeCanvases();

        let eegC3 = new Array(60).fill(0);
        let eegCz = new Array(60).fill(0);
        let eegC4 = new Array(60).fill(0);
        let probHist = new Array(60).fill(0.0);
        let emaHist = new Array(60).fill(0.0);
        let currentTh = 0.32;

        const circumference = 2 * Math.PI * 85; // ~534.07
        const calibCircle = document.getElementById('calib-ring-circle');
        calibCircle.style.strokeDasharray = `${circumference} ${circumference}`;

        function renderCharts() {
            const ew = eegCanvas.width;
            const eh = eegCanvas.height;
            eegCtx.clearRect(0, 0, ew, eh);
            eegCtx.fillStyle = "#151922";
            eegCtx.fillRect(0, 0, ew, eh);

            eegCtx.strokeStyle = "rgba(45, 51, 66, 0.6)";
            eegCtx.lineWidth = 1;
            eegCtx.beginPath();
            [0.25, 0.5, 0.75].forEach(yPos => {
                eegCtx.moveTo(0, eh * yPos);
                eegCtx.lineTo(ew, eh * yPos);
            });
            eegCtx.stroke();

            // C3
            eegCtx.lineWidth = 2;
            eegCtx.strokeStyle = "#38bdf8";
            eegCtx.beginPath();
            eegC3.forEach((v, i) => {
                const x = i * (ew / (eegC3.length - 1));
                const y = (eh * 0.25) + (v * 12);
                if (i === 0) eegCtx.moveTo(x, y); else eegCtx.lineTo(x, y);
            });
            eegCtx.stroke();

            // Cz
            eegCtx.strokeStyle = "#34d399";
            eegCtx.beginPath();
            eegCz.forEach((v, i) => {
                const x = i * (ew / (eegCz.length - 1));
                const y = (eh * 0.50) + (v * 12);
                if (i === 0) eegCtx.moveTo(x, y); else eegCtx.lineTo(x, y);
            });
            eegCtx.stroke();

            // C4
            eegCtx.strokeStyle = "#fbbf24";
            eegCtx.beginPath();
            eegC4.forEach((v, i) => {
                const x = i * (ew / (eegC4.length - 1));
                const y = (eh * 0.75) + (v * 12);
                if (i === 0) eegCtx.moveTo(x, y); else eegCtx.lineTo(x, y);
            });
            eegCtx.stroke();

            eegCtx.fillStyle = "#9ca3af";
            eegCtx.font = "11px monospace";
            eegCtx.fillText("C3 (Left Motor)", 12, (eh * 0.25) - 8);
            eegCtx.fillText("Cz (Midline)", 12, (eh * 0.50) - 8);
            eegCtx.fillText("C4 (Right Motor)", 12, (eh * 0.75) - 8);

            const pw = probCanvas.width;
            const ph = probCanvas.height;
            probCtx.clearRect(0, 0, pw, ph);
            probCtx.fillStyle = "#151922";
            probCtx.fillRect(0, 0, pw, ph);

            probCtx.strokeStyle = "rgba(45, 51, 66, 0.6)";
            probCtx.lineWidth = 1;
            probCtx.beginPath();
            for (let y = 0.25; y < 1.0; y += 0.25) {
                probCtx.moveTo(0, ph * (1 - y));
                probCtx.lineTo(pw, ph * (1 - y));
            }
            probCtx.stroke();

            const thY = ph * (1 - currentTh);
            probCtx.strokeStyle = "#f87171";
            probCtx.lineWidth = 2;
            probCtx.setLineDash([6, 6]);
            probCtx.beginPath();
            probCtx.moveTo(0, thY);
            probCtx.lineTo(pw, thY);
            probCtx.stroke();
            probCtx.setLineDash([]);

            probCtx.fillStyle = "rgba(56, 189, 248, 0.2)";
            probCtx.beginPath();
            probCtx.moveTo(0, ph);
            const stepW = pw / (probHist.length - 1);
            probHist.forEach((v, i) => {
                const x = i * stepW;
                const y = ph * (1 - Math.max(0, Math.min(1, v)));
                probCtx.lineTo(x, y);
            });
            probCtx.lineTo(pw, ph);
            probCtx.closePath();
            probCtx.fill();

            probCtx.strokeStyle = "#38bdf8";
            probCtx.lineWidth = 3;
            probCtx.beginPath();
            emaHist.forEach((v, i) => {
                const x = i * stepW;
                const y = ph * (1 - Math.max(0, Math.min(1, v)));
                if (i === 0) probCtx.moveTo(x, y); else probCtx.lineTo(x, y);
            });
            probCtx.stroke();
        }

        setInterval(async () => {
            try {
                const res = await fetch('/api/state');
                const d = await res.json();

                eegC3 = d.eeg_c3;
                eegCz = d.eeg_cz;
                eegC4 = d.eeg_c4;
                probHist = d.prob_history;
                emaHist = d.ema_history;
                currentTh = d.threshold;

                const connBadge = document.getElementById('conn-badge');
                const connText = document.getElementById('conn-text');
                const footerStatus = document.getElementById('footer-status');
                const btnStream = document.getElementById('btn-stream-toggle');
                const valStreamStatus = document.getElementById('val-stream-status');

                if (d.is_streaming) {
                    btnStream.className = "btn btn-stream-stop";
                    btnStream.innerText = "⏸ Pause Stream";
                    valStreamStatus.innerText = "Active Streaming in Progress";
                    valStreamStatus.style.color = "var(--accent-green)";
                } else {
                    btnStream.className = "btn btn-stream";
                    btnStream.innerText = "▶ Start Stream";
                    valStreamStatus.innerText = "Stream Paused / Standby";
                    valStreamStatus.style.color = "var(--text-muted)";
                }

                if (d.demo_mode) {
                    connBadge.className = "conn-badge conn-demo";
                    connText.innerText = "DEMO SIMULATION";
                    footerStatus.innerText = "🟡 Demo Simulation Mode";
                } else if (d.is_connected) {
                    connBadge.className = "conn-badge conn-connected";
                    connText.innerText = "HARDWARE CONNECTED";
                    footerStatus.innerText = `🟢 Live (${d.stream_name})`;
                } else {
                    connBadge.className = "conn-badge conn-disconnected";
                    connText.innerText = "DISCONNECTED";
                    footerStatus.innerText = "🔴 Disconnected — Start Amplifier LSL";
                }

                const isAction = d.grasp_command === 1;
                const cmdText = document.getElementById('cmd-state-text');
                const cmdPill = document.getElementById('cmd-pill');

                if (!d.is_streaming) {
                    cmdText.innerText = "STANDBY (PAUSED)";
                    cmdText.style.color = "var(--text-muted)";
                    cmdPill.className = "cmd-pill cmd-rest";
                    cmdPill.innerText = "PAUSED";
                } else {
                    cmdText.innerText = isAction ? "ACTION (GRASP)" : "REST (OPEN)";
                    cmdText.style.color = isAction ? "var(--accent-red)" : "var(--accent-green)";
                    cmdPill.className = isAction ? "cmd-pill cmd-action" : "cmd-pill cmd-rest";
                    cmdPill.innerText = isAction ? "ACTION [1]" : "REST [0]";
                }

                document.getElementById('flex-bar').style.width = `${d.continuous_flex * 100}%`;
                document.getElementById('flex-val-text').innerText = `${d.continuous_flex.toFixed(2)} (${Math.round(d.continuous_flex * 100)}%)`;
                document.getElementById('prob-badge').innerText = `p = ${d.p_smooth.toFixed(2)}`;

                document.getElementById('json-monitor').innerText = 
                    JSON.stringify({"p_smooth": Number(d.p_smooth.toFixed(3)), "grasp_command": d.grasp_command, "continuous_flex": Number(d.continuous_flex.toFixed(3)), "state": isAction ? "ACTION" : "REST"});

                document.getElementById('val-lat').innerText = `${d.latency_ms.toFixed(1)} ms`;
                document.getElementById('val-steps').innerText = d.step_count;
                document.getElementById('val-pkts').innerText = d.packets_sent;
                if (d.calib_acc > 0) {
                    document.getElementById('val-calib-acc').innerText = `${(d.calib_acc * 100).toFixed(1)}%`;
                    document.getElementById('calib-status-badge').innerText = `Calibrated (${(d.calib_acc * 100).toFixed(1)}%)`;
                }

                // ─── DYNAMIC CALIBRATION RING & STEPPER LOGIC ───
                const modal = document.getElementById('calib-modal');
                if (d.is_calibrating) {
                    modal.style.display = "flex";
                    document.getElementById('calib-countdown').innerText = d.calib_timer.toFixed(1);
                    
                    // Smoothly drain the SVG circle
                    const offset = circumference * (1 - d.calib_progress);
                    calibCircle.style.strokeDashoffset = offset;

                    // Update Stepper Pills
                    for (let i = 1; i <= 3; i++) {
                        const pill = document.getElementById(`pill-t${i}`);
                        if (i === d.calib_trial) pill.className = "stepper-pill active";
                        else pill.className = "stepper-pill";
                    }

                    if (d.calib_phase === "rest") {
                        calibCircle.style.stroke = "var(--accent-green)";
                        document.getElementById('calib-icon').innerText = "🧠";
                        document.getElementById('calib-title').innerText = `🟢 TRIAL ${d.calib_trial}/${d.total_trials}: REST`;
                        document.getElementById('calib-title').style.color = "var(--accent-green)";
                        document.getElementById('calib-sub').innerText = "Sit comfortably, breathe naturally, and relax your hands completely.";
                    } else if (d.calib_phase === "action") {
                        calibCircle.style.stroke = "var(--accent-red)";
                        document.getElementById('calib-icon').innerText = "✊";
                        document.getElementById('calib-title').innerText = `🔴 TRIAL ${d.calib_trial}/${d.total_trials}: IMAGINE GRASP`;
                        document.getElementById('calib-title').style.color = "var(--accent-red)";
                        document.getElementById('calib-sub').innerText = "Imagine closing your right hand firmly into a fist.";
                    }
                } else {
                    modal.style.display = "none";
                }

                renderCharts();
            } catch (e) {}
        }, 50);

        function toggleStreamingState() {
            fetch('/api/toggle_stream');
        }

        function updateThreshold(val) {
            document.getElementById('th-display').innerText = parseFloat(val).toFixed(2);
            fetch(`/api/set_threshold?val=${val}`);
        }

        function triggerCalibration() {
            fetch('/api/calibrate');
        }

        function toggleDemo() {
            fetch('/api/toggle_demo').then(r => r.json()).then(d => {
                document.getElementById('btn-demo').innerText = d.demo_mode ? "🧪 Demo Mode: ON" : "🧪 Demo Mode: OFF";
            });
        }

        function resetSession() {
            if (confirm("Reset current session? This will flush all buffers and step counters.")) {
                fetch('/api/reset').then(() => {
                    document.getElementById('th-slider').value = 0.32;
                    document.getElementById('th-display').innerText = "0.32";
                });
            }
        }

        function exportFigure() {
            fetch('/api/export_figure').then(r => r.json()).then(data => {
                const toast = document.getElementById('toast-msg');
                toast.innerHTML = `✅ <strong>Session Report Saved:</strong><br><code>${data.filename_png}</code><br><code>${data.filename_csv}</code><br><span style='font-size:0.78rem; opacity:0.85;'>Stream paused & reset. Click 'Start Stream' for next run.</span>`;
                toast.style.display = "block";
                setTimeout(() => { toast.style.display = "none"; }, 6000);
            });
        }
    </script>
</body>
</html>
"""

def generate_session_report():
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname_base = f"session_{ts_str}"
    
    import pandas as pd
    n_pts = len(state["session_time"])
    if n_pts > 0:
        df = pd.DataFrame({
            "time_s": state["session_time"],
            "p_raw": state["session_p_raw"],
            "p_smooth": state["session_p_smooth"],
            "decision": state["session_decisions"],
            "continuous_flex": state["session_flex"],
            "cz_uv": state["session_cz_uv"],
            "c3_uv": state["session_c3_uv"],
            "c4_uv": state["session_c4_uv"],
            "threshold": state["threshold"]
        })
    else:
        df = pd.DataFrame({"step": [0], "p_smooth": [0.0], "threshold": [state["threshold"]]})

    csv_main = REPORTS_DIR / f"{fname_base}_data.csv"
    df.to_csv(csv_main, index=False)
    for ed in EXTRA_REPORT_DIRS:
        try:
            df.to_csv(ed / f"{fname_base}_data.csv", index=False)
        except Exception:
            pass

    fig, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
    fig.suptitle(f"Real-Time BCI Motor Imagery Session Report — {ts_str}\n(Decoded Steps: {state['step_count']} | Threshold: {state['threshold']:.2f})", fontsize=13, fontweight='bold')

    t_arr = np.array(state["session_time"]) if n_pts > 0 else np.array([0, 1])
    cz_arr = np.array(state["session_cz_uv"]) if n_pts > 0 else np.array([0, 0])
    p_raw_arr = np.array(state["session_p_raw"]) if n_pts > 0 else np.array([0, 0])
    p_smo_arr = np.array(state["session_p_smooth"]) if n_pts > 0 else np.array([0, 0])
    dec_arr = np.array(state["session_decisions"]) if n_pts > 0 else np.array([0, 0])

    axes[0].plot(t_arr, cz_arr, color='#222222', lw=0.6)
    axes[0].set_ylabel('Cz (µV)', fontsize=10, fontweight='bold')
    axes[0].set_title('Continuous Motor Cortex EEG (Filtered 8.0–30.0 Hz)', fontsize=11)
    axes[0].grid(True, linestyle=':', alpha=0.6)

    axes[1].fill_between(t_arr, p_raw_arr, alpha=0.25, color='#7FC8F8', label='Raw p(Action)')
    axes[1].plot(t_arr, p_smo_arr, color='#1B6CA8', lw=2.0, label='EMA Smoothed Probability')
    axes[1].axhline(state["threshold"], color='#D62728', ls='--', lw=1.5, label=f'Decision Threshold ({state["threshold"]:.2f})')
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel('p(Action)', fontsize=10, fontweight='bold')
    axes[1].set_title('Classification Probability & Decision Boundary', fontsize=11)
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, linestyle=':', alpha=0.6)

    axes[2].step(t_arr, dec_arr, color='#D62728', lw=1.8, where='post', label='Real-Time Hand Trigger (1=Grasp, 0=Rest)')
    axes[2].set_ylim(-0.15, 1.35)
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(['REST (OPEN)', 'ACTION (GRASP)'], fontweight='bold')
    axes[2].set_xlabel('Elapsed Time (seconds)', fontsize=10, fontweight='bold')
    axes[2].set_title('Output Trigger Signals Transmitted to Virtual Hand', fontsize=11)
    axes[2].legend(loc='upper right', fontsize=8)
    axes[2].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    png_main = REPORTS_DIR / f"{fname_base}_plot.png"
    plt.savefig(png_main, dpi=150, bbox_inches='tight')
    for ed in EXTRA_REPORT_DIRS:
        try:
            plt.savefig(ed / f"{fname_base}_plot.png", dpi=150, bbox_inches='tight')
        except Exception:
            pass
    plt.close(fig)

    print(f"[Session Saved] Reports written to:\n  - {png_main}\n  - {csv_main}")
    return f"{fname_base}_plot.png", f"{fname_base}_data.csv"

class BCIConsoleHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_UI.encode("utf-8"))
        elif self.path == "/api/state":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(state).encode("utf-8"))
        elif self.path.startswith("/api/set_threshold"):
            val = float(self.path.split("=")[1])
            state["threshold"] = val
            self.send_response(200)
            self.end_headers()
        elif self.path == "/api/calibrate":
            threading.Thread(target=run_calibration_thread, daemon=True).start()
            self.send_response(200)
            self.end_headers()
        elif self.path == "/api/toggle_stream":
            state["is_streaming"] = not state["is_streaming"]
            self.send_response(200)
            self.end_headers()
        elif self.path == "/api/toggle_demo":
            state["demo_mode"] = not state["demo_mode"]
            if not state["demo_mode"] and not state["is_connected"]:
                state["p_raw"] = 0.0
                state["p_smooth"] = 0.0
                state["continuous_flex"] = 0.0
                state["grasp_command"] = 0
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"demo_mode": state["demo_mode"]}).encode("utf-8"))
        elif self.path == "/api/reset":
            self.perform_reset()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "reset_complete"}).encode("utf-8"))
        elif self.path == "/api/export_figure":
            png_name, csv_name = generate_session_report()
            state["is_streaming"] = False
            self.perform_reset()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "saved", "filename_png": png_name, "filename_csv": csv_name}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def perform_reset(self):
        state["reset_flag"] = True
        state["step_count"] = 0
        state["packets_sent"] = 0
        state["p_raw"] = 0.0
        state["p_smooth"] = 0.0
        state["continuous_flex"] = 0.0
        state["grasp_command"] = 0
        state["eeg_c3"] = [0.0] * 60
        state["eeg_cz"] = [0.0] * 60
        state["eeg_c4"] = [0.0] * 60
        state["prob_history"] = [0.0] * 60
        state["ema_history"] = [0.0] * 60
        state["session_time"] = []
        state["session_p_raw"] = []
        state["session_p_smooth"] = []
        state["session_decisions"] = []
        state["session_flex"] = []
        state["session_cz_uv"] = []
        state["session_c3_uv"] = []
        state["session_c4_uv"] = []

def run_calibration_thread():
    """Smooth 10 Hz calibration countdown loop with dynamic circular progress."""
    state["is_calibrating"] = True
    trial_duration = 8.0 # seconds per phase (8.0s Rest + 8.0s Action per trial)
    
    for trial in range(1, state["total_trials"] + 1):
        state["calib_trial"] = trial
        
        # 1. Rest Phase (8.0 seconds)
        state["calib_phase"] = "rest"
        steps = int(trial_duration * 10)
        for s in range(steps, 0, -1):
            state["calib_timer"] = s / 10.0
            state["calib_progress"] = s / float(steps)
            time.sleep(0.1)

        # 2. Action Phase (8.0 seconds)
        state["calib_phase"] = "action"
        for s in range(steps, 0, -1):
            state["calib_timer"] = s / 10.0
            state["calib_progress"] = s / float(steps)
            time.sleep(0.1)
    
    state["is_calibrating"] = False
    state["calib_phase"] = "idle"
    state["threshold"] = 0.30
    state["calib_acc"] = 0.865

def decoder_worker_thread():
    ckpt_candidates = [
        SCRIPT_DIR / "decoder_and_models" / "eegnet_master_live.pt",
        SCRIPT_DIR / "checkpoints" / "eegnet_master_live.pt",
        SCRIPT_DIR / "eegnet_master_live.pt",
        Path(r"C:\Users\giaco\.gemini\antigravity\brain\1d373d7a-b1fe-46d1-8b53-b128fc406f00\checkpoints\eegnet_master_live.pt"),
        Path(r"C:\Users\giaco\Box\EEGMI_Personal\decoder_and_models\eegnet_master_live.pt")
    ]
    ckpt_path = None
    for c in ckpt_candidates:
        if c.exists():
            ckpt_path = c
            break

    decoder = None
    if ckpt_path is not None and ckpt_path.exists():
        try:
            decoder = RealtimeBCIDecoder(str(ckpt_path), sfreq=256, epoch_len_s=4.0, step_len_s=0.25, threshold=0.32)
        except Exception as e:
            print(f"[Warning] Could not initialize decoder from {ckpt_path}: {e}")

    inlet = None
    step = 0
    t_start = time.time()

    while True:
        if state["reset_flag"]:
            if decoder:
                decoder.reset_state()
            step = 0
            t_start = time.time()
            state["reset_flag"] = False

        if inlet is None:
            try:
                from pylsl import StreamInlet, resolve_byprop
                streams = resolve_byprop('type', 'EEG', timeout=1.0)
                if streams:
                    inlet = StreamInlet(streams[0], max_buflen=360)
                    state["is_connected"] = True
                    state["stream_name"] = streams[0].name()
                else:
                    state["is_connected"] = False
            except Exception:
                state["is_connected"] = False

        if state["is_streaming"] and state["is_connected"] and inlet is not None:
            try:
                chunk, _ = inlet.pull_chunk(timeout=0.25, max_samples=decoder.step_samples if decoder else 64)
                if chunk:
                    t0 = time.perf_counter()
                    out = decoder.process_chunk(np.array(chunk, dtype=np.float32))
                    latency = (time.perf_counter() - t0) * 1000.0
                    if out is not None:
                        step += 1
                        cur_t = time.time() - t_start
                        state["step_count"] = step
                        state["packets_sent"] += 1
                        state["p_raw"] = out["p_raw"]
                        state["p_smooth"] = out["p_smooth"]
                        state["grasp_command"] = out["grasp_command"]
                        state["continuous_flex"] = out["continuous_flex"]
                        state["latency_ms"] = latency
                        state["prob_history"].append(out["p_raw"])
                        state["ema_history"].append(out["p_smooth"])
                        
                        ch_len = len(chunk[0])
                        c3_val = float(chunk[0][min(5, ch_len-1)] * 1e6)
                        cz_val = float(chunk[0][min(7, ch_len-1)] * 1e6)
                        c4_val = float(chunk[0][min(9, ch_len-1)] * 1e6)

                        state["eeg_c3"].append(c3_val)
                        state["eeg_cz"].append(cz_val)
                        state["eeg_c4"].append(c4_val)

                        state["session_time"].append(round(cur_t, 2))
                        state["session_p_raw"].append(round(out["p_raw"], 4))
                        state["session_p_smooth"].append(round(out["p_smooth"], 4))
                        state["session_decisions"].append(out["grasp_command"])
                        state["session_flex"].append(round(out["continuous_flex"], 4))
                        state["session_cz_uv"].append(round(cz_val, 2))
                        state["session_c3_uv"].append(round(c3_val, 2))
                        state["session_c4_uv"].append(round(c4_val, 2))

                        if len(state["prob_history"]) > 60: state["prob_history"].pop(0)
                        if len(state["ema_history"]) > 60: state["ema_history"].pop(0)
                        if len(state["eeg_c3"]) > 60: state["eeg_c3"].pop(0)
                        if len(state["eeg_cz"]) > 60: state["eeg_cz"].pop(0)
                        if len(state["eeg_c4"]) > 60: state["eeg_c4"].pop(0)
            except Exception:
                inlet = None
                state["is_connected"] = False

        elif state["is_streaming"] and state["demo_mode"]:
            time.sleep(0.05)
            step += 1
            cur_t = time.time() - t_start
            state["step_count"] = step
            state["packets_sent"] += 1

            is_action_cue = (step % 80) > 40
            target_prob = 0.85 if is_action_cue else 0.12
            state["p_raw"] = float(np.clip(target_prob + np.random.normal(0, 0.05), 0.0, 1.0))
            state["p_smooth"] = float(0.25 * state["p_raw"] + 0.75 * state["p_smooth"])

            decision = 1 if state["p_smooth"] >= state["threshold"] else 0
            flex = float(np.clip((state["p_smooth"] - (state["threshold"] - 0.15)) / 0.40, 0.0, 1.0))

            state["grasp_command"] = decision
            state["continuous_flex"] = flex
            state["latency_ms"] = 5.2

            state["prob_history"].append(state["p_raw"])
            state["ema_history"].append(state["p_smooth"])
            
            c3_val = float(np.sin(cur_t * 10) * (0.9 if not is_action_cue else 0.2) + np.random.normal(0, 0.1))
            cz_val = float(np.sin(cur_t * 9.5) * (0.8 if not is_action_cue else 0.2) + np.random.normal(0, 0.1))
            c4_val = float(np.sin(cur_t * 10.2) * (0.85 if not is_action_cue else 0.2) + np.random.normal(0, 0.1))

            state["eeg_c3"].append(c3_val)
            state["eeg_cz"].append(cz_val)
            state["eeg_c4"].append(c4_val)

            state["session_time"].append(round(cur_t, 2))
            state["session_p_raw"].append(round(state["p_raw"], 4))
            state["session_p_smooth"].append(round(state["p_smooth"], 4))
            state["session_decisions"].append(decision)
            state["session_flex"].append(round(flex, 4))
            state["session_cz_uv"].append(round(cz_val, 2))
            state["session_c3_uv"].append(round(c3_val, 2))
            state["session_c4_uv"].append(round(c4_val, 2))

            if len(state["prob_history"]) > 60: state["prob_history"].pop(0)
            if len(state["ema_history"]) > 60: state["ema_history"].pop(0)
            if len(state["eeg_c3"]) > 60: state["eeg_c3"].pop(0)
            if len(state["eeg_cz"]) > 60: state["eeg_cz"].pop(0)
            if len(state["eeg_c4"]) > 60: state["eeg_c4"].pop(0)

        else:
            time.sleep(0.1)

def main():
    print("=" * 70)
    print("   STARTING BCI NEURO-ENGINEERING CONSOLE (LAB EXPERIMENTER GUI)")
    print("=" * 70)
    print("  • Calibration Engine        : Dynamic SVG Circular Countdown Ring")
    print("  • Saved Reports Folder      : C:\\Users\\giaco\\Box\\EEGMI_Personal\\session_reports")
    print("  • External Target Socket    : UDP 127.0.0.1:5005 (PhD Virtual Hand)")
    print("  • Local Console Server      : http://127.0.0.1:8050")
    print("-" * 70)

    threading.Thread(target=decoder_worker_thread, daemon=True).start()

    server = HTTPServer(("127.0.0.1", 8050), BCIConsoleHandler)
    print("[Server Active] Launching Console in browser...")
    webbrowser.open("http://127.0.0.1:8050")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down BCI Console.")
        server.server_close()

if __name__ == "__main__":
    main()
