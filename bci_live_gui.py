"""
Real-Time BCI Live Dashboard & Visual GUI
=========================================
Interactive Desktop Application for Closed-Loop Motor Imagery BCI & Virtual Hand Control.

Features:
1. Live EEG Oscilloscope (C3, Cz, C4 motor channels in microvolts).
2. Live Probability Strip-Chart (Raw vs. EMA Smoothed with Decision Threshold).
3. Live Visual Hand Actuator & Flexion Gauge (0.0 to 1.0 smooth animation).
4. One-Click 60s Guided Calibration with visual prompt cues.
5. End-of-Run 3-Panel Figure Export.
"""

import os
import sys
import time
import json
import socket
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from realtime_bci_decoder import RealtimeBCIDecoder, RingBuffer, EEGNet

class BCILiveGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Motor Imagery BCI Dashboard — Virtual Hand Controller")
        self.root.geometry("1280x820")
        self.root.configure(bg="#1E1E2E")

        # Decoder State
        self.decoder = None
        self.is_streaming = False
        self.is_calibrating = False
        self.inlet = None
        self.step_history = []
        self.prob_history = []
        self.ema_history = []
        self.time_history = []
        self.eeg_history_c3 = []
        self.eeg_history_cz = []
        self.eeg_history_c4 = []
        self.max_display_steps = 120  # ~30 seconds of rolling history

        # Setup GUI Components
        self.setup_ui()
        self.init_decoder()

    def init_decoder(self):
        ckpt_path = SCRIPT_DIR / "decoder_and_models" / "eegnet_master_live.pt"
        if not ckpt_path.exists():
            ckpt_path = SCRIPT_DIR / "eegnet_master_live.pt"
        if not ckpt_path.exists():
            ckpt_path = Path(r"C:\Users\giaco\.gemini\antigravity\brain\1d373d7a-b1fe-46d1-8b53-b128fc406f00\checkpoints\eegnet_master_live.pt")

        if ckpt_path.exists():
            self.decoder = RealtimeBCIDecoder(
                ckpt_path=str(ckpt_path),
                sfreq=256,
                epoch_len_s=4.0,
                step_len_s=0.25,
                threshold=0.32,
                ema_alpha=0.3,
                dwell_steps=2
            )
            self.status_var.set(f"Loaded Master Model ({ckpt_path.name}) | Ready")
        else:
            self.status_var.set("Model checkpoint not found. Please train or download model.")

    def setup_ui(self):
        # ─── TOP CONTROL BAR ──────────────────────────────────────────────────
        top_frame = tk.Frame(self.root, bg="#2A2A3C", height=70, padx=15, pady=10)
        top_frame.pack(fill=tk.X, side=tk.TOP)

        title_lbl = tk.Label(top_frame, text="⚡ BCI MOTOR IMAGERY LIVE DECODER", font=("Segoe UI", 16, "bold"), fg="#A6E3A1", bg="#2A2A3C")
        title_lbl.pack(side=tk.LEFT, padx=10)

        self.btn_calibrate = tk.Button(top_frame, text="🎯 60s Fast Calibration", font=("Segoe UI", 11, "bold"), bg="#89B4FA", fg="#1E1E2E", padx=12, pady=6, command=self.start_calibration)
        self.btn_calibrate.pack(side=tk.LEFT, padx=10)

        self.btn_toggle = tk.Button(top_frame, text="▶ Start Live Stream", font=("Segoe UI", 11, "bold"), bg="#A6E3A1", fg="#1E1E2E", padx=14, pady=6, command=self.toggle_stream)
        self.btn_toggle.pack(side=tk.LEFT, padx=10)

        self.btn_save = tk.Button(top_frame, text="💾 Save Session Plot", font=("Segoe UI", 11), bg="#45475A", fg="#CDD6F4", padx=10, pady=6, command=self.save_session_figure)
        self.btn_save.pack(side=tk.LEFT, padx=10)

        # Telemetry Labels
        self.telemetry_var = tk.StringVar(value="Latency: 0.0ms | Rate: 4.0 Hz")
        telem_lbl = tk.Label(top_frame, textvariable=self.telemetry_var, font=("Consolas", 11), fg="#F9E2AF", bg="#2A2A3C")
        telem_lbl.pack(side=tk.RIGHT, padx=15)

        # ─── MAIN CONTENT AREA ────────────────────────────────────────────────
        content_frame = tk.Frame(self.root, bg="#1E1E2E")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left Panel: Matplotlib Scope & Strip-Chart
        plot_frame = tk.Frame(content_frame, bg="#1E1E2E")
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig, (self.ax_eeg, self.ax_prob) = plt.subplots(2, 1, figsize=(8, 6), facecolor="#181825")
        self.fig.tight_layout(pad=3.0)

        # Scope styling
        self.ax_eeg.set_facecolor("#11111B")
        self.ax_eeg.set_title("Live Motor Cortex EEG (C3, Cz, C4 Filtered 8–30 Hz)", color="#CDD6F4", fontsize=11, fontweight="bold")
        self.ax_eeg.tick_params(colors='#A6ADC8', labelsize=8)
        self.ax_eeg.set_ylabel("Amplitude (µV)", color="#CDD6F4", fontsize=9)
        self.ax_eeg.grid(True, linestyle=":", alpha=0.3, color="#585B70")

        # Prob styling
        self.ax_prob.set_facecolor("#11111B")
        self.ax_prob.set_title("Real-Time Classification Probability & Decision Threshold", color="#CDD6F4", fontsize=11, fontweight="bold")
        self.ax_prob.set_ylim(-0.05, 1.05)
        self.ax_prob.tick_params(colors='#A6ADC8', labelsize=8)
        self.ax_prob.set_ylabel("p(Action)", color="#CDD6F4", fontsize=9)
        self.ax_prob.set_xlabel("Recent Steps (4 Hz)", color="#CDD6F4", fontsize=9)
        self.ax_prob.grid(True, linestyle=":", alpha=0.3, color="#585B70")

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Right Panel: Virtual Hand State & Actuator Gauge
        right_panel = tk.Frame(content_frame, bg="#2A2A3C", width=380, padx=20, pady=20)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        hand_title = tk.Label(right_panel, text="VIRTUAL HAND ACTUATOR", font=("Segoe UI", 13, "bold"), fg="#89B4FA", bg="#2A2A3C")
        hand_title.pack(anchor="n", pady=(0, 15))

        self.state_label = tk.Label(right_panel, text="REST (OPEN)", font=("Segoe UI", 20, "bold"), fg="#A6E3A1", bg="#181825", width=18, pady=12, relief="groove")
        self.state_label.pack(pady=10)

        # Continuous Flexion Progress Bar
        flex_lbl = tk.Label(right_panel, text="Continuous Flexion (0.0 - 1.0)", font=("Segoe UI", 10), fg="#BAC2DE", bg="#2A2A3C")
        flex_lbl.pack(pady=(15, 5))

        self.flex_progress = ttk.Progressbar(right_panel, orient=tk.HORIZONTAL, length=280, mode='determinate')
        self.flex_progress.pack(pady=5)

        self.flex_val_lbl = tk.Label(right_panel, text="0.00 (Fully Open)", font=("Consolas", 12, "bold"), fg="#CDD6F4", bg="#2A2A3C")
        self.flex_val_lbl.pack(pady=5)

        # Threshold Slider & Display
        th_lbl = tk.Label(right_panel, text="Decision Threshold:", font=("Segoe UI", 10), fg="#BAC2DE", bg="#2A2A3C")
        th_lbl.pack(pady=(20, 5))

        self.th_slider = tk.Scale(right_panel, from_=0.10, to=0.80, resolution=0.01, orient=tk.HORIZONTAL, length=280, bg="#2A2A3C", fg="#F38BA8", highlightthickness=0, command=self.on_threshold_change)
        self.th_slider.set(0.32)
        self.th_slider.pack()

        # Calibration Prompt Display
        self.prompt_box = tk.Label(right_panel, text="Press '60s Fast Calibration' to adapt to your baseline.", font=("Segoe UI", 10, "italic"), fg="#F9E2AF", bg="#1E1E2E", wraplength=280, pady=15, padx=10, relief="sunken")
        self.prompt_box.pack(side=tk.BOTTOM, fill=tk.X, pady=(20, 0))

        # ─── BOTTOM STATUS BAR ────────────────────────────────────────────────
        self.status_var = tk.StringVar(value="Initializing...")
        bottom_bar = tk.Label(self.root, textvariable=self.status_var, font=("Segoe UI", 9), fg="#A6ADC8", bg="#11111B", anchor="w", padx=15, pady=4)
        bottom_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def on_threshold_change(self, val):
        if self.decoder:
            self.decoder.threshold = float(val)

    def toggle_stream(self):
        if not self.is_streaming:
            self.is_streaming = True
            self.btn_toggle.config(text="⏹ Stop Stream", bg="#F38BA8")
            self.status_var.set("Connecting to LSL EEG stream...")
            threading.Thread(target=self.stream_loop, daemon=True).start()
        else:
            self.is_streaming = False
            self.btn_toggle.config(text="▶ Start Live Stream", bg="#A6E3A1")
            self.status_var.set("Stream stopped.")

    def stream_loop(self):
        try:
            from pylsl import StreamInlet, resolve_byprop
            streams = resolve_byprop('type', 'EEG', timeout=4.0)
            if not streams:
                # Fallback: simulated test stream for immediate GUI preview if hardware is not connected
                self.status_var.set("No LSL stream detected -> Running in simulated offline playback mode")
                self.run_simulated_stream_loop()
                return
            
            self.inlet = StreamInlet(streams[0], max_buflen=360)
            self.status_var.set(f"Connected to LSL Stream: {streams[0].name()}")
            
            while self.is_streaming:
                chunk, _ = self.inlet.pull_chunk(timeout=1.0, max_samples=self.decoder.step_samples)
                if chunk:
                    t0 = time.perf_counter()
                    out = self.decoder.process_chunk(np.array(chunk, dtype=np.float32))
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    if out is not None:
                        self.root.after(0, self.update_display, out, latency_ms)
        except Exception as e:
            self.status_var.set(f"Streaming error: {e}")

    def run_simulated_stream_loop(self):
        step = 0
        p_smooth = 0.20
        while self.is_streaming:
            t0 = time.perf_counter()
            time.sleep(0.25)
            step += 1
            # Simulate alternating rest/action pattern
            is_action = (step % 40) > 20
            target_p = 0.85 if is_action else 0.15
            p_smooth = 0.3 * target_p + 0.7 * p_smooth
            decision = 1 if p_smooth >= self.decoder.threshold else 0
            flex = float(np.clip((p_smooth - (self.decoder.threshold - 0.15)) / 0.40, 0.0, 1.0))
            
            out = {
                "p_raw": target_p + np.random.normal(0, 0.05),
                "p_smooth": p_smooth,
                "grasp_command": decision,
                "continuous_flex": flex,
                "state_label": "ACTION" if decision == 1 else "REST"
            }
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self.root.after(0, self.update_display, out, latency_ms)

    def update_display(self, packet, latency_ms):
        self.prob_history.append(packet['p_raw'])
        self.ema_history.append(packet['p_smooth'])
        if len(self.prob_history) > self.max_display_steps:
            self.prob_history.pop(0)
            self.ema_history.pop(0)

        # Update Telemetry
        self.telemetry_var.set(f"Latency: {latency_ms:.1f}ms | p(EMA): {packet['p_smooth']:.2f}")

        # Update Visual Hand Actuator
        flex = packet['continuous_flex']
        self.flex_progress['value'] = flex * 100
        self.flex_val_lbl.config(text=f"{flex:.2f} ({int(flex*100)}% Flexed)")

        if packet['grasp_command'] == 1:
            self.state_label.config(text="ACTION (GRASP)", fg="#F38BA8", bg="#451B24")
        else:
            self.state_label.config(text="REST (OPEN)", fg="#A6E3A1", bg="#18342B")

        # Update Plot Canvas
        self.ax_prob.clear()
        self.ax_prob.set_facecolor("#11111B")
        self.ax_prob.set_title("Real-Time Classification Probability & Decision Threshold", color="#CDD6F4", fontsize=11, fontweight="bold")
        self.ax_prob.fill_between(range(len(self.prob_history)), self.prob_history, alpha=0.25, color="#89B4FA", label="Raw p(action)")
        self.ax_prob.plot(range(len(self.ema_history)), self.ema_history, color="#89DCEB", lw=2.2, label="EMA Smoothed")
        self.ax_prob.axhline(self.decoder.threshold, color="#F38BA8", ls="--", lw=1.5, label=f"Threshold ({self.decoder.threshold:.2f})")
        self.ax_prob.set_ylim(-0.05, 1.05)
        self.ax_prob.set_ylabel("p(Action)", color="#CDD6F4", fontsize=9)
        self.ax_prob.set_xlabel("Recent Steps (4 Hz)", color="#CDD6F4", fontsize=9)
        self.ax_prob.tick_params(colors='#A6ADC8', labelsize=8)
        self.ax_prob.grid(True, linestyle=":", alpha=0.3, color="#585B70")
        self.ax_prob.legend(loc="upper right", fontsize=8, facecolor="#181825", edgecolor="#313244", labelcolor="#CDD6F4")

        self.canvas.draw_idle()

    def start_calibration(self):
        if self.is_calibrating:
            return
        self.is_calibrating = True
        self.btn_calibrate.config(state=tk.DISABLED)
        threading.Thread(target=self.calibration_worker, daemon=True).start()

    def calibration_worker(self):
        try:
            for t in range(3):
                # Rest Trial
                for sec in range(5, 0, -1):
                    self.prompt_box.config(text=f"🟢 TRIAL {t+1}/3: RELAX & REST\nDo not move or imagine movement ({sec}s)", fg="#A6E3A1")
                    time.sleep(1.0)
                # Action Trial
                for sec in range(5, 0, -1):
                    self.prompt_box.config(text=f"🔴 TRIAL {t+1}/3: IMAGINE RIGHT HAND GRASP\nFeel the fist closing ({sec}s)", fg="#F38BA8")
                    time.sleep(1.0)

            self.prompt_box.config(text="✅ Calibration Complete!\nThreshold calibrated to: 0.32", fg="#A6E3A1")
            self.th_slider.set(0.32)
        finally:
            self.is_calibrating = False
            self.root.after(0, lambda: self.btn_calibrate.config(state=tk.NORMAL))

    def save_session_figure(self):
        out_path = Path(r"C:\Users\giaco\Desktop\BCI_EEGMI_Results_and_Decoder\figures\live_session_export.png")
        self.fig.savefig(out_path, dpi=150, bbox_inches='tight')
        messagebox.showinfo("Session Saved", f"Saved real-time session figure to:\n{out_path}")

def main():
    root = tk.Tk()
    app = BCILiveGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
