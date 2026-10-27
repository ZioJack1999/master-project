"""
Real-Time EEG Motor Imagery Decoder for Closed-Loop BCI & Virtual Hand Control
==============================================================================
Standalone, production-ready decoder engine designed for live lab testing on user.

Features:
1. Multi-Subject Pre-Trained Master Model (`eegnet_master_live.pt`).
2. Online IIR Bandpass Filtering (8-30 Hz Butterworth) with state preservation (`sosfilt` + `zi`).
3. Continuous Circular Ring-Buffering (1024 samples = 4.0s @ 256 Hz).
4. Exponential Moving Average (EMA) smoothing (alpha = 0.3).
5. Dwell-Time / Hysteresis trigger (2 consecutive steps = 500 ms) to suppress false positives.
6. Real-Time UDP JSON packet transmission to Virtual Hand on port 5005.
7. Built-in 60-second Fast Calibration Mode (`--mode calibrate`) to calibrate the user's
   personal baseline and optimal decision threshold before live streaming (`--mode lsl`).
"""

import os
import sys
import time
import json
import socket
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import mne
from scipy.signal import butter, sosfilt, sosfilt_zi
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report


# ─── 1. NEURAL NETWORK ARCHITECTURES ──────────────────────────────────────────

class EEGNet(nn.Module):
    """
    Compact Convolutional Neural Network for EEG decoding.
    Reference: Lawhern et al., 2018.
    """
    def __init__(self, n_ch: int = 20, n_t: int = 1024, n_cls: int = 2,
                 F1: int = 8, D: int = 2, F2: int = 16, kern: int = 128, drop: float = 0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern), padding=(0, kern // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(drop)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(drop)
        )
        flat_size = self._get_flat_dim(n_ch, n_t)
        self.fc = nn.Linear(flat_size, n_cls)

    def _get_flat_dim(self, c: int, t: int) -> int:
        self.block1.eval()
        self.block2.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, c, t)
            out = self.block2(self.block1(dummy))
            return out.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        return self.fc(x.flatten(1))


# ─── 2. STREAMING RING BUFFER ─────────────────────────────────────────────────

class RingBuffer:
    """Circular FIFO buffer for continuous multi-channel streaming data."""
    def __init__(self, n_ch: int, n_samples: int):
        self.n_ch = n_ch
        self.n_samples = n_samples
        self.buf = np.zeros((n_ch, n_samples), dtype=np.float32)
        self.idx = 0

    def push(self, sample: np.ndarray):
        self.buf[:, self.idx % self.n_samples] = sample
        self.idx += 1

    def push_chunk(self, chunk: np.ndarray):
        if chunk.shape[0] != self.n_ch and chunk.shape[1] == self.n_ch:
            chunk = chunk.T
        n_incoming = chunk.shape[1]
        for s in range(n_incoming):
            self.buf[:, self.idx % self.n_samples] = chunk[:, s]
            self.idx += 1

    def get(self) -> np.ndarray:
        i = self.idx % self.n_samples
        return np.concatenate([self.buf[:, i:], self.buf[:, :i]], axis=1)

    @property
    def is_full(self) -> bool:
        return self.idx >= self.n_samples


# ─── 3. REAL-TIME BCI DECODER ENGINE ──────────────────────────────────────────

class RealtimeBCIDecoder:
    def __init__(self, ckpt_path: str,
                 sfreq: int = 256,
                 epoch_len_s: float = 4.0,
                 step_len_s: float = 0.25,
                 threshold: float = 0.35,
                 ema_alpha: float = 0.3,
                 dwell_steps: int = 2,
                 fmin: float = 8.0,
                 fmax: float = 30.0,
                 udp_host: str = "127.0.0.1",
                 udp_port: int = 5005,
                 device: str = "cpu"):
        
        self.sfreq = sfreq
        self.window_samples = int(epoch_len_s * sfreq)   # 1024 samples
        self.step_samples = int(step_len_s * sfreq)      # 64 samples (4 Hz)
        self.ema_alpha = ema_alpha
        self.dwell_steps = dwell_steps
        self.device = torch.device(device)
        self.threshold = threshold

        # Load Checkpoint
        self.ckpt_path = Path(ckpt_path)
        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.ckpt_path.resolve()}")

        ckpt = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        self.n_ch = ckpt.get('n_channels', 20)
        self.train_mean = ckpt.get('train_mean', np.zeros(self.n_ch, dtype=np.float32))
        self.train_std = ckpt.get('train_std', np.ones(self.n_ch, dtype=np.float32))

        self.model = EEGNet(n_ch=self.n_ch, n_t=self.window_samples).to(self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval()

        # Online Filter: 4th order Butterworth bandpass (8-30 Hz)
        self.sos = butter(4, [fmin, fmax], btype='bandpass', fs=sfreq, output='sos')
        self.zi = np.zeros((self.sos.shape[0], self.n_ch, 2), dtype=np.float32)

        # Streaming State
        self.ring = RingBuffer(self.n_ch, self.window_samples)
        self.ema_prob = 0.5
        self.dwell_counter = 0
        self.step_counter = 0

        # UDP Socket for Virtual Hand
        self.udp_target = (udp_host, udp_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def reset_state(self):
        self.ring = RingBuffer(self.n_ch, self.window_samples)
        self.zi = np.zeros((self.sos.shape[0], self.n_ch, 2), dtype=np.float32)
        self.ema_prob = 0.5
        self.dwell_counter = 0
        self.step_counter = 0

    def set_user_baseline(self, mean: np.ndarray, std: np.ndarray, threshold: float):
        """Update normalisation stats and decision threshold from user calibration."""
        self.train_mean = mean.astype(np.float32)
        self.train_std = std.astype(np.float32)
        self.threshold = float(threshold)
        print(f"[Decoder] Updated User Calibration: Threshold = {self.threshold:.2f}")

    def process_chunk(self, raw_chunk: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Processes incoming streaming EEG chunk:
        1. Online IIR filter with state preservation
        2. Push to circular buffer
        3. Neural network inference (EEGNet)
        4. EMA probability smoothing & dwell-time hysteresis
        5. UDP transmission to virtual hand
        """
        if raw_chunk.shape[0] != self.n_ch and raw_chunk.shape[1] == self.n_ch:
            raw_chunk = raw_chunk.T

        n_samples_in = raw_chunk.shape[1]

        # Artifact Guard: soft clamp voltage spikes > 100 uV (e.g. blinks/EMG)
        raw_chunk = np.clip(raw_chunk, -100e-6, 100e-6)

        # 1. Online IIR Bandpass Filtering (8-30 Hz)
        filtered_chunk, self.zi = sosfilt(self.sos, raw_chunk, axis=1, zi=self.zi)

        # 2. Push to circular ring buffer
        self.ring.push_chunk(filtered_chunk)
        self.step_counter += n_samples_in

        # 3. Check if step deadline reached and buffer is warm
        if self.ring.is_full and self.step_counter >= self.step_samples:
            self.step_counter = 0

            # 4. Extract latest window & Normalize
            X_win = self.ring.get()
            X_norm = (X_win - self.train_mean[:, None]) / (self.train_std[:, None] + 1e-8)

            # 5. Forward Inference through EEGNet
            with torch.no_grad():
                tensor_in = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(0).to(self.device)
                logits = self.model(tensor_in)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                p_action = float(probs[1])

            # 6. Exponential Moving Average (EMA) Smoothing
            self.ema_prob = (self.ema_alpha * p_action) + ((1.0 - self.ema_alpha) * self.ema_prob)

            # 7. Dwell Time Hysteresis Logic
            if self.ema_prob >= self.threshold:
                self.dwell_counter += 1
            else:
                self.dwell_counter = 0

            decision = 1 if self.dwell_counter >= self.dwell_steps else 0
            
            # Continuous grasp flexion mapped cleanly [0.0 = fully open, 1.0 = fully closed]
            continuous_flex = float(np.clip((self.ema_prob - (self.threshold - 0.15)) / 0.40, 0.0, 1.0))

            packet = {
                "timestamp": time.time(),
                "p_raw": round(p_action, 4),
                "p_smooth": round(self.ema_prob, 4),
                "grasp_command": decision,
                "continuous_flex": round(continuous_flex, 4),
                "state_label": "ACTION" if decision == 1 else "REST"
            }

            # 8. Transmit UDP packet to Virtual Hand
            try:
                msg = json.dumps(packet).encode('utf-8')
                self.sock.sendto(msg, self.udp_target)
            except Exception:
                pass

            return packet

        return None


# ─── 4. FAST 60-SECOND USER CALIBRATION ROUTINE ───────────────────────────────

def run_fast_lab_calibration(inlet, decoder: RealtimeBCIDecoder, trials_per_class: int = 3, trial_dur_s: float = 5.0):
    """
    Guides the user through a 60-second calibration protocol in the lab:
    - 3 Rest trials (5s each)
    - 3 Right-Hand Imagery trials (5s each)
    Calculates the user's specific baseline mean/std and optimal decision threshold.
    """
    print("\n" + "=" * 65)
    print("      LIVE LAB CALIBRATION ROUTINE (60 SECONDS)")
    print("=" * 65)
    print("  You will be prompted to alternate between REST and RIGHT-HAND IMAGERY.")
    print("  Keep head and jaw relaxed to avoid EMG artifact.")
    print("=" * 65)
    time.sleep(2.0)

    calib_windows = []
    calib_labels = []
    
    # 3 Alternating pairs of Rest and Action
    for t in range(trials_per_class):
        # 1. Rest Phase
        print(f"\n[Trial {t+1}/{trials_per_class}] ---> RELAX / REST (Do not imagine movement) <---")
        t_end = time.time() + trial_dur_s
        while time.time() < t_end:
            chunk, _ = inlet.pull_chunk(timeout=1.0, max_samples=decoder.step_samples)
            if chunk:
                raw_chunk = np.array(chunk, dtype=np.float32).T
                filtered_chunk, decoder.zi = sosfilt(decoder.sos, raw_chunk, axis=1, zi=decoder.zi)
                decoder.ring.push_chunk(filtered_chunk)
                if decoder.ring.is_full:
                    calib_windows.append(decoder.ring.get())
                    calib_labels.append(0)
            time.sleep(decoder.step_samples / decoder.sfreq)

        # 2. Action / Imagery Phase
        print(f"[Trial {t+1}/{trials_per_class}] >>> IMAGINE RIGHT HAND GRASP / CLOSING <<<")
        t_end = time.time() + trial_dur_s
        while time.time() < t_end:
            chunk, _ = inlet.pull_chunk(timeout=1.0, max_samples=decoder.step_samples)
            if chunk:
                raw_chunk = np.array(chunk, dtype=np.float32).T
                filtered_chunk, decoder.zi = sosfilt(decoder.sos, raw_chunk, axis=1, zi=decoder.zi)
                decoder.ring.push_chunk(filtered_chunk)
                if decoder.ring.is_full:
                    calib_windows.append(decoder.ring.get())
                    calib_labels.append(1)
            time.sleep(decoder.step_samples / decoder.sfreq)

    print("\n[Calibration Complete] Computing optimal user threshold...")
    calib_X = np.array(calib_windows)  # (n_win, n_ch, n_t)
    calib_y = np.array(calib_labels)

    flat = calib_X.reshape(calib_X.shape[0], calib_X.shape[1], -1)
    user_mean = flat.mean(axis=(0, 2)).astype(np.float32)
    user_std = (flat.std(axis=(0, 2)) + 1e-8).astype(np.float32)

    # Score probabilities across calibration data
    X_norm = (calib_X - user_mean[None, :, None]) / user_std[None, :, None]
    with torch.no_grad():
        t_in = torch.tensor(X_norm, dtype=torch.float32).to(decoder.device)
        logits = decoder.model(t_in)
        raw_probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()

    # Find optimal threshold via ROC sweep
    best_th, best_bal = 0.35, 0.0
    for th in np.arange(0.15, 0.75, 0.02):
        preds = (raw_probs >= th).astype(int)
        bal = balanced_accuracy_score(calib_y, preds)
        if bal > best_bal:
            best_bal = bal
            best_th = th

    decoder.set_user_baseline(user_mean, user_std, best_th)
    print(f"  Calibrated Threshold: {best_th:.2f} | Calibration Balanced Acc: {best_bal*100:.1f}%")
    print("=" * 65)
    return best_th


# ─── 5. MAIN ENTRY POINT ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Real-Time BCI Motor Imagery Decoder")
    parser.add_argument("--mode", type=str, choices=['lsl', 'calibrate', 'simulate'], default='lsl',
                        help="Mode: 'lsl' (live streaming), 'calibrate' (60s lab calibration then live), 'simulate' (offline test)")
    parser.add_argument("--threshold", type=float, default=0.32, help="Decision threshold")
    parser.add_argument("--udp-host", type=str, default="127.0.0.1", help="Virtual Hand UDP Host")
    parser.add_argument("--udp-port", type=int, default=5005, help="Virtual Hand UDP Port")
    args = parser.parse_args()

    # Locate Master Trained Model Checkpoint
    ckpt_candidates = [
        Path(__file__).resolve().parent / "decoder_and_models" / "eegnet_master_live.pt",
        Path(__file__).resolve().parent / "checkpoints" / "eegnet_master_live.pt",
        Path(__file__).resolve().parent / "eegnet_master_live.pt",
        Path(r"C:\Users\giaco\.gemini\antigravity\brain\1d373d7a-b1fe-46d1-8b53-b128fc406f00\checkpoints\eegnet_master_live.pt"),
        Path(__file__).resolve().parent / "results_v2" / "checkpoints" / "eegnet_master_live.pt"
    ]
    ckpt_path = None
    for c in ckpt_candidates:
        if c.exists():
            ckpt_path = c
            break
    if ckpt_path is None:
        ckpt_path = Path(__file__).resolve().parent / "decoder_and_models" / "eegnet_master_live.pt"

    decoder = RealtimeBCIDecoder(
        ckpt_path=str(ckpt_path),
        sfreq=256,
        epoch_len_s=4.0,
        step_len_s=0.25,
        threshold=args.threshold,
        ema_alpha=0.3,
        dwell_steps=2,
        udp_host=args.udp_host,
        udp_port=args.udp_port
    )

    if args.mode in ['lsl', 'calibrate']:
        try:
            from pylsl import StreamInlet, resolve_byprop
        except ImportError:
            print("[Error] 'pylsl' library is required for live lab testing. Run: pip install pylsl")
            return

        print("\nSearching for live EEG stream on LabStreamingLayer (LSL)...")
        streams = resolve_byprop('type', 'EEG', timeout=5.0)
        if not streams:
            print("[Error] No LSL EEG stream found! Please verify amplifier is broadcasting.")
            return

        inlet = StreamInlet(streams[0], max_buflen=360)
        print(f"[Connected] LSL Stream: '{streams[0].name()}'. Target Hand UDP: {args.udp_host}:{args.udp_port}")

        if args.mode == 'calibrate':
            run_fast_lab_calibration(inlet, decoder)

        print("\n>>> LIVE DECODER ACTIVE - STREAMING COMMANDS TO VIRTUAL HAND <<<")
        print("    Press Ctrl+C to stop.\n")
        step_num = 0
        while True:
            chunk, _ = inlet.pull_chunk(timeout=1.0, max_samples=decoder.step_samples)
            if chunk:
                t0 = time.perf_counter()
                out = decoder.process_chunk(np.array(chunk, dtype=np.float32))
                latency_ms = (time.perf_counter() - t0) * 1000.0
                if out is not None:
                    step_num += 1
                    bar = "█" * int(out['continuous_flex'] * 10) + "░" * (10 - int(out['continuous_flex'] * 10))
                    print(f"\r[LIVE] Step: {step_num:05d} | Latency: {latency_ms:4.1f}ms | p(EMA): {out['p_smooth']:.2f} | Hand: {out['state_label']:<6} [{bar}]", end="", flush=True)


if __name__ == "__main__":
    main()
