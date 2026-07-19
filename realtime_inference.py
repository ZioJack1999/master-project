"""
realtime_inference.py
---------------------
Simulates real-time Motor Imagery classification using a pre-trained EEGNet.

The pipeline mirrors what would happen with a live EEG stream:
  1. Raw EEG arrives sample-by-sample (simulated from saved .edf files)
  2. A sliding window buffer is maintained
  3. Each time the buffer is full (epoch_len s), it is:
       • Band-pass filtered
       • Z-score normalized
       • Fed through the neural network
  4. A smoothed probability trace is displayed

Usage:
    python realtime_inference.py \
        --edf   sub-01_ses-01_task-MI_run-01_eeg.edf \
        --events sub-01_ses-01_task-MI_run-01_events.tsv \
        --ckpt  checkpoint_sub01_ses01.pt
"""

import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import mne
import pandas as pd
from collections import deque
from scipy.signal import butter, sosfilt, sosfilt_zi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from model import build_model
from data_loader import EEG_CHANNELS, SFREQ, FMIN, FMAX


# ── IIR filter helpers ─────────────────────────────────────────────────────────

def build_bandpass_filter(fmin=FMIN, fmax=FMAX, fs=SFREQ, order=4):
    """Build 4th-order Butterworth bandpass SOS filter."""
    sos = butter(order, [fmin, fmax], btype='bandpass', fs=fs, output='sos')
    return sos


def init_filter_state(sos, n_channels):
    """Initialise filter state for each channel (zero-phase start)."""
    zi = np.stack([sosfilt_zi(sos) for _ in range(n_channels)], axis=1)
    # zi shape: (n_sections, n_channels, 2)
    return zi


# ── Real-time classifier ───────────────────────────────────────────────────────

class RealTimeClassifier:
    """
    Simulates online EEG classification using a sliding window.

    Parameters
    ----------
    model_path : str
        Path to a .pt checkpoint saved by train.py
    epoch_len : float
        Length of classification window in seconds (default 4 s).
    step_len : float
        Step between consecutive windows (default 0.25 s = 250 ms latency).
    smooth_win : int
        Number of consecutive predictions to average for smoothing.
    threshold : float
        Probability threshold to output 'action' decision.
    device : str
        'cpu', 'cuda', or 'mps'.
    """

    def __init__(self,
                 model_path: str,
                 epoch_len: float  = 4.0,
                 step_len: float   = 0.25,
                 smooth_win: int   = 5,
                 threshold: float  = 0.6,
                 device: str       = 'cpu'):

        self.epoch_samples = int(epoch_len * SFREQ)
        self.step_samples  = int(step_len  * SFREQ)
        self.smooth_win    = smooth_win
        self.threshold     = threshold
        self.device        = torch.device(device)

        # ── Load model ────────────────────────────────────────────────
        ckpt = torch.load(model_path, map_location=self.device)
        arch  = ckpt.get('arch', 'eegnet')
        n_ch  = ckpt['n_channels']
        n_t   = ckpt['n_times']
        self.n_channels = n_ch

        self.model = build_model(arch, n_channels=n_ch, n_times=n_t)
        self.model.load_state_dict(ckpt['model_state'])
        self.model.to(self.device).eval()
        print(f"Loaded {arch} checkpoint — n_channels={n_ch}, n_times={n_t}")

        # ── Ring buffer: (n_channels, epoch_samples) ──────────────────
        self.buffer    = np.zeros((n_ch, self.epoch_samples), dtype=np.float32)
        self.buf_count = 0       # samples written since last classification
        self.sample_n  = 0       # global sample counter

        # ── Online filter state ────────────────────────────────────────
        self.sos = build_bandpass_filter()
        self.zi  = init_filter_state(self.sos, n_ch)

        # ── Probability smoothing ──────────────────────────────────────
        self.prob_history = deque(maxlen=smooth_win)
        self.decisions    = []   # (sample_n, raw_prob, smoothed_prob, decision)

    def push_sample(self, sample: np.ndarray):
        """
        Push one multi-channel sample into the buffer.

        Parameters
        ----------
        sample : np.ndarray, shape (n_channels,)
        """
        # Online bandpass filter — apply sample-by-sample
        sample_2d = sample[:, np.newaxis]           # (C, 1)
        filt_out  = np.zeros_like(sample_2d)
        for c in range(self.n_channels):
            out, self.zi[:, c] = sosfilt(self.sos,
                                          sample_2d[c],
                                          zi=self.zi[:, c])
            filt_out[c] = out

        # Shift buffer and insert
        self.buffer = np.roll(self.buffer, -1, axis=1)
        self.buffer[:, -1] = filt_out[:, 0]

        self.buf_count += 1
        self.sample_n  += 1

        # Classify every step_samples after the buffer is full
        if self.sample_n >= self.epoch_samples and \
                self.buf_count >= self.step_samples:
            self.buf_count = 0
            return self._classify()
        return None

    def _classify(self):
        """Run inference on the current buffer content."""
        X = self.buffer.copy()
        # Z-score normalise
        X = (X - X.mean(axis=1, keepdims=True)) / \
            (X.std(axis=1,  keepdims=True) + 1e-8)

        tensor = torch.tensor(X, dtype=torch.float32) \
                       .unsqueeze(0).to(self.device)   # (1, C, T)

        with torch.no_grad():
            prob_action = F.softmax(self.model(tensor), dim=1)[0, 1].item()

        self.prob_history.append(prob_action)
        smoothed = np.mean(self.prob_history)
        decision = int(smoothed >= self.threshold)

        self.decisions.append((self.sample_n, prob_action, smoothed, decision))
        return prob_action, smoothed, decision

    def push_chunk(self, chunk: np.ndarray):
        """
        Push a chunk of samples. Convenient for batch simulation.

        Parameters
        ----------
        chunk : np.ndarray, shape (n_channels, n_samples)

        Returns
        -------
        List of (prob_action, smoothed_prob, decision) tuples (one per trigger).
        """
        results = []
        for i in range(chunk.shape[1]):
            r = self.push_sample(chunk[:, i])
            if r is not None:
                results.append(r)
        return results


# ── Simulation & visualisation ─────────────────────────────────────────────────

def run_simulation(edf_path: str, events_path: str, ckpt_path: str,
                   epoch_len=4.0, step_len=0.25, smooth_win=5,
                   threshold=0.6, out_fig='realtime_output.png'):
    """
    Simulate real-time classification on a saved EDF file.
    Produces a figure comparing predicted vs true labels over time.
    """
    # ── Load raw EEG ──────────────────────────────────────────────────
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    available = [ch for ch in EEG_CHANNELS if ch in raw.ch_names]
    raw.pick_channels(available, ordered=True)

    data = raw.get_data()      # (n_channels, n_samples)
    n_ch, n_samples = data.shape
    times = np.arange(n_samples) / SFREQ

    # ── Load events ────────────────────────────────────────────────────
    events_df = pd.read_csv(events_path, sep='\t')

    # Build ground-truth label trace (0 = rest, 1 = action, -1 = unlabeled)
    gt = np.full(n_samples, -1, dtype=int)
    for _, row in events_df.iterrows():
        start = int(row['sample'])
        dur   = int(row['duration'] * SFREQ)
        label = 1 if row['trial_type'] == 'action' else 0
        gt[start:start + dur] = label

    # ── Initialise classifier ──────────────────────────────────────────
    clf = RealTimeClassifier(ckpt_path, epoch_len=epoch_len,
                              step_len=step_len, smooth_win=smooth_win,
                              threshold=threshold)

    # ── Stream data ────────────────────────────────────────────────────
    print("Simulating real-time stream…")
    t0 = time.time()
    clf.push_chunk(data)
    elapsed = time.time() - t0
    print(f"Streamed {n_samples/SFREQ:.1f} s of EEG in {elapsed:.2f} s wall-clock")

    # ── Extract results ────────────────────────────────────────────────
    if not clf.decisions:
        print("No decisions made — check checkpoint compatibility.")
        return

    dec_arr = np.array(clf.decisions)           # (N, 4): sample, raw, smooth, decision
    dec_samples   = dec_arr[:, 0].astype(int)
    raw_probs     = dec_arr[:, 1]
    smooth_probs  = dec_arr[:, 2]
    decisions     = dec_arr[:, 3].astype(int)

    dec_times = dec_samples / SFREQ

    # ── Accuracy against ground truth ─────────────────────────────────
    labeled_mask = gt[dec_samples] >= 0
    if labeled_mask.sum() > 0:
        true_labels = gt[dec_samples[labeled_mask]]
        pred_labels = decisions[labeled_mask]
        acc = (true_labels == pred_labels).mean()
        print(f"\nOnline accuracy (labeled windows): {acc*100:.1f}%")

    # ── Plot ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
    fig.suptitle('Real-Time Motor Imagery Classification', fontsize=14, fontweight='bold')

    # Ground truth shading
    for ax in axes:
        prev, prev_t = -1, 0
        for ev in events_df.itertuples():
            col = '#AEDEA7' if ev.trial_type == 'action' else '#AEC6E8'
            ax.axvspan(ev.onset, ev.onset + ev.duration, color=col, alpha=0.3, lw=0)

    # Panel 1: EEG (Cz channel)
    cz_idx = available.index('Cz') if 'Cz' in available else 0
    axes[0].plot(times, data[cz_idx] * 1e6, lw=0.4, color='#333', alpha=0.8)
    axes[0].set_ylabel('Cz (µV)')
    axes[0].set_title('Raw EEG — Cz channel')

    # Panel 2: Raw + smoothed probability
    axes[1].plot(dec_times, raw_probs,    lw=0.8, color='#7FC8F8', label='Raw p(action)', alpha=0.6)
    axes[1].plot(dec_times, smooth_probs, lw=1.5, color='#1B6CA8', label='Smoothed p(action)')
    axes[1].axhline(threshold, color='r', ls='--', lw=1, label=f'Threshold ({threshold})')
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel('p(action)')
    axes[1].set_title('Classification Probability')
    axes[1].legend(loc='upper right', fontsize=8)

    # Panel 3: Binary decisions vs ground truth
    axes[2].step(dec_times, decisions, lw=1.5, color='#D62728', label='Prediction', where='post')
    gt_vis = np.where(gt >= 0, gt, np.nan)
    axes[2].plot(times, gt_vis, lw=1.5, color='#2CA02C', ls='--', label='Ground truth', alpha=0.8)
    axes[2].set_ylim(-0.1, 1.3)
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(['REST', 'ACTION'])
    axes[2].set_xlabel('Time (s)')
    axes[2].set_title('Decision vs Ground Truth')
    axes[2].legend(loc='upper right', fontsize=8)

    # Legend for shading
    p1 = mpatches.Patch(color='#AEDEA7', alpha=0.5, label='Action period')
    p2 = mpatches.Patch(color='#AEC6E8', alpha=0.5, label='Rest period')
    axes[0].legend(handles=[p1, p2], loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(out_fig, dpi=150)
    print(f"Figure saved → {out_fig}")
    return clf


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Real-time MI simulation")
    p.add_argument('--edf',       required=True)
    p.add_argument('--events',    required=True)
    p.add_argument('--ckpt',      required=True)
    p.add_argument('--epoch_len', type=float, default=4.0)
    p.add_argument('--step',      type=float, default=0.25)
    p.add_argument('--smooth',    type=int,   default=5)
    p.add_argument('--threshold', type=float, default=0.6)
    p.add_argument('--out',       default='realtime_output.png')
    args = p.parse_args()

    run_simulation(
        edf_path    = args.edf,
        events_path = args.events,
        ckpt_path   = args.ckpt,
        epoch_len   = args.epoch_len,
        step_len    = args.step,
        smooth_win  = args.smooth,
        threshold   = args.threshold,
        out_fig     = args.out,
    )
