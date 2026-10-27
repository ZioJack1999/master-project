"""
Standalone 60-Second EEG Calibration Routine
=============================================
Calibrates the user's specific resting baseline voltage, mu/beta power,
and optimal decision threshold before beginning closed-loop testing.

Usage:
    python run_calibration.py [--threshold 0.32] [--lsl-stream EEG]
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np

# Ensure parent path is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from realtime_bci_decoder import RealtimeBCIDecoder, run_fast_lab_calibration

def main():
    parser = argparse.ArgumentParser(description="8-Second EEG User Calibration Routine")
    parser.add_argument("--trials", type=int, default=3, help="Number of Rest/Action trial pairs (default: 3)")
    parser.add_argument("--trial-dur", type=float, default=8.0, help="Duration of each trial in seconds (default: 8.0s)")
    parser.add_argument("--ckpt", type=str, default="eegnet_master_live.pt", help="Base model checkpoint")
    parser.add_argument("--udp-port", type=int, default=5005, help="Target UDP port for Virtual Hand")
    args = parser.parse_args()

    ckpt_path = SCRIPT_DIR / "decoder_and_models" / args.ckpt
    if not ckpt_path.exists():
        ckpt_path = SCRIPT_DIR / args.ckpt
    if not ckpt_path.exists():
        ckpt_path = Path(r"C:\Users\giaco\.gemini\antigravity\brain\1d373d7a-b1fe-46d1-8b53-b128fc406f00\checkpoints") / args.ckpt

    print("=" * 70)
    print("        8-SECOND EEG USER CALIBRATION & BASELINE LOCK")
    print("=" * 70)
    print(f"  Base Model Checkpoint : {ckpt_path.name}")
    print(f"  Calibration Protocol  : {args.trials} pairs x {args.trial_dur:.1f}s = {args.trials * 2 * args.trial_dur:.1f}s total")
    print("-" * 70)

    decoder = RealtimeBCIDecoder(
        ckpt_path=str(ckpt_path),
        sfreq=256,
        epoch_len_s=4.0,
        step_len_s=0.25,
        threshold=0.32,
        ema_alpha=0.3,
        dwell_steps=2,
        udp_port=args.udp_port
    )

    try:
        from pylsl import StreamInlet, resolve_byprop
    except ImportError:
        print("[Error] 'pylsl' library is required. Run: pip install pylsl")
        return

    print("Searching for live EEG stream on LabStreamingLayer (LSL)...")
    streams = resolve_byprop('type', 'EEG', timeout=5.0)
    if not streams:
        print("[Error] No LSL EEG stream found! Please verify amplifier software is broadcasting.")
        return

    inlet = StreamInlet(streams[0], max_buflen=360)
    print(f"[Connected] LSL Stream: '{streams[0].name()}'")

    best_th = run_fast_lab_calibration(inlet, decoder, trials_per_class=args.trials, trial_dur_s=args.trial_dur)

    print("\n[OK] Calibration parameters saved to decoder memory.")
    print("     Starting live streaming to virtual hand on UDP port", args.udp_port)
    print("     Press Ctrl+C to stop.\n")

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
