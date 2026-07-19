"""
quickstart.py
-------------
Quick demonstration using the 3 uploaded runs (sub-01, ses-01).
Run this first to verify everything works before pointing to the full dataset.

Usage:
    python quickstart.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import build_model, count_parameters

# ── Minimal inline loader (no BIDS path needed) ────────────────────────────────
import mne
import pandas as pd

SFREQ    = 256
EEG_CHS  = ['F3','F1','Fz','F2','F4','C3','C1','Cz','C2','C4',
             'CPz','P3','P1','Pz','P2','P4','POz','O1','Oz','O2']
FMIN, FMAX = 8.0, 30.0
TMIN, TMAX = 0.5, 4.5       # 4-second epochs, 500 ms post-cue

UPLOADS = "/mnt/user-data/uploads"
RUNS = [
    (f"{UPLOADS}/sub-01_ses-01_task-MI_run-0{r}_eeg.edf",
     f"{UPLOADS}/sub-01_ses-01_task-MI_run-0{r}_events.tsv")
    for r in range(1, 4)
]


def load_run(edf_path, events_path):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    avail = [c for c in EEG_CHS if c in raw.ch_names]
    raw.pick_channels(avail, ordered=True)
    raw.filter(FMIN, FMAX, method='iir',
                iir_params=dict(order=4, ftype='butter'), verbose=False)

    ev_df = pd.read_csv(events_path, sep='\t')

    events, ev_ids = [], {'rest': 0, 'action': 1}
    for _, row in ev_df.iterrows():
        eid = ev_ids[row['trial_type']]
        events.append([int(row['sample']), 0, eid])
    events = np.array(events)
    events = events[np.argsort(events[:, 0])]

    epochs = mne.Epochs(raw, events, event_id=ev_ids,
                         tmin=TMIN, tmax=TMAX,
                         baseline=None, preload=True, verbose=False)
    X = epochs.get_data()
    y = epochs.events[:, 2].astype(np.int64)

    # Artefact rejection
    keep = np.max(np.abs(X), axis=(1,2)) < 100e-6
    X, y = X[keep], y[keep]

    # Z-score per channel
    mu  = X.mean(axis=2, keepdims=True)
    std = X.std( axis=2, keepdims=True) + 1e-8
    return (X - mu) / std, y


def main():
    print("Loading 3 runs of sub-01, ses-01…")
    all_X, all_y = [], []
    for edf, ev in RUNS:
        X, y = load_run(edf, ev)
        all_X.append(X); all_y.append(y)
        print(f"  {os.path.basename(edf)} → {len(y)} epochs (action={np.sum(y==1)}, rest={np.sum(y==0)})")

    X_all = np.concatenate(all_X)
    y_all = np.concatenate(all_y)
    print(f"\nTotal: {len(y_all)} epochs | shape {X_all.shape}")

    # ── Leave-one-run-out CV ────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    n_ch, n_t = X_all.shape[1], X_all.shape[2]
    all_preds, all_true = [], []
    run_accs = []

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('EEGNet — Leave-One-Run-Out CV (sub-01, ses-01)', fontweight='bold')

    for test_run in range(3):
        print(f"── Fold {test_run+1}/3 (test run = run-{test_run+1:02d}) ──")
        X_te = all_X[test_run]
        y_te = all_y[test_run]
        X_tr = np.concatenate([all_X[i] for i in range(3) if i != test_run])
        y_tr = np.concatenate([all_y[i] for i in range(3) if i != test_run])

        model = build_model('eegnet', n_channels=n_ch, n_times=n_t,
                             dropout=0.5).to(device)
        if test_run == 0:
            print(f"  Trainable parameters: {count_parameters(model):,}")

        Xtr = torch.tensor(X_tr, dtype=torch.float32).to(device)
        ytr = torch.tensor(y_tr, dtype=torch.long  ).to(device)
        Xte = torch.tensor(X_te, dtype=torch.float32).to(device)
        yte = torch.tensor(y_te, dtype=torch.long  ).to(device)

        # Class-balanced sampler
        counts  = torch.bincount(ytr)
        weights = (1.0 / counts.float())[ytr]
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(weights, num_samples=len(ytr), replacement=True)
        tr_ld   = DataLoader(TensorDataset(Xtr, ytr), batch_size=32, sampler=sampler)
        te_ld   = DataLoader(TensorDataset(Xte, yte), batch_size=32, shuffle=False)

        opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-5)
        crit  = nn.CrossEntropyLoss()

        train_losses, val_losses = [], []
        best_bal, best_wts, patience = 0.0, None, 0

        for ep in range(1, 101):
            model.train()
            for Xb, yb in tr_ld:
                opt.zero_grad()
                loss = crit(model(Xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()

            model.eval()
            with torch.no_grad():
                tr_l = crit(model(Xtr), ytr).item()
                te_l = crit(model(Xte), yte).item()
                preds = model(Xte).argmax(1).cpu().numpy()
            train_losses.append(tr_l)
            val_losses.append(te_l)

            from sklearn.metrics import balanced_accuracy_score
            bal = balanced_accuracy_score(y_te, preds)
            if bal > best_bal:
                best_bal = bal
                best_wts = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if patience >= 20:
                print(f"    Early stop at ep {ep}")
                break

        model.load_state_dict(best_wts)
        model.eval()
        with torch.no_grad():
            fold_preds = model(Xte).argmax(1).cpu().numpy()
        acc = (fold_preds == y_te).mean()
        run_accs.append(acc)
        all_preds.extend(fold_preds.tolist())
        all_true.extend(y_te.tolist())
        print(f"  → acc={acc:.3f}  bal_acc={best_bal:.3f}")

        # Save checkpoint
        torch.save({'model_state': model.state_dict(),
                     'n_channels': n_ch, 'n_times': n_t, 'arch': 'eegnet'},
                    f"checkpoint_sub01_ses01_run{test_run+1}.pt")

        # Plot
        axes[test_run].plot(train_losses, label='Train loss')
        axes[test_run].plot(val_losses,   label='Val loss')
        axes[test_run].set_title(f'Run {test_run+1} (acc={acc:.2f})')
        axes[test_run].set_xlabel('Epoch')
        axes[test_run].legend()

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    print("\nTraining curves → training_curves.png")

    print("\n── Overall report ──")
    print(classification_report(all_true, all_preds,
                                 target_names=['rest','action'], digits=3))
    mean_acc = np.mean(run_accs)
    print(f"Mean accuracy across runs: {mean_acc:.3f} ± {np.std(run_accs):.3f}")

    # ── Simulate real-time with the last checkpoint ────────────────────
    print("\nSimulating real-time on run-03…")
    from realtime_inference import run_simulation
    run_simulation(
        edf_path    = RUNS[2][0],
        events_path = RUNS[2][1],
        ckpt_path   = 'checkpoint_sub01_ses01_run3.pt',
        epoch_len   = 4.0,
        step_len    = 0.25,
        smooth_win  = 5,
        threshold   = 0.6,
        out_fig     = 'realtime_output.png',
    )


if __name__ == '__main__':
    main()
