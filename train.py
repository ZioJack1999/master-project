"""
train.py
--------
Training pipeline for the Motor Imagery EEGNet.

Evaluation strategies:
  • within_subject  — train/test on same subject, leave-one-session-out CV
  • cross_subject   — leave-one-subject-out CV (subject-independent BCI)

Usage (command line):
    python train.py --root "C:/Users/giaco/Box/EEGMI" --mode within_subject
    python train.py --root "C:/Users/giaco/Box/EEGMI" --mode cross_subject --arch transformer

Usage (import):
    from train import Trainer
    trainer = Trainer(root="...", mode="within_subject")
    trainer.run()
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_loader import MIDataLoader, N_SUBJECTS, N_SESSIONS
from model import build_model, count_parameters


# ── Hyperparameters ────────────────────────────────────────────────────────────
DEFAULTS = dict(
    arch        = 'eegnet',
    epochs      = 150,
    batch_size  = 32,
    lr          = 1e-3,
    weight_decay= 1e-4,
    patience    = 20,          # early stopping patience
    dropout     = 0.5,
    seed        = 42,
)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device('cuda')
    elif torch.backends.mps.is_available():
        dev = torch.device('mps')
    else:
        dev = torch.device('cpu')
    print(f"Using device: {dev}")
    return dev


# ── Dataset helpers ────────────────────────────────────────────────────────────

def to_tensors(X, y, device):
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    yt = torch.tensor(y, dtype=torch.long).to(device)
    return Xt, yt


def make_loader(X, y, batch_size, shuffle=True, balance=False):
    dataset = TensorDataset(X, y)
    if balance and shuffle:
        # Compute per-sample weights for class balancing
        counts  = torch.bincount(y)
        weights = (1.0 / counts.float())[y]
        sampler = WeightedRandomSampler(weights, num_samples=len(y), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ── Core training loop ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for Xb, yb in loader:
        optimizer.zero_grad()
        logits = model(Xb)
        loss   = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct    += (logits.argmax(1) == yb).sum().item()
        total      += len(yb)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_true = 0.0, [], []
    for Xb, yb in loader:
        logits = model(Xb)
        loss   = criterion(logits, yb)
        total_loss += loss.item() * len(yb)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_true.extend(yb.cpu().numpy())
    n   = len(all_true)
    acc = accuracy_score(all_true, all_preds)
    bal = balanced_accuracy_score(all_true, all_preds)
    return total_loss / n, acc, bal, np.array(all_preds), np.array(all_true)


def fit(model, train_loader, val_loader, cfg, device, fold_label=""):
    """
    Full training loop with early stopping and cosine LR annealing.
    Returns history dict.
    """
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=cfg['lr'],
                                   weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['epochs'], eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    history   = dict(train_loss=[], val_loss=[], train_acc=[], val_acc=[], val_bal=[])
    best_val_bal  = 0.0
    best_weights  = None
    patience_ctr  = 0

    for ep in range(1, cfg['epochs'] + 1):
        t_loss, t_acc           = train_one_epoch(model, train_loader, optimizer,
                                                    criterion, device)
        v_loss, v_acc, v_bal, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)
        history['val_bal'].append(v_bal)

        if v_bal > best_val_bal:
            best_val_bal = v_bal
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if ep % 10 == 0 or ep == 1:
            print(f"  [{fold_label}] Ep {ep:03d} | "
                  f"loss {t_loss:.4f}/{v_loss:.4f} | "
                  f"acc {t_acc:.3f}/{v_acc:.3f} | "
                  f"bal_acc {v_bal:.3f}  {'★' if patience_ctr == 0 else ''}")

        if patience_ctr >= cfg['patience']:
            print(f"  Early stopping at epoch {ep}")
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    return history, best_val_bal


# ── Evaluation strategies ──────────────────────────────────────────────────────

class Trainer:
    """
    Orchestrates training and evaluation.

    Parameters
    ----------
    root : str
        Dataset root path.
    mode : str
        'within_subject' or 'cross_subject'
    arch : str
        Model architecture: 'eegnet' or 'transformer'
    cfg : dict
        Hyperparameters (merged with DEFAULTS).
    subjects : list, optional
        Subset of subject IDs to process. None = all.
    """

    def __init__(self, root: str,
                 mode: str   = 'within_subject',
                 arch: str   = 'eegnet',
                 cfg: dict   = None,
                 subjects    = None):
        self.root     = root
        self.mode     = mode
        self.arch     = arch
        self.cfg      = {**DEFAULTS, **(cfg or {}), 'arch': arch}
        self.subjects = subjects
        self.device   = get_device()
        set_seed(self.cfg['seed'])

    def run(self):
        print(f"\n{'='*60}")
        print(f"  Mode: {self.mode}  |  Architecture: {self.arch}")
        print(f"{'='*60}")

        loader = MIDataLoader(self.root)

        if self.mode == 'within_subject':
            return self._within_subject(loader)
        elif self.mode == 'cross_subject':
            return self._cross_subject(loader)
        else:
            raise ValueError(f"Unknown mode '{self.mode}'")

    # ── Within-subject: leave-one-session-out ──────────────────────────────

    def _within_subject(self, loader: MIDataLoader):
        subjects = self.subjects or list(range(1, N_SUBJECTS + 1))
        all_results = []

        for sub in subjects:
            print(f"\n── Subject {sub:02d} ──")
            n_ses = N_SESSIONS[sub]
            sub_accs, sub_bals = [], []

            for test_ses in range(1, n_ses + 1):
                train_ses = [s for s in range(1, n_ses + 1) if s != test_ses]
                X_tr, y_tr, _ = loader.load_all(subjects=[sub], sessions=train_ses)
                X_te, y_te, _ = loader.load_all(subjects=[sub], sessions=[test_ses])

                if len(X_te) == 0 or len(X_tr) == 0:
                    continue

                n_ch, n_t = X_tr.shape[1], X_tr.shape[2]
                model = build_model(self.arch, n_channels=n_ch, n_times=n_t,
                                     dropout=self.cfg['dropout']).to(self.device)

                if test_ses == 1:
                    print(f"  Model params: {count_parameters(model):,}")

                Xtr_t, ytr_t = to_tensors(X_tr, y_tr, self.device)
                Xte_t, yte_t = to_tensors(X_te, y_te, self.device)

                tr_loader = make_loader(Xtr_t, ytr_t, self.cfg['batch_size'],
                                         balance=True)
                te_loader = make_loader(Xte_t, yte_t, self.cfg['batch_size'],
                                         shuffle=False)

                fold_label = f"sub-{sub:02d} ses-{test_ses:02d}"
                _, _ = fit(model, tr_loader, te_loader, self.cfg,
                            self.device, fold_label)

                crit = nn.CrossEntropyLoss()
                _, acc, bal, preds, true = evaluate(model, te_loader, crit, self.device)
                print(f"\n  ✓ {fold_label} → acc={acc:.3f}  bal_acc={bal:.3f}")
                print(classification_report(true, preds,
                                             target_names=['rest','action'],
                                             digits=3))

                sub_accs.append(acc)
                sub_bals.append(bal)

                # Save model checkpoint
                ckpt_path = f"checkpoint_sub{sub:02d}_ses{test_ses:02d}.pt"
                torch.save({'model_state': model.state_dict(),
                             'n_channels': n_ch, 'n_times': n_t,
                             'arch': self.arch}, ckpt_path)

            if sub_accs:
                print(f"\n  Subject {sub:02d} mean acc={np.mean(sub_accs):.3f} ± "
                      f"{np.std(sub_accs):.3f}  |  "
                      f"bal_acc={np.mean(sub_bals):.3f} ± {np.std(sub_bals):.3f}")
                all_results.append({'subject': sub,
                                     'acc': np.mean(sub_accs),
                                     'bal_acc': np.mean(sub_bals)})

        print(f"\n{'='*60}")
        print("  Overall results (within-subject LOSO-CV):")
        accs = [r['acc']     for r in all_results]
        bals = [r['bal_acc'] for r in all_results]
        print(f"  Accuracy  : {np.mean(accs):.3f} ± {np.std(accs):.3f}")
        print(f"  Bal. Acc  : {np.mean(bals):.3f} ± {np.std(bals):.3f}")
        return all_results

    # ── Cross-subject: leave-one-subject-out ──────────────────────────────

    def _cross_subject(self, loader: MIDataLoader):
        subjects = self.subjects or list(range(1, N_SUBJECTS + 1))
        print("Loading all subject data…")
        X_all, y_all, meta = loader.load_all(subjects=subjects)
        sub_ids = meta['subject'].values

        results = []
        for test_sub in subjects:
            tr_mask = sub_ids != test_sub
            te_mask = sub_ids == test_sub
            X_tr, y_tr = X_all[tr_mask], y_all[tr_mask]
            X_te, y_te = X_all[te_mask], y_all[te_mask]

            if len(X_te) == 0:
                continue

            n_ch, n_t = X_tr.shape[1], X_tr.shape[2]
            model = build_model(self.arch, n_channels=n_ch, n_times=n_t,
                                 dropout=self.cfg['dropout']).to(self.device)

            Xtr_t, ytr_t = to_tensors(X_tr, y_tr, self.device)
            Xte_t, yte_t = to_tensors(X_te, y_te, self.device)

            tr_loader = make_loader(Xtr_t, ytr_t, self.cfg['batch_size'], balance=True)
            te_loader = make_loader(Xte_t, yte_t, self.cfg['batch_size'], shuffle=False)

            print(f"\n── Leave-out sub-{test_sub:02d} ──")
            fit(model, tr_loader, te_loader, self.cfg,
                self.device, f"sub-{test_sub:02d}")

            crit = nn.CrossEntropyLoss()
            _, acc, bal, preds, true = evaluate(model, te_loader, crit, self.device)
            print(f"  sub-{test_sub:02d} → acc={acc:.3f}  bal_acc={bal:.3f}")
            results.append({'subject': test_sub, 'acc': acc, 'bal_acc': bal})

        accs = [r['acc']     for r in results]
        bals = [r['bal_acc'] for r in results]
        print(f"\n  Overall LOSO-CV  acc={np.mean(accs):.3f}±{np.std(accs):.3f}  "
              f"bal_acc={np.mean(bals):.3f}±{np.std(bals):.3f}")
        return results


# ── CLI entry point ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train EEGNet for Motor Imagery")
    p.add_argument('--root',    required=True,
                    help='Path to BIDS dataset root, e.g. C:/Users/giaco/Box/EEGMI')
    p.add_argument('--mode',    default='within_subject',
                    choices=['within_subject', 'cross_subject'])
    p.add_argument('--arch',    default='eegnet',
                    choices=['eegnet', 'transformer'])
    p.add_argument('--epochs',  type=int,   default=150)
    p.add_argument('--lr',      type=float, default=1e-3)
    p.add_argument('--batch',   type=int,   default=32)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--subjects',type=int, nargs='+', default=None,
                    help='Subset of subject IDs (default: all)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg  = dict(epochs=args.epochs, lr=args.lr,
                 batch_size=args.batch, dropout=args.dropout)
    trainer = Trainer(root=args.root, mode=args.mode, arch=args.arch,
                       cfg=cfg, subjects=args.subjects)
    trainer.run()
