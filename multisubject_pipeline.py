"""Multi-Subject Pretraining and Calibration Fine-Tuning Pipeline for EEG BCI.

Includes:
- Euclidean Alignment (EA) spatial covariance whitening (He & Wu 2019).
- Robust per-window standardization.
- Multi-subject data loaders for Binary (EEGMI_Personal) and Three-Class (BCIFEST).
- Strict whole-cue leakage-free splitting (no window overlap across splits).
- Pretraining TemporalEEGNet backbones.
- Personal calibration fine-tuning routine with cue-disjoint early stopping.
- Cue-trial and window-level evaluation benchmarks.
"""
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import time

import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from temporal_networks import TemporalEEGNet
except ImportError:
    from .temporal_networks import TemporalEEGNet

try:
    from hierarchical_classifier import (
        UnsupervisedRestingAlignment,
        Stage1GatingNetwork,
        Stage2DirectionalNetwork,
        HierarchicalTwoStageEEGNet,
        HierarchicalEnsemble,
        compute_hierarchical_metrics,
        train_hierarchical_two_stage,
        finetune_hierarchical_personal,
    )
except ImportError:
    from .hierarchical_classifier import (
        UnsupervisedRestingAlignment,
        Stage1GatingNetwork,
        Stage2DirectionalNetwork,
        HierarchicalTwoStageEEGNet,
        HierarchicalEnsemble,
        compute_hierarchical_metrics,
        train_hierarchical_two_stage,
        finetune_hierarchical_personal,
    )


# Montage Definitions
COMMON_X24 = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'POz', 'O1', 'O2']
EEGMI_CUSTOM20 = ['F3', 'F1', 'Fz', 'F2', 'F4', 'C3', 'C1', 'Cz', 'C2', 'C4', 'CPz', 'P3', 'P1', 'Pz', 'P2', 'P4', 'POz', 'O1', 'Oz', 'O2']
X24_STOCK20 = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'POz', 'O2']

BINARY_CLASSES = ['rest', 'action']
THREE_CLASSES = ['rest', 'volar_grasp', 'finger_extension']


def canonical_label(text):
    return '_'.join(str(text).strip().casefold().replace('_', ' ').split())


# =====================================================================
# Euclidean Alignment / Spatial Covariance Whitening
# =====================================================================

class EuclideanAlignment:
    """Euclidean Alignment for EEG multi-subject domain adaptation.

    Normalizes subject spatial covariance distributions to the identity matrix:
        R_ref = 1/N sum_i Cov(X_i)
        W = R_ref^(-1/2)
        X_aligned = W @ X
    """

    def __init__(self, shrinkage=0.05):
        self.shrinkage = float(shrinkage)
        self.reference_cov_ = None
        self.alignment_matrix_ = None

    def fit(self, X):
        """Fit reference covariance and alignment matrix from training trials.

        Parameters:
            X: numpy array of shape (N, C, T) or list of (C, T) arrays.
        """
        if isinstance(X, list):
            X = np.stack(X)
        if X.ndim != 3:
            raise ValueError(f"Expected 3D array (N, C, T), got shape {X.shape}")
        
        n_trials, n_channels, n_times = X.shape
        covs = []
        for i in range(n_trials):
            trial = X[i]
            centered = trial - trial.mean(axis=1, keepdims=True)
            cov = (centered @ centered.T) / max(1, (n_times - 1))
            covs.append(cov)
        
        ref_cov = np.mean(covs, axis=0)
        # Apply shrinkage regularization for well-conditioned inverse square root
        trace_mean = np.trace(ref_cov) / n_channels
        reg_cov = (1.0 - self.shrinkage) * ref_cov + self.shrinkage * trace_mean * np.eye(n_channels)

        # SVD / Eigendecomposition: reg_cov = V @ diag(s) @ V.T
        vals, vecs = np.linalg.eigh(reg_cov)
        vals = np.maximum(vals, 1e-12)
        inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T

        self.reference_cov_ = reg_cov.astype(np.float32)
        self.alignment_matrix_ = inv_sqrt.astype(np.float32)
        return self

    def transform(self, X):
        """Apply alignment matrix to trials."""
        if self.alignment_matrix_ is None:
            raise RuntimeError("EuclideanAlignment has not been fitted.")
        if isinstance(X, list):
            X = np.stack(X)
        if X.ndim == 2:
            return (self.alignment_matrix_ @ X).astype(np.float32)
        elif X.ndim == 3:
            # (N, C, T) -> W @ X_i
            return np.einsum('ij,njk->nik', self.alignment_matrix_, X).astype(np.float32)
        else:
            raise ValueError(f"Invalid input shape {X.shape}")

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def state_dict(self):
        return {
            'shrinkage': self.shrinkage,
            'reference_cov': self.reference_cov_.tolist() if self.reference_cov_ is not None else None,
            'alignment_matrix': self.alignment_matrix_.tolist() if self.alignment_matrix_ is not None else None,
        }

    def load_state_dict(self, state):
        self.shrinkage = state['shrinkage']
        if state['reference_cov'] is not None:
            self.reference_cov_ = np.array(state['reference_cov'], dtype=np.float32)
        if state['alignment_matrix'] is not None:
            self.alignment_matrix_ = np.array(state['alignment_matrix'], dtype=np.float32)
        return self


# =====================================================================
# Whole-Cue Non-Crossing Window Extraction & Splitting
# =====================================================================

def extract_cue_windows(filtered_data, events_df, fs=256, window_s=2.0, stride_s=0.5,
                         settle_s=0.5, class_map=None, artifact_limit_v=100e-6,
                         min_signal_std_v=1e-9):
    """Extract sliding windows strictly contained inside individual cue boundaries.

    Guarantees no window crosses across cue boundaries, settling period is excluded,
    and corrupted windows (peak > 100uV or std < 1e-9) are dropped.
    """
    n_samples = round(window_s * fs)
    stride_samples = round(stride_s * fs)
    settle_samples = round(settle_s * fs)
    n_channels, total_samples = filtered_data.shape
    bad_mask = (np.abs(filtered_data) > artifact_limit_v).any(axis=0)

    windows = []
    labels = []
    cue_indices = []
    records = []

    for cue_idx, row in events_df.iterrows():
        raw_type = row['trial_type']
        canon = canonical_label(raw_type)
        if class_map is not None and canon not in class_map:
            continue
        label = class_map[canon] if class_map is not None else canon

        onset_sec = float(row['onset'])
        duration_sec = float(row['duration'])
        start_samp = round(onset_sec * fs)
        end_samp = min(total_samples, round((onset_sec + duration_sec) * fs))

        eval_start = start_samp + settle_samples
        if eval_start >= end_samp or (end_samp - eval_start) < n_samples:
            continue

        for a in range(eval_start, end_samp - n_samples + 1, stride_samples):
            b = a + n_samples
            win = filtered_data[:, a:b]
            if bad_mask[a:b].any():
                continue
            if np.any(win.std(axis=1) < min_signal_std_v):
                continue
            windows.append(win.astype(np.float32))
            labels.append(label)
            cue_indices.append(cue_idx)
            records.append({
                'start_sample': a,
                'end_sample': b,
                'label': label,
                'label_name': canon,
                'cue_idx': cue_idx,
                'onset_s': onset_sec,
                'duration_s': duration_sec,
            })

    if not windows:
        return np.empty((0, n_channels, n_samples), dtype=np.float32), np.empty(0, dtype=int), [], []
    return np.stack(windows), np.array(labels, dtype=int), cue_indices, records


def split_whole_cues_stratified(records, y, cue_ids, test_size=0.25, random_state=42):
    """Split whole cue trials into train and tuning/test partitions.

    Guarantees zero window leakage: every cue trial is either fully in train
    or fully in test/tuning, stratified by class.
    """
    rng = np.random.RandomState(random_state)
    cue_to_class = {}
    cue_to_window_indices = defaultdict(list)

    for win_idx, (r, c_idx, label) in enumerate(zip(records, cue_ids, y)):
        cue_key = (r.get('run', 'run0'), c_idx)
        cue_to_class[cue_key] = label
        cue_to_window_indices[cue_key].append(win_idx)

    cues_by_class = defaultdict(list)
    for cue_key, cls in cue_to_class.items():
        cues_by_class[cls].append(cue_key)

    train_cues = set()
    val_cues = set()

    for cls, cue_list in cues_by_class.items():
        shuffled = list(cue_list)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * test_size)))
        val_subset = set(shuffled[:n_val])
        train_subset = set(shuffled[n_val:])
        if not train_subset:  # Ensure training has at least one cue
            train_subset = {shuffled[0]}
            val_subset = set(shuffled[1:])
        train_cues.update(train_subset)
        val_cues.update(val_subset)

    train_idx = [i for cue_key in train_cues for i in cue_to_window_indices[cue_key]]
    val_idx = [i for cue_key in val_cues for i in cue_to_window_indices[cue_key]]

    # Verify strictly disjoint
    overlap = set(train_cues) & set(val_cues)
    assert not overlap, f"Cue overlap detected across partitions: {overlap}"

    return np.array(sorted(train_idx), dtype=int), np.array(sorted(val_idx), dtype=int)


# =====================================================================
# Dataset Loaders (EEGMI_Personal & BCIFEST)
# =====================================================================

def load_eegmi_binary_subjects(root_dir, subjects, channels=EEGMI_CUSTOM20,
                               window_s=2.0, stride_s=0.5, settle_s=0.5, fs=256):
    """Load binary (Rest vs. Action) data from EEGMI_Personal.

    Returns dict mapping subject_id -> dict(X, y, cue_ids, records, alignment).
    """
    root = Path(root_dir)
    sos = butter(4, [8.0, 30.0], fs=fs, btype='bandpass', output='sos')
    class_map = {'rest': 0, 'action': 1}
    dataset = {}

    for sub_num in subjects:
        sub_str = f"sub-{sub_num:02d}"
        sub_dir = root / sub_str
        if not sub_dir.exists():
            continue
        edf_files = sorted(sub_dir.glob("ses-*/eeg/*_eeg.edf"))
        all_windows = []
        all_labels = []
        all_cues = []
        all_records = []
        cue_offset = 0
        print(f"  [load_binary] Scanning {sub_str} ({len(edf_files)} runs)...", flush=True)

        for edf_path in edf_files:
            run_id = edf_path.stem.replace('_eeg', '')
            tsv_path = edf_path.with_name(run_id + '_events.tsv')
            if not tsv_path.exists():
                continue
            events = pd.read_csv(tsv_path, sep='\t')
            if 'trial_type' not in events.columns:
                continue

            try:
                raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
                if not np.isclose(raw.info['sfreq'], fs):
                    continue
                if not set(channels).issubset(set(raw.ch_names)):
                    continue
                data = raw.get_data(picks=channels).astype(np.float32)
                filtered = sosfilt(sos, data, axis=1).astype(np.float32)
            except Exception as e:
                print(f"    [load_binary] Skipping {edf_path.name} due to read error: {e}")
                continue

            wins, lbls, cues, recs = extract_cue_windows(
                filtered, events, fs=fs, window_s=window_s, stride_s=stride_s,
                settle_s=settle_s, class_map=class_map
            )
            for r in recs:
                r['subject'] = sub_str
                r['run'] = run_id
                r['global_cue'] = cue_offset + r['cue_idx']

            if len(wins) > 0:
                all_windows.append(wins)
                all_labels.append(lbls)
                all_cues.extend([cue_offset + c for c in cues])
                all_records.extend(recs)
                cue_offset += len(events)

        if all_windows:
            X_sub = np.concatenate(all_windows, axis=0)
            y_sub = np.concatenate(all_labels, axis=0)
            dataset[sub_str] = {
                'X': X_sub,
                'y': y_sub,
                'cue_ids': all_cues,
                'records': all_records,
                'n_trials': len(y_sub),
                'n_cues': len(set(all_cues)),
            }
    return dataset


def load_bcifest_multiclass_subjects(root_dir, subjects, channels=COMMON_X24,
                                      classes=THREE_CLASSES, window_s=2.0,
                                      stride_s=0.5, settle_s=0.5, fs=256):
    """Load multiclass (Rest vs Volar Grasp vs Finger Extension) data from BCIFEST/bids.

    Returns dict mapping subject_id -> dict(X, y, cue_ids, records).
    """
    root = Path(root_dir)
    sos = butter(4, [8.0, 30.0], fs=fs, btype='bandpass', output='sos')
    class_map = {canonical_label(c): i for i, c in enumerate(classes)}
    dataset = {}

    for sub_num in subjects:
        sub_str = f"sub-{sub_num:02d}"
        sub_dir = root / sub_str
        if not sub_dir.exists():
            continue
        edf_files = sorted(sub_dir.glob("ses-*/eeg/*task-olrandmcmitrain*_eeg.edf"))
        all_windows = []
        all_labels = []
        all_cues = []
        all_records = []
        cue_offset = 0
        print(f"  [load_threeclass] Scanning {sub_str} ({len(edf_files)} runs)...", flush=True)

        for edf_path in edf_files:
            run_id = edf_path.stem.replace('_eeg', '')
            tsv_path = edf_path.with_name(run_id + '_events.tsv')

            try:
                raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
                if not np.isclose(raw.info['sfreq'], fs):
                    continue
                if not set(channels).issubset(set(raw.ch_names)):
                    continue
            except Exception as e:
                continue

            # Load events or recover from raw.annotations
            events = None
            if tsv_path.exists():
                try:
                    events = pd.read_csv(tsv_path, sep='\t')
                    if events.empty or 'trial_type' not in events:
                        events = None
                except Exception:
                    events = None

            if events is None or events.empty:
                annots = raw.annotations
                if annots and len(annots) > 0:
                    events = pd.DataFrame({
                        'onset': annots.onset,
                        'duration': annots.duration,
                        'trial_type': annots.description
                    })
                else:
                    continue

            data = raw.get_data(picks=channels).astype(np.float32)
            filtered = sosfilt(sos, data, axis=1).astype(np.float32)

            wins, lbls, cues, recs = extract_cue_windows(
                filtered, events, fs=fs, window_s=window_s, stride_s=stride_s,
                settle_s=settle_s, class_map=class_map
            )
            for r in recs:
                r['subject'] = sub_str
                r['run'] = run_id
                r['global_cue'] = cue_offset + r['cue_idx']

            if len(wins) > 0:
                all_windows.append(wins)
                all_labels.append(lbls)
                all_cues.extend([cue_offset + c for c in cues])
                all_records.extend(recs)
                cue_offset += len(events)

        if all_windows:
            X_sub = np.concatenate(all_windows, axis=0)
            y_sub = np.concatenate(all_labels, axis=0)
            dataset[sub_str] = {
                'X': X_sub,
                'y': y_sub,
                'cue_ids': all_cues,
                'records': all_records,
                'n_trials': len(y_sub),
                'n_cues': len(set(all_cues)),
            }
    return dataset


# =====================================================================
# Multi-Subject Alignment & Pipeline Assembly
# =====================================================================

def align_multisubject_data(subject_data_dict):
    """Fit EuclideanAlignment per subject and align all trials.

    Returns:
        X_all: (N, C, T) concatenated aligned windows
        y_all: (N,) labels
        groups_all: (N,) subject strings
        cues_all: (N,) global cue IDs
        records_all: list of record dicts
        alignments: dict of subject_id -> EuclideanAlignment instance
    """
    X_parts = []
    y_parts = []
    groups_parts = []
    cues_parts = []
    records_parts = []
    alignments = {}

    for sub_id, data in subject_data_dict.items():
        ea = EuclideanAlignment(shrinkage=0.05).fit(data['X'])
        X_aligned = ea.transform(data['X'])
        alignments[sub_id] = ea

        X_parts.append(X_aligned)
        y_parts.append(data['y'])
        groups_parts.append(np.full(len(data['y']), sub_id))
        cues_parts.append(data['cue_ids'])
        records_parts.extend(data['records'])

    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    groups_all = np.concatenate(groups_parts, axis=0)
    cues_all = np.concatenate(cues_parts, axis=0)

    return X_all, y_all, groups_all, cues_all, records_parts, alignments


# =====================================================================
# Model Training, Fine-Tuning & Evaluation
# =====================================================================

def evaluate_predictions(y_true, y_pred_prob, cue_ids=None, class_names=None):
    """Compute comprehensive window and cue-level metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred_prob = np.asarray(y_pred_prob, dtype=np.float32)
    n_classes = y_pred_prob.shape[1] if y_pred_prob.ndim > 1 else 2
    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]

    # Window-level
    win_pred = np.argmax(y_pred_prob, axis=1) if y_pred_prob.ndim > 1 else (y_pred_prob >= 0.5).astype(int)
    win_ba = balanced_accuracy_score(y_true, win_pred)
    win_cm = confusion_matrix(y_true, win_pred, labels=list(range(n_classes))).tolist()
    win_f1 = f1_score(y_true, win_pred, average='macro')

    win_recalls = {}
    for i, name in enumerate(class_names):
        mask = (y_true == i)
        win_recalls[name] = float(np.mean(win_pred[mask] == i)) if np.any(mask) else 0.0

    metrics = {
        'window': {
            'balanced_accuracy': float(win_ba),
            'macro_f1': float(win_f1),
            'confusion_matrix': win_cm,
            'class_recalls': win_recalls,
            'n_windows': len(y_true),
        }
    }

    # Cue-level (average window probabilities per cue)
    if cue_ids is not None and len(cue_ids) == len(y_true):
        cues_dict = defaultdict(list)
        cue_labels = {}
        for idx, (c_id, p, y) in enumerate(zip(cue_ids, y_pred_prob, y_true)):
            cues_dict[c_id].append(p)
            cue_labels[c_id] = y

        cue_trues = []
        cue_preds = []
        for c_id, probs in cues_dict.items():
            mean_prob = np.mean(probs, axis=0)
            cue_pred = np.argmax(mean_prob) if mean_prob.ndim > 0 else int(mean_prob >= 0.5)
            cue_trues.append(cue_labels[c_id])
            cue_preds.append(cue_pred)

        cue_trues = np.array(cue_trues)
        cue_preds = np.array(cue_preds)
        cue_ba = balanced_accuracy_score(cue_trues, cue_preds)
        cue_cm = confusion_matrix(cue_trues, cue_preds, labels=list(range(n_classes))).tolist()
        cue_f1 = f1_score(cue_trues, cue_preds, average='macro')

        cue_recalls = {}
        for i, name in enumerate(class_names):
            mask = (cue_trues == i)
            cue_recalls[name] = float(np.mean(cue_preds[mask] == i)) if np.any(mask) else 0.0

        metrics['cue'] = {
            'balanced_accuracy': float(cue_ba),
            'macro_f1': float(cue_f1),
            'confusion_matrix': cue_cm,
            'class_recalls': cue_recalls,
            'n_cues': len(cue_trues),
        }
    return metrics


def train_temporal_backbone(model, train_loader, val_loader, epochs=40,
                            lr=1e-3, weight_decay=1e-2, patience=10,
                            class_weights=None, device='cpu'):
    """Pretrain TemporalEEGNet backbone across subjects with early stopping."""
    model = model.to(device)
    if class_weights is not None:
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    else:
        loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_ba = -1.0
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = loss_fn(logits, by)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        train_loss = float(np.mean(train_losses))

        # Evaluate on validation
        model.eval()
        val_losses = []
        all_val_preds = []
        all_val_trues = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                loss = loss_fn(logits, by)
                val_losses.append(loss.item())
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_val_preds.append(probs)
                all_val_trues.append(by.cpu().numpy())

        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        if all_val_preds:
            cat_preds = np.concatenate(all_val_preds, axis=0)
            cat_trues = np.concatenate(all_val_trues, axis=0)
            val_ba = float(balanced_accuracy_score(cat_trues, np.argmax(cat_preds, axis=1)))
        else:
            val_ba = 0.0

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_balanced_accuracy': val_ba,
        })
        print(f"  [Epoch {epoch:02d}/{epochs:02d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_ba={val_ba:.4f} (best={max(best_val_ba, val_ba):.4f})", flush=True)

        if val_ba > best_val_ba:
            best_val_ba = val_ba
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [Early Stopping] No improvement for {patience} epochs. Stopping at epoch {epoch}.", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_ba, history


def finetune_temporal_personal(pretrained_model, calib_train_x, calib_train_y,
                               calib_train_cues, calib_val_x, calib_val_y,
                               calib_val_cues, epochs=30, lr_head=1e-3,
                               lr_backbone=1e-4, patience=8,
                               freeze_backbone_epochs=3, batch_size=32,
                               device='cpu'):
    """Fine-tune personal calibration model on whole-cue held-out tuning split.

    1. Fits personal EuclideanAlignment on calibration training trials ONLY.
    2. Aligns calib_train and calib_val.
    3. Fine-tunes classification head (first freeze_backbone_epochs), then full model.
    4. Early stops based on calib_val balanced accuracy to prevent overfitting.
    """
    # 1. Fit personal covariance alignment strictly on calib_train
    ea = EuclideanAlignment(shrinkage=0.05).fit(calib_train_x)
    aligned_train_x = ea.transform(calib_train_x)
    aligned_val_x = ea.transform(calib_val_x)

    train_ds = TensorDataset(torch.from_numpy(aligned_train_x), torch.from_numpy(calib_train_y).long())
    val_ds = TensorDataset(torch.from_numpy(aligned_val_x), torch.from_numpy(calib_val_y).long())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = copy.deepcopy(pretrained_model).to(device)

    # Class balance weighting
    class_counts = Counter(calib_train_y)
    n_classes = len(class_counts)
    total = len(calib_train_y)
    weights = [total / (n_classes * max(1, class_counts[c])) for c in range(n_classes)]
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    # Parameter groups: head vs backbone
    head_params = list(model.head.parameters())
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith('head')]

    optimizer = torch.optim.AdamW([
        {'params': head_params, 'lr': lr_head},
        {'params': backbone_params, 'lr': lr_backbone if freeze_backbone_epochs == 0 else 0.0}
    ], weight_decay=1e-2)

    best_val_ba = -1.0
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        if epoch == freeze_backbone_epochs + 1 and freeze_backbone_epochs > 0:
            # Unfreeze backbone
            optimizer.param_groups[1]['lr'] = lr_backbone

        model.train()
        train_losses = []
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = loss_fn(logits, by)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Evaluate on validation cues
        model.eval()
        val_losses = []
        all_val_probs = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                loss = loss_fn(logits, by)
                val_losses.append(loss.item())
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_val_probs.append(probs)

        cat_probs = np.concatenate(all_val_probs, axis=0) if all_val_probs else np.empty((0, n_classes))
        val_eval = evaluate_predictions(calib_val_y, cat_probs, calib_val_cues)
        val_cue_ba = val_eval.get('cue', {}).get('balanced_accuracy', val_eval['window']['balanced_accuracy'])
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0

        history.append({
            'epoch': epoch,
            'train_loss': float(np.mean(train_losses)),
            'val_loss': val_loss,
            'val_cue_balanced_accuracy': float(val_cue_ba),
            'val_window_balanced_accuracy': float(val_eval['window']['balanced_accuracy']),
        })

        if val_cue_ba > best_val_ba:
            best_val_ba = val_cue_ba
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, ea, best_val_ba, history


def evaluate_test_session(model, test_x, test_y, test_cues, alignment, class_names=None, device='cpu'):
    """Evaluate trained/fine-tuned model on held-out test data with personal alignment."""
    aligned_test_x = alignment.transform(test_x)
    test_ds = TensorDataset(torch.from_numpy(aligned_test_x), torch.from_numpy(test_y).long())
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = model.to(device)
    model.eval()
    all_probs = []
    with torch.no_grad():
        for bx, _ in test_loader:
            bx = bx.to(device)
            logits = model(bx)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

    cat_probs = np.concatenate(all_probs, axis=0)
    metrics = evaluate_predictions(test_y, cat_probs, test_cues, class_names=class_names)
    return metrics, cat_probs


# =====================================================================
# Multi-Band Filtering & Dual-Band MRCP + SMR Architecture
# =====================================================================

def filter_multiband_eeg(data, fs=256, mrcp_band=(0.3, 3.0), smr_band=(4.0, 40.0)):
    """Filter continuous raw EEG into dual-band streams: MRCP (0.3-3 Hz) and SMR (4-40 Hz).
    
    data: (n_channels, n_samples)
    returns: (2 * n_channels, n_samples) where channels 0..C-1 are MRCP, and C..2C-1 are SMR.
    """
    sos_mrcp = butter(4, list(mrcp_band), fs=fs, btype='bandpass', output='sos')
    sos_smr = butter(4, list(smr_band), fs=fs, btype='bandpass', output='sos')
    mrcp = sosfilt(sos_mrcp, data, axis=1).astype(np.float32)
    smr = sosfilt(sos_smr, data, axis=1).astype(np.float32)
    return np.concatenate([mrcp, smr], axis=0)


def extract_cue_windows_multiband(raw_data, events_df, fs=256, window_s=2.0, stride_s=0.5,
                                  settle_s=0.5, class_map=None, mrcp_band=(0.3, 3.0),
                                  smr_band=(4.0, 40.0), artifact_limit_v=100e-6,
                                  min_signal_std_v=1e-9):
    """Extract synchronized multi-band sliding windows inside individual cue boundaries.
    
    Returns:
        windows_multiband: (N, 2 * C, T)
        windows_smr: (N, C, T)
        labels: (N,)
        cues: (N,)
        records: list of dicts
    """
    sos_mrcp = butter(4, list(mrcp_band), fs=fs, btype='bandpass', output='sos')
    sos_smr = butter(4, list(smr_band), fs=fs, btype='bandpass', output='sos')
    mrcp_data = sosfilt(sos_mrcp, raw_data, axis=1).astype(np.float32)
    smr_data = sosfilt(sos_smr, raw_data, axis=1).astype(np.float32)

    n_channels = raw_data.shape[0]
    win_len = round(window_s * fs)
    stride_len = round(stride_s * fs)
    settle_len = round(settle_s * fs)

    windows_mb = []
    windows_smr = []
    labels = []
    cue_ids = []
    records = []

    for cue_idx, row in events_df.iterrows():
        raw_label = row.get('trial_type', '')
        if class_map is not None:
            norm_label = canonical_label(raw_label)
            if norm_label not in class_map:
                continue
            class_id = class_map[norm_label]
        else:
            class_id = raw_label

        cue_start = round(float(row['onset']) * fs)
        cue_duration = round(float(row['duration']) * fs)
        cue_end = cue_start + cue_duration

        analysis_start = cue_start + settle_len
        analysis_end = cue_end

        if analysis_end - analysis_start < win_len:
            continue

        pos = analysis_start
        window_in_cue = 0
        while pos + win_len <= analysis_end:
            w_mrcp = mrcp_data[:, pos:pos + win_len]
            w_smr = smr_data[:, pos:pos + win_len]

            # Reject artifacts on SMR / broadband
            if np.any(np.abs(w_smr) > artifact_limit_v):
                pos += stride_len
                window_in_cue += 1
                continue
            if np.any(np.std(w_smr, axis=1) < min_signal_std_v):
                pos += stride_len
                window_in_cue += 1
                continue

            w_mb = np.concatenate([w_mrcp, w_smr], axis=0) # (2*C, T)
            windows_mb.append(w_mb)
            windows_smr.append(w_smr)
            labels.append(class_id)
            cue_ids.append(cue_idx)
            records.append({
                'cue_idx': cue_idx,
                'window_in_cue': window_in_cue,
                'class_id': class_id,
                'sample_start': pos,
                'sample_end': pos + win_len,
            })
            pos += stride_len
            window_in_cue += 1

    if not windows_mb:
        return (np.empty((0, 2 * n_channels, win_len), dtype=np.float32),
                np.empty((0, n_channels, win_len), dtype=np.float32),
                np.empty(0, dtype=int), np.empty(0, dtype=int), [])

    return (np.stack(windows_mb).astype(np.float32),
            np.stack(windows_smr).astype(np.float32),
            np.array(labels, dtype=int),
            np.array(cue_ids, dtype=int),
            records)


class DualBandTemporalEEGNet(nn.Module):
    """Hierarchical / Dual-Band EEGNet decoupling Slow MRCP (0.3-3 Hz) and SMR (4-40 Hz).
    
    Inputs:
        x: tensor of shape (batch, 2 * n_channels, n_times)
           Channels 0 .. n_channels-1: Slow MRCP band (0.3-3.0 Hz)
           Channels n_channels .. 2*n_channels-1: SMR band (4.0-40.0 Hz)
    Outputs:
        logits: tensor of shape (batch, n_classes)
    """
    def __init__(self, n_channels, n_classes=2, window_standardization=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.window_standardization = window_standardization

        # Branch 1: Slow MRCP stream (linear potential shape + slow energy)
        # Wider temporal kernels: 65 (~0.25s), 129 (~0.5s), 257 (~1.0s at 256 Hz)
        self.mrcp_temporal = nn.ModuleList([
            nn.Conv2d(1, 4, (1, width), padding=(0, width // 2), bias=False)
            for width in (65, 129, 257)
        ])
        self.mrcp_spatial = nn.Sequential(
            nn.BatchNorm2d(12),
            nn.Conv2d(12, 16, (n_channels, 1), groups=4, bias=False),
            nn.BatchNorm2d(16),
        )
        self.mrcp_pool = nn.AvgPool1d(16)
        self.mrcp_context = nn.Sequential(
            nn.Conv1d(16, 16, 5, padding=2, bias=False),
            nn.BatchNorm1d(16), nn.ELU(),
            nn.Conv1d(16, 16, 5, dilation=2, padding=4, bias=False),
            nn.BatchNorm1d(16), nn.ELU(),
        )

        # Branch 2: SMR stream (ERD/ERS band-power dynamics)
        # Fast temporal kernels: 17 (~0.06s), 33 (~0.13s), 65 (~0.25s)
        self.smr_temporal = nn.ModuleList([
            nn.Conv2d(1, 4, (1, width), padding=(0, width // 2), bias=False)
            for width in (17, 33, 65)
        ])
        self.smr_spatial = nn.Sequential(
            nn.BatchNorm2d(12),
            nn.Conv2d(12, 16, (n_channels, 1), groups=4, bias=False),
            nn.BatchNorm2d(16),
        )
        self.smr_power = nn.AvgPool1d(16)
        self.smr_context = nn.Sequential(
            nn.Conv1d(16, 16, 5, padding=2, bias=False),
            nn.BatchNorm1d(16), nn.ELU(),
            nn.Conv1d(16, 16, 5, dilation=2, padding=4, bias=False),
            nn.BatchNorm1d(16), nn.ELU(),
        )

        # Fused classifier head: Adaptive pool to 2 points per channel -> (16 + 16) * 2 = 64 features
        self.pool = nn.AdaptiveAvgPool1d(2)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        if self.window_standardization:
            x = x - x.mean(dim=1, keepdim=True)
            channel_rms = x.square().mean(dim=2).sqrt()
            scale = channel_rms.median(dim=1).values.clamp_min(1e-6)
            x = x / scale[:, None, None]

        x_mrcp = x[:, :self.n_channels, :].unsqueeze(1)
        x_smr = x[:, self.n_channels:, :].unsqueeze(1)

        # MRCP branch: capture both directional amplitude deflection and slow energy
        mrcp_filt = self.mrcp_spatial(torch.cat([b(x_mrcp) for b in self.mrcp_temporal], dim=1)).squeeze(2)
        mrcp_amp = self.mrcp_pool(mrcp_filt)
        mrcp_power = torch.log1p(self.mrcp_pool(mrcp_filt.square()))
        mrcp_feat = mrcp_amp + mrcp_power
        mrcp_out = mrcp_feat + self.mrcp_context(mrcp_feat)

        # SMR branch: standard log power
        smr_filt = self.smr_spatial(torch.cat([b(x_smr) for b in self.smr_temporal], dim=1)).squeeze(2)
        smr_power = torch.log1p(self.smr_power(smr_filt.square()))
        smr_out = smr_power + self.smr_context(smr_power)

        # Fuse representations
        pooled = torch.cat([self.pool(mrcp_out), self.pool(smr_out)], dim=1)
        return self.head(pooled)


class HierarchicalTwoStageDecoder:
    """Decoupled Two-Stage Hierarchical BCI Decoder:
    Stage 1: Rest vs. Intentional Movement (SMR band 4-40 Hz / 8-30 Hz)
    Stage 2: Volar Grasp vs. Finger Extension (Multi-band MRCP 0.3-3 Hz + SMR 4-40 Hz)
    """
    def __init__(self, stage1_model, stage2_model,
                 rest_threshold=0.50, min_action_margin=0.0,
                 classes=('rest', 'volar_grasp', 'finger_extension'),
                 device='cpu'):
        self.stage1_model = stage1_model.to(device).eval()
        self.stage2_model = stage2_model.to(device).eval()
        self.rest_threshold = rest_threshold
        self.min_action_margin = min_action_margin
        self.classes = list(classes)
        self.device = device

    def predict_window(self, x_smr, x_multiband):
        """Classify window or batch of windows."""
        if isinstance(x_smr, np.ndarray):
            x_smr = torch.from_numpy(x_smr).float()
        if isinstance(x_multiband, np.ndarray):
            x_multiband = torch.from_numpy(x_multiband).float()
        if x_smr.ndim == 2:
            x_smr = x_smr.unsqueeze(0)
            x_multiband = x_multiband.unsqueeze(0)

        with torch.no_grad():
            # Stage 1: Rest (class 0) vs Action (class 1)
            logits1 = self.stage1_model(x_smr.to(self.device))
            probs1 = torch.softmax(logits1, dim=1).cpu().numpy()
            p_rest = probs1[:, 0]
            p_action = probs1[:, 1]

            # Stage 2: Volar Grasp (class 0) vs Finger Extension (class 1)
            logits2 = self.stage2_model(x_multiband.to(self.device))
            probs2 = torch.softmax(logits2, dim=1).cpu().numpy()
            p_grasp_given_action = probs2[:, 0]
            p_ext_given_action = probs2[:, 1]

        b_size = len(p_rest)
        combined_probs = np.zeros((b_size, 3), dtype=np.float32)
        combined_probs[:, 0] = p_rest
        combined_probs[:, 1] = p_action * p_grasp_given_action
        combined_probs[:, 2] = p_action * p_ext_given_action

        # Row-normalize
        row_sums = combined_probs.sum(axis=1, keepdims=True)
        combined_probs = combined_probs / np.maximum(row_sums, 1e-12)

        decisions = []
        for i in range(b_size):
            if p_rest[i] >= self.rest_threshold:
                decisions.append(0)  # Rest
            else:
                if abs(p_grasp_given_action[i] - p_ext_given_action[i]) < self.min_action_margin:
                    decisions.append(-1)  # Abstain
                elif p_grasp_given_action[i] >= p_ext_given_action[i]:
                    decisions.append(1)  # Volar Grasp
                else:
                    decisions.append(2)  # Finger Extension

        return np.array(decisions, dtype=int), combined_probs


def load_bcifest_multiband_subjects(root_dir, subjects, channels=COMMON_X24,
                                    classes=THREE_CLASSES, window_s=2.0,
                                    stride_s=0.5, settle_s=0.5, fs=256):
    """Load multiclass BCIFEST subjects with synchronized Multi-Band (MRCP + SMR).
    
    Returns dict mapping subject_id -> dict(X_mb, X_smr, y, cue_ids, records, rest_windows).
    """
    root = Path(root_dir)
    class_map = {canonical_label(c): i for i, c in enumerate(classes)}
    dataset = {}

    for sub_num in subjects:
        sub_str = f"sub-{sub_num:02d}"
        sub_dir = root / sub_str
        if not sub_dir.exists():
            continue
        edf_files = sorted(sub_dir.glob("ses-*/eeg/*task-olrandmcmitrain*_eeg.edf"))
        all_mb = []
        all_smr = []
        all_labels = []
        all_cues = []
        all_records = []
        rest_windows = []
        cue_offset = 0
        print(f"  [load_multiband] Scanning {sub_str} ({len(edf_files)} runs)...", flush=True)

        for edf_path in edf_files:
            run_id = edf_path.stem.replace('_eeg', '')
            tsv_path = edf_path.with_name(run_id + '_events.tsv')

            try:
                raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
                if not np.isclose(raw.info['sfreq'], fs):
                    continue
                if not set(channels).issubset(set(raw.ch_names)):
                    continue
            except Exception as e:
                continue

            events = None
            if tsv_path.exists():
                try:
                    events = pd.read_csv(tsv_path, sep='\t')
                    if events.empty or 'trial_type' not in events:
                        events = None
                except Exception:
                    events = None

            if events is None or events.empty:
                annots = raw.annotations
                if annots and len(annots) > 0:
                    events = pd.DataFrame({
                        'onset': annots.onset,
                        'duration': annots.duration,
                        'trial_type': annots.description
                    })
                else:
                    continue

            raw_data = raw.get_data(picks=channels).astype(np.float32)

            w_mb, w_smr, lbls, cues, recs = extract_cue_windows_multiband(
                raw_data, events, fs=fs, window_s=window_s, stride_s=stride_s,
                settle_s=settle_s, class_map=class_map
            )

            for r in recs:
                r['subject'] = sub_str
                r['run'] = run_id
                r['global_cue'] = cue_offset + r['cue_idx']

            if len(w_mb) > 0:
                all_mb.append(w_mb)
                all_smr.append(w_smr)
                all_labels.append(lbls)
                all_cues.extend([cue_offset + c for c in cues])
                all_records.extend(recs)

                # Collect pure rest windows for unsupervised resting alignment
                rest_mask = (lbls == 0)
                if np.any(rest_mask):
                    rest_windows.append(w_mb[rest_mask])

                cue_offset += len(events)

        if all_mb:
            dataset[sub_str] = {
                'X_mb': np.concatenate(all_mb, axis=0),
                'X_smr': np.concatenate(all_smr, axis=0),
                'y': np.concatenate(all_labels, axis=0),
                'cue_ids': all_cues,
                'records': all_records,
                'rest_windows': np.concatenate(rest_windows, axis=0) if rest_windows else np.empty((0, 2*len(channels), round(window_s*fs))),
                'n_trials': len(all_labels[0]) if len(all_labels)==1 else sum(len(l) for l in all_labels),
                'n_cues': len(set(all_cues)),
            }

    return dataset


def align_multisubject_resting_data(subject_data_dict):
    """Fit UnsupervisedRestingAlignment per subject and align multi-band and SMR trials.

    Returns:
        X_mb_all: (N, 2*C, T) concatenated aligned multi-band windows
        X_smr_all: (N, C, T) concatenated aligned SMR windows
        y_all: (N,) labels
        groups_all: (N,) subject strings
        cues_all: (N,) global cue IDs
        records_all: list of record dicts
        alignments: dict of subject_id -> UnsupervisedRestingAlignment instance
    """
    X_mb_parts = []
    X_smr_parts = []
    y_parts = []
    groups_parts = []
    cues_parts = []
    records_parts = []
    alignments = {}

    for sub_id, data in subject_data_dict.items():
        n_ch = data['X_smr'].shape[1]
        rest_win = data.get('rest_windows')
        if rest_win is not None and len(rest_win) > 0:
            # Fit C x C URA on C-channel SMR resting windows so both (N, C, T) and (N, 2*C, T) transform cleanly without cross-band mixing
            rest_c = rest_win[:, n_ch:, :] if rest_win.shape[1] == 2 * n_ch else rest_win
            ura = UnsupervisedRestingAlignment(shrinkage=0.05).fit(rest_c)
        else:
            ura = UnsupervisedRestingAlignment(shrinkage=0.05).fit_from_dataset(data['X_smr'], data['y'], rest_label=0)

        X_mb_aligned = ura.transform(data['X_mb'])
        X_smr_aligned = ura.transform(data['X_smr'])
        alignments[sub_id] = ura

        X_mb_parts.append(X_mb_aligned)
        X_smr_parts.append(X_smr_aligned)
        y_parts.append(data['y'])
        groups_parts.append(np.full(len(data['y']), sub_id))
        cues_parts.append(data['cue_ids'])
        records_parts.extend(data['records'])

    X_mb_all = np.concatenate(X_mb_parts, axis=0)
    X_smr_all = np.concatenate(X_smr_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    groups_all = np.concatenate(groups_parts, axis=0)
    cues_all = np.concatenate(cues_parts, axis=0)

    return X_mb_all, X_smr_all, y_all, groups_all, cues_all, records_parts, alignments
