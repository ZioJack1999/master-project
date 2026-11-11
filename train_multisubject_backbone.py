"""Train multi-subject TemporalEEGNet backbones for Binary and Three-Class BCI tasks.

Applies:
- Euclidean Alignment (spatial covariance whitening) per subject.
- Robust window standardization inside TemporalEEGNet.
- Whole-cue evaluation monitoring and early stopping.
- Complete contract serialization with SHA256 and training diagnostics.
"""
import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from multisubject_pipeline import (
    EuclideanAlignment,
    extract_cue_windows,
    split_whole_cues_stratified,
    load_eegmi_binary_subjects,
    load_bcifest_multiclass_subjects,
    align_multisubject_data,
    train_temporal_backbone,
    evaluate_predictions,
    EEGMI_CUSTOM20,
    COMMON_X24,
    BINARY_CLASSES,
    THREE_CLASSES,
)
from temporal_networks import TemporalEEGNet


def train_binary_backbone(data_root=r"C:\Users\giaco\Box\EEGMI_Personal",
                          train_subjects=(1, 3, 4, 6, 7, 9, 10),
                          val_subjects=(4, 9),
                          channels=EEGMI_CUSTOM20,
                          epochs=18,
                          batch_size=64,
                          output_path=Path("decoder_and_models/backbone_binary_temporaleegnet.pt"),
                          device='cpu'):
    """Pretrain binary backbone across multiple subjects."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[Binary Backbone] Loading subjects {train_subjects} from {data_root}...")
    dataset = load_eegmi_binary_subjects(data_root, train_subjects, channels=channels)
    print(f"[Binary Backbone] Loaded {len(dataset)} subjects. Total trials: {sum(d['n_trials'] for d in dataset.values())}")

    # Align each subject to identity spatial covariance
    X_all, y_all, groups_all, cues_all, records, alignments = align_multisubject_data(dataset)

    # Train / Validation split: use dedicated validation subjects or whole-cue split
    val_mask = np.isin(groups_all, [f"sub-{s:02d}" for s in val_subjects])
    train_mask = ~val_mask

    train_ds = TensorDataset(torch.from_numpy(X_all[train_mask]), torch.from_numpy(y_all[train_mask]).long())
    val_ds = TensorDataset(torch.from_numpy(X_all[val_mask]), torch.from_numpy(y_all[val_mask]).long())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = TemporalEEGNet(n_channels=len(channels), n_classes=2, window_standardization=True)
    t0 = time.time()
    trained_model, best_val_ba, history = train_temporal_backbone(
        model, train_loader, val_loader, epochs=epochs, lr=1e-3, weight_decay=1e-2, patience=10, device=device
    )
    elapsed = time.time() - t0

    actual_pretrain_subjects = [f"sub-{s:02d}" for s in train_subjects if s not in val_subjects]

    # Save checkpoint
    torch.save({
        'model_state': trained_model.state_dict(),
        'arch': 'TemporalEEGNet',
        'n_channels': len(channels),
        'channels': channels,
        'n_classes': 2,
        'classes': BINARY_CLASSES,
        'task': 'binary_rest_action',
        'trained_subjects': actual_pretrain_subjects,
        'validation_subjects': [f"sub-{s:02d}" for s in val_subjects],
        'best_val_ba': best_val_ba,
        'elapsed_s': elapsed,
    }, output_path)

    contract = {
        'model_path': str(output_path),
        'sha256': hashlib.sha256(output_path.read_bytes()).hexdigest(),
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'task': 'binary_rest_action',
        'channels': channels,
        'n_channels': len(channels),
        'sfreq': 256,
        'classes': BINARY_CLASSES,
        'n_classes': 2,
        'pretrain_subjects': actual_pretrain_subjects,
        'validation_subjects': [f"sub-{s:02d}" for s in val_subjects],
        'n_train_windows': int(train_mask.sum()),
        'n_val_windows': int(val_mask.sum()),
        'best_val_balanced_accuracy': float(best_val_ba),
        'training_time_s': float(elapsed),
        'history': history,
    }
    Path(str(output_path) + '.json').write_text(json.dumps(contract, indent=2), encoding='utf-8')
    print(f"[Binary Backbone] Saved: {output_path} (Val Bal Acc: {best_val_ba:.4f}, Elapsed: {elapsed:.1f}s)")
    return trained_model, contract


def train_threeclass_backbone(data_root=r"C:\Users\giaco\Box\BCIFEST\bids",
                              train_subjects=(4, 7),
                              channels=COMMON_X24,
                              classes=THREE_CLASSES,
                              epochs=18,
                              batch_size=64,
                              output_path=Path("decoder_and_models/backbone_threeclass_temporaleegnet.pt"),
                              device='cpu'):
    """Pretrain three-class backbone across subjects (sub-04 and sub-07)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[Three-Class Backbone] Loading subjects {train_subjects} from {data_root}...")
    dataset = load_bcifest_multiclass_subjects(data_root, train_subjects, channels=channels, classes=classes)
    print(f"[Three-Class Backbone] Loaded {len(dataset)} subjects. Total trials: {sum(d['n_trials'] for d in dataset.values())}")

    # Split whole cues per subject FIRST, then fit EuclideanAlignment strictly on train split
    X_parts = []
    y_parts = []
    is_val_parts = []
    for sub_str, data in dataset.items():
        sub_train_idx, sub_val_idx = split_whole_cues_stratified(
            data['records'], data['y'], data['cue_ids'], test_size=0.20, random_state=42
        )
        sub_is_val = np.zeros(len(data['y']), dtype=bool)
        sub_is_val[sub_val_idx] = True

        ea = EuclideanAlignment(shrinkage=0.05).fit(data['X'][sub_train_idx])
        X_parts.append(ea.transform(data['X']))
        y_parts.append(data['y'])
        is_val_parts.append(sub_is_val)

    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    is_val_all = np.concatenate(is_val_parts, axis=0)
    train_idx = np.flatnonzero(~is_val_all)
    val_idx = np.flatnonzero(is_val_all)

    train_ds = TensorDataset(torch.from_numpy(X_all[train_idx]), torch.from_numpy(y_all[train_idx]).long())
    val_ds = TensorDataset(torch.from_numpy(X_all[val_idx]), torch.from_numpy(y_all[val_idx]).long())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = TemporalEEGNet(n_channels=len(channels), n_classes=len(classes), window_standardization=True)
    t0 = time.time()
    trained_model, best_val_ba, history = train_temporal_backbone(
        model, train_loader, val_loader, epochs=epochs, lr=1e-3, weight_decay=1e-2, patience=10, device=device
    )
    elapsed = time.time() - t0

    torch.save({
        'model_state': trained_model.state_dict(),
        'arch': 'TemporalEEGNet',
        'n_channels': len(channels),
        'channels': channels,
        'n_classes': len(classes),
        'classes': classes,
        'task': 'threeclass_rest_volar_finger',
        'trained_subjects': [f"sub-{s:02d}" for s in train_subjects],
        'best_val_ba': best_val_ba,
        'elapsed_s': elapsed,
    }, output_path)

    contract = {
        'model_path': str(output_path),
        'sha256': hashlib.sha256(output_path.read_bytes()).hexdigest(),
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'task': 'threeclass_rest_volar_finger',
        'channels': channels,
        'n_channels': len(channels),
        'sfreq': 256,
        'classes': classes,
        'n_classes': len(classes),
        'pretrain_subjects': [f"sub-{s:02d}" for s in train_subjects],
        'n_train_windows': len(train_idx),
        'n_val_windows': len(val_idx),
        'best_val_balanced_accuracy': float(best_val_ba),
        'training_time_s': float(elapsed),
        'history': history,
    }
    Path(str(output_path) + '.json').write_text(json.dumps(contract, indent=2), encoding='utf-8')
    print(f"[Three-Class Backbone] Saved: {output_path} (Val Bal Acc: {best_val_ba:.4f}, Elapsed: {elapsed:.1f}s)")
    return trained_model, contract


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train multi-subject backbones.")
    parser.add_argument('--binary-only', action='store_true')
    parser.add_argument('--threeclass-only', action='store_true')
    args = parser.parse_args()

    if not args.threeclass_only:
        train_binary_backbone()
    if not args.binary_only:
        train_threeclass_backbone()
