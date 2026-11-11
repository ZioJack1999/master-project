"""Train Hierarchical Decoupled Models for BCIFEST 12-channel (COMMON_X24):
Stage 1: Rest vs Movement (SMR 4-40 Hz, 12 channels)
Stage 2: Volar Grasp vs Finger Extension (DualBand MRCP 0.3-3 Hz + SMR 4-40 Hz, 24 channels)
Baseline Stage 2: Volar Grasp vs Finger Extension (SMR-Only 4-40 Hz, 12 channels)

Uses consistent Unsupervised Resting Alignment (An et al.) fit on resting cues only.
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
    load_bcifest_multiband_subjects,
    UnsupervisedRestingAlignment,
    split_whole_cues_stratified,
    DualBandTemporalEEGNet,
    train_temporal_backbone,
    evaluate_predictions,
    COMMON_X24,
)
from temporal_networks import TemporalEEGNet


def train_hierarchical_models(data_root=r"C:\Users\giaco\Box\BCIFEST\bids",
                              train_subjects=(4, 7),
                              channels=COMMON_X24,
                              output_dir=Path("decoder_and_models"),
                              epochs=18,
                              batch_size=32,
                              device='cpu'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("TRAINING HIERARCHICAL DECOUPLED BCI MODELS (12-Channel COMMON_X24)")
    print("Alignment: Unsupervised Resting Alignment (An et al.) on Resting Cues")
    print("="*70)

    # 1. Load multi-band dataset for sub-04 and sub-07
    print(f"\n[Multi-Band Loader] Loading training subjects {train_subjects} from {data_root}...")
    dataset = load_bcifest_multiband_subjects(
        data_root, train_subjects, channels=channels,
        classes=['rest', 'volar_grasp', 'finger_extension']
    )

    # 2. Split whole cues per subject FIRST, then fit per-band C x C URA strictly on train-split rest windows
    uras_mb = {}
    uras_smr = {}
    aligned_mb_parts = []
    aligned_smr_parts = []
    y_parts = []
    groups_parts = []
    cues_parts = []
    records_parts = []
    is_val_parts = []
    cue_offset = 0

    n_ch = len(channels)
    for sub_str, data in dataset.items():
        sub_train_idx, sub_val_idx = split_whole_cues_stratified(
            data['records'], data['y'], data['cue_ids'], test_size=0.25, random_state=42
        )
        sub_is_val = np.zeros(len(data['y']), dtype=bool)
        sub_is_val[sub_val_idx] = True

        # Select ONLY train-split rest windows for URA fitting (strictly 0 val-split windows)
        train_rest_indices = [i for i in sub_train_idx if data['y'][i] == 0]
        if not train_rest_indices:
            train_rest_indices = sub_train_idx
        train_rest_indices = np.asarray(train_rest_indices)

        rest_mrcp = data['X_mb'][train_rest_indices, :n_ch, :]  # C-channel MRCP slice
        rest_smr = data['X_smr'][train_rest_indices]            # C-channel SMR slice

        # Fit per-band C x C URA on train rest windows (strictly no 2C x 2C cross-band mixing)
        ura_mrcp = UnsupervisedRestingAlignment(shrinkage=0.1).fit_from_rest(rest_mrcp)
        ura_smr = UnsupervisedRestingAlignment(shrinkage=0.1).fit_from_rest(rest_smr)
        uras_mb[sub_str] = {'mrcp': ura_mrcp, 'smr': ura_smr}
        uras_smr[sub_str] = ura_smr

        # Transform all windows for this subject per C-channel band
        sub_aligned_mrcp = ura_mrcp.transform(data['X_mb'][:, :n_ch, :])
        sub_aligned_smr = ura_smr.transform(data['X_smr'])
        sub_aligned_mb = np.concatenate([sub_aligned_mrcp, sub_aligned_smr], axis=1)

        aligned_mb_parts.append(sub_aligned_mb)
        aligned_smr_parts.append(sub_aligned_smr)
        y_parts.append(data['y'])
        groups_parts.append(np.array([sub_str] * len(data['y'])))
        is_val_parts.append(sub_is_val)

        sub_cues = [cue_offset + c for c in data['cue_ids']]
        cues_parts.append(np.array(sub_cues))

        for r in data['records']:
            rc = dict(r)
            rc['global_cue'] = cue_offset + r['global_cue']
            records_parts.append(rc)
        cue_offset += len(set(data['cue_ids'])) + 100

    X_mb_all = np.concatenate(aligned_mb_parts, axis=0)
    X_smr_all = np.concatenate(aligned_smr_parts, axis=0)
    y_3class = np.concatenate(y_parts, axis=0)
    groups_all = np.concatenate(groups_parts, axis=0)
    cues_all = np.concatenate(cues_parts, axis=0)
    is_val_all = np.concatenate(is_val_parts, axis=0)

    print(f"[Total Data] {len(y_3class)} windows across subjects: Rest={np.sum(y_3class==0)}, Grasp={np.sum(y_3class==1)}, Ext={np.sum(y_3class==2)}")

    # =========================================================================
    # PART 1: Stage 1 Rest vs Movement (SMR 12-Channel)
    # =========================================================================
    print("\n" + "-"*70)
    print("TRAINING STAGE 1: Rest (0) vs. Intentional Movement (1) [12 channels SMR]")
    print("-"*70)
    # 0 = rest, 1 = movement (grasp or extension)
    y_stage1 = np.where(y_3class == 0, 0, 1)
    train_idx1 = np.flatnonzero(~is_val_all)
    val_idx1 = np.flatnonzero(is_val_all)
    print(f"[Stage 1 Split] Train: {len(train_idx1)} (Rest={np.sum(y_stage1[train_idx1]==0)}, Mov={np.sum(y_stage1[train_idx1]==1)}) | Val: {len(val_idx1)}")

    s1_train_ds = TensorDataset(torch.from_numpy(X_smr_all[train_idx1]), torch.from_numpy(y_stage1[train_idx1]).long())
    s1_val_ds = TensorDataset(torch.from_numpy(X_smr_all[val_idx1]), torch.from_numpy(y_stage1[val_idx1]).long())
    s1_train_loader = DataLoader(s1_train_ds, batch_size=batch_size, shuffle=True)
    s1_val_loader = DataLoader(s1_val_ds, batch_size=batch_size, shuffle=False)

    stage1_model = TemporalEEGNet(n_channels=len(channels), n_classes=2, window_standardization=True)
    t0 = time.time()
    stage1_trained, s1_best_ba, s1_history = train_temporal_backbone(
        stage1_model, s1_train_loader, s1_val_loader, epochs=epochs, lr=1e-3, weight_decay=1e-2, patience=8, device=device
    )
    s1_elapsed = time.time() - t0

    s1_out_path = output_dir / "stage1_rest_movement_x24.pt"
    torch.save({
        'model_state': stage1_trained.state_dict(),
        'arch': 'TemporalEEGNet',
        'n_channels': len(channels),
        'channels': channels,
        'n_classes': 2,
        'classes': ['rest', 'movement'],
        'task': 'stage1_rest_vs_movement',
        'best_val_ba': s1_best_ba,
    }, s1_out_path)

    s1_contract = {
        'model_path': str(s1_out_path),
        'sha256': hashlib.sha256(s1_out_path.read_bytes()).hexdigest(),
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'task': 'stage1_rest_vs_movement',
        'channels': channels,
        'n_channels': len(channels),
        'classes': ['rest', 'movement'],
        'best_val_balanced_accuracy': float(s1_best_ba),
        'training_time_s': float(s1_elapsed),
    }
    Path(str(s1_out_path) + '.json').write_text(json.dumps(s1_contract, indent=2), encoding='utf-8')
    print(f"[Stage 1 Model] Saved: {s1_out_path} (Val Bal Acc: {s1_best_ba:.4f}, Elapsed: {s1_elapsed:.1f}s)")

    # =========================================================================
    # PART 2: Stage 2 Volar Grasp vs Finger Extension (Dual-Band MRCP + SMR)
    # =========================================================================
    print("\n" + "-"*70)
    print("TRAINING STAGE 2: Volar Grasp (0) vs. Finger Extension (1) [DualBand MRCP+SMR]")
    print("-"*70)
    mov_mask = (y_3class == 1) | (y_3class == 2)
    X_mb_mov = X_mb_all[mov_mask]
    X_smr_mov = X_smr_all[mov_mask]
    y_stage2 = np.where(y_3class[mov_mask] == 1, 0, 1)
    cues_mov = cues_all[mov_mask]
    is_val_mov = is_val_all[mov_mask]

    train_idx2 = np.flatnonzero(~is_val_mov)
    val_idx2 = np.flatnonzero(is_val_mov)
    print(f"[Stage 2 Split] Train: {len(train_idx2)} (Grasp={np.sum(y_stage2[train_idx2]==0)}, Ext={np.sum(y_stage2[train_idx2]==1)}) | Val: {len(val_idx2)}")

    s2_train_ds = TensorDataset(torch.from_numpy(X_mb_mov[train_idx2]), torch.from_numpy(y_stage2[train_idx2]).long())
    s2_val_ds = TensorDataset(torch.from_numpy(X_mb_mov[val_idx2]), torch.from_numpy(y_stage2[val_idx2]).long())
    s2_train_loader = DataLoader(s2_train_ds, batch_size=batch_size, shuffle=True)
    s2_val_loader = DataLoader(s2_val_ds, batch_size=batch_size, shuffle=False)

    stage2_mb_model = DualBandTemporalEEGNet(n_channels=len(channels), n_classes=2, window_standardization=True)
    t0 = time.time()
    stage2_mb_trained, s2_mb_best_ba, s2_mb_history = train_temporal_backbone(
        stage2_mb_model, s2_train_loader, s2_val_loader, epochs=epochs, lr=1e-3, weight_decay=1e-2, patience=8, device=device
    )
    s2_mb_elapsed = time.time() - t0

    # Evaluate validation metrics
    stage2_mb_trained.eval()
    val_probs_mb = []
    with torch.no_grad():
        for bx, _ in s2_val_loader:
            val_probs_mb.append(torch.softmax(stage2_mb_trained(bx), dim=1).numpy())
    val_probs_mb = np.concatenate(val_probs_mb, axis=0)
    mb_val_eval = evaluate_predictions(y_stage2[val_idx2], val_probs_mb, cues_mov[val_idx2], class_names=['volar_grasp', 'finger_extension'])

    s2_mb_out_path = output_dir / "stage2_direction_dualband_mrcp.pt"
    torch.save({
        'model_state': stage2_mb_trained.state_dict(),
        'arch': 'DualBandTemporalEEGNet',
        'n_channels': len(channels),
        'channels': channels,
        'n_classes': 2,
        'classes': ['volar_grasp', 'finger_extension'],
        'bands': {'mrcp': [0.3, 3.0], 'smr': [4.0, 40.0]},
        'best_val_ba': s2_mb_best_ba,
        'val_metrics': mb_val_eval,
    }, s2_mb_out_path)

    s2_mb_contract = {
        'model_path': str(s2_mb_out_path),
        'sha256': hashlib.sha256(s2_mb_out_path.read_bytes()).hexdigest(),
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'arch': 'DualBandTemporalEEGNet',
        'task': 'stage2_volar_vs_extension_dualband_mrcp',
        'channels': channels,
        'n_channels': len(channels),
        'sfreq': 256,
        'classes': ['volar_grasp', 'finger_extension'],
        'n_classes': 2,
        'pretrain_subjects': [f"sub-{s:02d}" for s in train_subjects],
        'best_val_balanced_accuracy': float(s2_mb_best_ba),
        'val_extension_recall': float(mb_val_eval['window']['class_recalls']['finger_extension']),
        'val_grasp_recall': float(mb_val_eval['window']['class_recalls']['volar_grasp']),
        'training_time_s': float(s2_mb_elapsed),
    }
    Path(str(s2_mb_out_path) + '.json').write_text(json.dumps(s2_mb_contract, indent=2), encoding='utf-8')
    print(f"[Stage 2 Dual-Band MRCP] Saved: {s2_mb_out_path}")
    print(f"  Val Bal Acc: {s2_mb_best_ba:.4f} | Grasp Recall: {s2_mb_contract['val_grasp_recall']:.4f} | Extension Recall: {s2_mb_contract['val_extension_recall']:.4f}")

    # =========================================================================
    # PART 3: Stage 2 Baseline Volar Grasp vs Finger Extension (SMR-Only)
    # =========================================================================
    print("\n" + "-"*70)
    print("TRAINING STAGE 2 BASELINE: Volar Grasp vs. Finger Extension [SMR-Only 4-40 Hz]")
    print("-"*70)
    s2_smr_train_ds = TensorDataset(torch.from_numpy(X_smr_mov[train_idx2]), torch.from_numpy(y_stage2[train_idx2]).long())
    s2_smr_val_ds = TensorDataset(torch.from_numpy(X_smr_mov[val_idx2]), torch.from_numpy(y_stage2[val_idx2]).long())
    s2_smr_train_loader = DataLoader(s2_smr_train_ds, batch_size=batch_size, shuffle=True)
    s2_smr_val_loader = DataLoader(s2_smr_val_ds, batch_size=batch_size, shuffle=False)

    stage2_smr_model = TemporalEEGNet(n_channels=len(channels), n_classes=2, window_standardization=True)
    t0 = time.time()
    stage2_smr_trained, s2_smr_best_ba, s2_smr_history = train_temporal_backbone(
        stage2_smr_model, s2_smr_train_loader, s2_smr_val_loader, epochs=epochs, lr=1e-3, weight_decay=1e-2, patience=8, device=device
    )
    s2_smr_elapsed = time.time() - t0

    stage2_smr_trained.eval()
    val_probs_smr = []
    with torch.no_grad():
        for bx, _ in s2_smr_val_loader:
            val_probs_smr.append(torch.softmax(stage2_smr_trained(bx), dim=1).numpy())
    val_probs_smr = np.concatenate(val_probs_smr, axis=0)
    smr_val_eval = evaluate_predictions(y_stage2[val_idx2], val_probs_smr, cues_mov[val_idx2], class_names=['volar_grasp', 'finger_extension'])

    s2_smr_out_path = output_dir / "stage2_direction_smr_only.pt"
    torch.save({
        'model_state': stage2_smr_trained.state_dict(),
        'arch': 'TemporalEEGNet',
        'n_channels': len(channels),
        'channels': channels,
        'n_classes': 2,
        'classes': ['volar_grasp', 'finger_extension'],
        'bands': {'smr': [4.0, 40.0]},
        'best_val_ba': s2_smr_best_ba,
        'val_metrics': smr_val_eval,
    }, s2_smr_out_path)

    s2_smr_contract = {
        'model_path': str(s2_smr_out_path),
        'sha256': hashlib.sha256(s2_smr_out_path.read_bytes()).hexdigest(),
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'arch': 'TemporalEEGNet',
        'task': 'stage2_volar_vs_extension_smr_only',
        'channels': channels,
        'n_channels': len(channels),
        'sfreq': 256,
        'classes': ['volar_grasp', 'finger_extension'],
        'n_classes': 2,
        'pretrain_subjects': [f"sub-{s:02d}" for s in train_subjects],
        'best_val_balanced_accuracy': float(s2_smr_best_ba),
        'val_extension_recall': float(smr_val_eval['window']['class_recalls']['finger_extension']),
        'val_grasp_recall': float(smr_val_eval['window']['class_recalls']['volar_grasp']),
        'training_time_s': float(s2_smr_elapsed),
    }
    Path(str(s2_smr_out_path) + '.json').write_text(json.dumps(s2_smr_contract, indent=2), encoding='utf-8')
    print(f"[Stage 2 SMR Baseline] Saved: {s2_smr_out_path}")
    print(f"  Val Bal Acc: {s2_smr_best_ba:.4f} | Grasp Recall: {s2_smr_contract['val_grasp_recall']:.4f} | Extension Recall: {s2_smr_contract['val_extension_recall']:.4f}")

    print("\n" + "="*70)
    print("STAGE 2 EMPIRICAL COMPARISON")
    print("="*70)
    print(f"SMR-Only Baseline:   Bal Acc={s2_smr_best_ba*100:.1f}%, Grasp={s2_smr_contract['val_grasp_recall']*100:.1f}%, Extension Recall={s2_smr_contract['val_extension_recall']*100:.1f}%")
    print(f"Dual-Band MRCP+SMR:  Bal Acc={s2_mb_best_ba*100:.1f}%, Grasp={s2_mb_contract['val_grasp_recall']*100:.1f}%, Extension Recall={s2_mb_contract['val_extension_recall']*100:.1f}%")
    print("="*70)

    return {
        'stage1': s1_contract,
        'stage2_dualband_mrcp': s2_mb_contract,
        'stage2_smr_only': s2_smr_contract,
    }


if __name__ == '__main__':
    train_hierarchical_models()
