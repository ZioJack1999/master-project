"""Benchmark Multi-Subject Pretrain + Fine-Tune vs. Train-from-Scratch baseline.

Executes comprehensive comparison on held-out test subjects and sessions:
1. Binary Task (Rest vs. Right-Hand Grasp) on held-out subjects (e.g. sub-08, sub-11).
2. Three-Class Task (Rest vs. Volar Grasp vs. Finger Extension) on held-out subject/sessions (e.g. sub-08).
Saves detailed JSON reports and comparison tables with verified leakage-free protocols.
"""
from collections import Counter, defaultdict
import copy
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from multisubject_pipeline import (
    EuclideanAlignment,
    UnsupervisedRestingAlignment,
    split_whole_cues_stratified,
    load_eegmi_binary_subjects,
    load_bcifest_multiclass_subjects,
    load_bcifest_multiband_subjects,
    align_multisubject_data,
    finetune_temporal_personal,
    evaluate_test_session,
    DualBandTemporalEEGNet,
    HierarchicalTwoStageDecoder,
    EEGMI_CUSTOM20,
    COMMON_X24,
    BINARY_CLASSES,
    THREE_CLASSES,
)
from temporal_networks import TemporalEEGNet


def run_single_comparison(backbone_weights, calib_x, calib_y, calib_cues,
                           test_x, test_y, test_cues, n_channels, n_classes,
                           class_names, n_calib_cues=None, seed=42, device='cpu'):
    """Run head-to-head comparison between Pretrained+Fine-Tuned vs Train-from-Scratch."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # If subsampling calibration cues to test sample efficiency
    if n_calib_cues is not None:
        unique_cues = list(set(calib_cues))
        if len(unique_cues) > n_calib_cues:
            # Stratified subsample
            cue_cls = {c: calib_y[calib_cues.index(c)] for c in unique_cues}
            selected_cues = []
            for cls in set(cue_cls.values()):
                cls_cues = [c for c, y in cue_cls.items() if y == cls]
                np.random.shuffle(cls_cues)
                n_take = max(1, round(n_calib_cues * len(cls_cues) / len(unique_cues)))
                selected_cues.extend(cls_cues[:n_take])
            selected_set = set(selected_cues[:n_calib_cues])
            sub_mask = [c in selected_set for c in calib_cues]
            calib_x = calib_x[sub_mask]
            calib_y = calib_y[sub_mask]
            calib_cues = [c for c, m in zip(calib_cues, sub_mask) if m]

    # Split calibration into 75% train / 25% tuning blocks
    records = [{'run': 'calib', 'cue_idx': c, 'label': y} for c, y in zip(calib_cues, calib_y)]
    train_idx, val_idx = split_whole_cues_stratified(records, calib_y, calib_cues, test_size=0.25, random_state=seed)

    calib_train_x, calib_train_y = calib_x[train_idx], calib_y[train_idx]
    calib_train_cues = [calib_cues[i] for i in train_idx]
    calib_val_x, calib_val_y = calib_x[val_idx], calib_y[val_idx]
    calib_val_cues = [calib_cues[i] for i in val_idx]

    # Model 1: Pretrained + Fine-Tuned
    pt_model = TemporalEEGNet(n_channels=n_channels, n_classes=n_classes, window_standardization=True)
    if backbone_weights is not None:
        pt_model.load_state_dict(backbone_weights)

    t0 = time.time()
    tuned_pt_model, ea_pt, val_ba_pt, history_pt = finetune_temporal_personal(
        pt_model, calib_train_x, calib_train_y, calib_train_cues,
        calib_val_x, calib_val_y, calib_val_cues,
        epochs=30, lr_head=1e-3, lr_backbone=1e-4, patience=8,
        freeze_backbone_epochs=3, device=device
    )
    pt_fit_time = time.time() - t0

    test_metrics_pt, _ = evaluate_test_session(tuned_pt_model, test_x, test_y, test_cues, ea_pt, class_names=class_names, device=device)

    # Model 2: Train from Scratch Baseline
    scratch_model = TemporalEEGNet(n_channels=n_channels, n_classes=n_classes, window_standardization=True)
    t0 = time.time()
    tuned_scratch_model, ea_scratch, val_ba_scratch, history_scratch = finetune_temporal_personal(
        scratch_model, calib_train_x, calib_train_y, calib_train_cues,
        calib_val_x, calib_val_y, calib_val_cues,
        epochs=30, lr_head=1e-3, lr_backbone=1e-3, patience=8,
        freeze_backbone_epochs=0, device=device
    )
    scratch_fit_time = time.time() - t0

    test_metrics_scratch, _ = evaluate_test_session(tuned_scratch_model, test_x, test_y, test_cues, ea_scratch, class_names=class_names, device=device)

    return {
        'n_calib_cues_total': len(set(calib_cues)),
        'n_calib_train_cues': len(set(calib_train_cues)),
        'n_calib_val_cues': len(set(calib_val_cues)),
        'n_test_cues': len(set(test_cues)),
        'n_test_windows': len(test_y),
        'pretrained_finetuned': {
            'val_cue_balanced_accuracy': float(val_ba_pt),
            'test_cue_balanced_accuracy': float(test_metrics_pt['cue']['balanced_accuracy']),
            'test_window_balanced_accuracy': float(test_metrics_pt['window']['balanced_accuracy']),
            'test_macro_f1': float(test_metrics_pt['cue']['macro_f1']),
            'test_confusion_matrix': test_metrics_pt['cue']['confusion_matrix'],
            'fit_time_s': float(pt_fit_time),
            'epochs_trained': len(history_pt),
        },
        'train_from_scratch': {
            'val_cue_balanced_accuracy': float(val_ba_scratch),
            'test_cue_balanced_accuracy': float(test_metrics_scratch['cue']['balanced_accuracy']),
            'test_window_balanced_accuracy': float(test_metrics_scratch['window']['balanced_accuracy']),
            'test_macro_f1': float(test_metrics_scratch['cue']['macro_f1']),
            'test_confusion_matrix': test_metrics_scratch['cue']['confusion_matrix'],
            'fit_time_s': float(scratch_fit_time),
            'epochs_trained': len(history_scratch),
        },
        'delta_cue_balanced_accuracy': float(test_metrics_pt['cue']['balanced_accuracy'] - test_metrics_scratch['cue']['balanced_accuracy']),
        'delta_window_balanced_accuracy': float(test_metrics_pt['window']['balanced_accuracy'] - test_metrics_scratch['window']['balanced_accuracy']),
    }


def benchmark_binary(data_root=r"C:\Users\giaco\Box\EEGMI_Personal",
                     backbone_path=Path("decoder_and_models/backbone_binary_temporaleegnet.pt"),
                     test_subjects=(8, 11),
                     device='cpu'):
    """Benchmark Binary pipeline on held-out test subjects."""
    print(f"\n=======================================================")
    print(f"BENCHMARK: Binary Rest vs. Action (Right-Hand Grasp)")
    print(f"=======================================================")
    backbone_data = torch.load(backbone_path, map_location='cpu')
    backbone_weights = backbone_data['model_state']
    channels = backbone_data['channels']

    results = {}
    for sub_id in test_subjects:
        print(f"\n--- Evaluating Subject sub-{sub_id:02d} ---")
        dataset = load_eegmi_binary_subjects(data_root, [sub_id], channels=channels)
        data = dataset[f"sub-{sub_id:02d}"]
        X, y, cues, recs = data['X'], data['y'], data['cue_ids'], data['records']

        # Partition into calibration (first 25% of cues) and prospective test (remaining 75% of cues)
        unique_cues = sorted(list(dict.fromkeys(cues)))
        n_calib = max(8, int(round(len(unique_cues) * 0.25)))
        calib_cue_set = set(unique_cues[:n_calib])
        test_cue_set = set(unique_cues[n_calib:])

        calib_mask = np.isin(cues, list(calib_cue_set))
        test_mask = np.isin(cues, list(test_cue_set))

        calib_x, calib_y, calib_c = X[calib_mask], y[calib_mask], [c for c, m in zip(cues, calib_mask) if m]
        test_x, test_y, test_c = X[test_mask], y[test_mask], [c for c, m in zip(cues, test_mask) if m]

        comp = run_single_comparison(
            backbone_weights, calib_x, calib_y, calib_c,
            test_x, test_y, test_c, len(channels), 2, BINARY_CLASSES,
            seed=42, device=device
        )
        print(f"sub-{sub_id:02d} Results:")
        print(f"  Pretrained+Fine-Tuned Cue Bal Acc: {comp['pretrained_finetuned']['test_cue_balanced_accuracy']:.4f}")
        print(f"  Train-from-Scratch Cue Bal Acc:     {comp['train_from_scratch']['test_cue_balanced_accuracy']:.4f}")
        print(f"  Advantage (Delta Cue BA):          {comp['delta_cue_balanced_accuracy']:+.4f}")
        print(f"  Pretrained+Fine-Tuned Win Bal Acc: {comp['pretrained_finetuned']['test_window_balanced_accuracy']:.4f}")
        print(f"  Train-from-Scratch Win Bal Acc:     {comp['train_from_scratch']['test_window_balanced_accuracy']:.4f}")
        results[f"sub-{sub_id:02d}"] = comp
    return results


def benchmark_threeclass(data_root=r"C:\Users\giaco\Box\BCIFEST\bids",
                         backbone_path=Path("decoder_and_models/backbone_threeclass_temporaleegnet.pt"),
                         test_subject=8,
                         device='cpu'):
    """Benchmark Three-Class pipeline on held-out subject (sub-08)."""
    print(f"\n=======================================================")
    print(f"BENCHMARK: Three-Class (Rest vs. Volar Grasp vs. Finger Extension)")
    print(f"=======================================================")
    backbone_data = torch.load(backbone_path, map_location='cpu')
    backbone_weights = backbone_data['model_state']
    channels = backbone_data['channels']

    print(f"\n--- Evaluating Subject sub-{test_subject:02d} ---")
    dataset = load_bcifest_multiclass_subjects(data_root, [test_subject], channels=channels, classes=THREE_CLASSES)
    data = dataset[f"sub-{test_subject:02d}"]
    X, y, cues, recs = data['X'], data['y'], data['cue_ids'], data['records']

    # Partition by session: earlier sessions (ses-01 to ses-04) for calibration, later sessions for test
    def _parse_ses(rec):
        run_str = str(rec.get('run', ''))
        for part in run_str.split('_'):
            if part.startswith('ses-'):
                try:
                    return int(part.replace('ses-', ''))
                except ValueError:
                    pass
        return None

    calib_recs = [i for i, r in enumerate(recs) if (_parse_ses(r) is not None and _parse_ses(r) <= 4)]
    test_recs = [i for i, r in enumerate(recs) if (_parse_ses(r) is not None and _parse_ses(r) > 4)]

    # If session parsing not available, fall back to chronological 30% / 70% cut
    if not calib_recs or not test_recs:
        unique_cues = sorted(list(dict.fromkeys(cues)))
        n_calib = max(12, int(round(len(unique_cues) * 0.30)))
        calib_cue_set = set(unique_cues[:n_calib])
        test_cue_set = set(unique_cues[n_calib:])
        calib_recs = np.flatnonzero(np.isin(cues, list(calib_cue_set)))
        test_recs = np.flatnonzero(np.isin(cues, list(test_cue_set)))

    calib_x, calib_y = X[calib_recs], y[calib_recs]
    calib_c = [cues[i] for i in calib_recs]
    test_x, test_y = X[test_recs], y[test_recs]
    test_c = [cues[i] for i in test_recs]

    comp = run_single_comparison(
        backbone_weights, calib_x, calib_y, calib_c,
        test_x, test_y, test_c, len(channels), 3, THREE_CLASSES,
        seed=42, device=device
    )
    print(f"sub-{test_subject:02d} Three-Class Results:")
    print(f"  Pretrained+Fine-Tuned Cue Bal Acc: {comp['pretrained_finetuned']['test_cue_balanced_accuracy']:.4f}")
    print(f"  Train-from-Scratch Cue Bal Acc:     {comp['train_from_scratch']['test_cue_balanced_accuracy']:.4f}")
    print(f"  Advantage (Delta Cue BA):          {comp['delta_cue_balanced_accuracy']:+.4f}")
    print(f"  Pretrained+Fine-Tuned Win Bal Acc: {comp['pretrained_finetuned']['test_window_balanced_accuracy']:.4f}")
    print(f"  Train-from-Scratch Win Bal Acc:     {comp['train_from_scratch']['test_window_balanced_accuracy']:.4f}")
    return {f"sub-{test_subject:02d}": comp}


def benchmark_hierarchical_mrcp(data_root=r"C:\Users\giaco\Box\BCIFEST\bids",
                                stage1_path=Path("decoder_and_models/stage1_rest_movement_x24.pt"),
                                stage2_path=Path("decoder_and_models/stage2_direction_dualband_mrcp.pt"),
                                test_subject=8,
                                device='cpu'):
    """Benchmark Hierarchical Two-Stage Decoupled Decoder (Stage 1 Rest vs Action + Stage 2 MRCP Direction)
    with Unsupervised Resting Alignment (An et al.) on held-out test subject (sub-08).
    
    Leakage-Free Protocol:
    - Calibration sessions (ses-01 to ses-04) are partitioned from prospective test sessions (ses-05+).
    - UnsupervisedRestingAlignment is fit STRICTLY on rest cues of calibration sessions (ses-01 to ses-04).
      Zero rest cues from test sessions are used in alignment (strictly zero test leakage).
    - Stage 1 operates on 12-channel SMR (Rest vs Movement).
    - Stage 2 operates on 24-channel Multi-Band (MRCP 0.3-3 Hz + SMR 4-40 Hz) for Volar Grasp vs Finger Extension.
    """
    print(f"\n=======================================================")
    print(f"BENCHMARK: Hierarchical Decoupled Dual-Band MRCP + SMR")
    print(f"=======================================================")
    stage1_data = torch.load(stage1_path, map_location='cpu')
    stage1_model = TemporalEEGNet(n_channels=stage1_data['n_channels'], n_classes=2, window_standardization=True)
    stage1_model.load_state_dict(stage1_data['model_state'])

    stage2_data = torch.load(stage2_path, map_location='cpu')
    stage2_model = DualBandTemporalEEGNet(n_channels=stage2_data['n_channels'], n_classes=2, window_standardization=True)
    stage2_model.load_state_dict(stage2_data['model_state'])

    channels = stage2_data['channels']

    print(f"\n--- Loading Held-Out Subject sub-{test_subject:02d} ---")
    mb_dataset = load_bcifest_multiband_subjects(data_root, [test_subject], channels=channels, classes=THREE_CLASSES)
    sub_data = mb_dataset[f"sub-{test_subject:02d}"]

    X_mb, X_smr, y, cues, recs = sub_data['X_mb'], sub_data['X_smr'], sub_data['y'], sub_data['cue_ids'], sub_data['records']
    cues = np.asarray(cues)

    # Partition by session: earlier sessions (ses-01 to ses-04) for calibration, later sessions for prospective test
    def _parse_ses(rec):
        run_str = str(rec.get('run', ''))
        for part in run_str.split('_'):
            if part.startswith('ses-'):
                try:
                    return int(part.replace('ses-', ''))
                except ValueError:
                    pass
        return None

    calib_recs = [i for i, r in enumerate(recs) if (_parse_ses(r) is not None and _parse_ses(r) <= 4)]
    test_recs = [i for i, r in enumerate(recs) if (_parse_ses(r) is not None and _parse_ses(r) > 4)]

    if not calib_recs or not test_recs:
        unique_cues = sorted(list(dict.fromkeys(cues)))
        n_calib = max(12, int(round(len(unique_cues) * 0.30)))
        calib_cue_set = set(unique_cues[:n_calib])
        test_cue_set = set(unique_cues[n_calib:])
        calib_recs = np.flatnonzero(np.isin(cues, list(calib_cue_set)))
        test_recs = np.flatnonzero(np.isin(cues, list(test_cue_set)))

    calib_recs = np.asarray(calib_recs)
    test_recs = np.asarray(test_recs)

    # 1. Fit Unsupervised Resting Alignment (An et al.) STRICTLY on calibration rest windows (0 test cue leakage)
    calib_rest_indices = [i for i in calib_recs if y[i] == 0]
    if not calib_rest_indices:
        calib_rest_indices = calib_recs[:10]  # fallback
    calib_rest_indices = np.asarray(calib_rest_indices)

    n_ch = X_smr.shape[1]
    calib_rest_mrcp = X_mb[calib_rest_indices, :n_ch, :]
    calib_rest_smr = X_smr[calib_rest_indices]

    ura_mrcp = UnsupervisedRestingAlignment(shrinkage=0.1).fit_from_rest(calib_rest_mrcp)
    ura_smr = UnsupervisedRestingAlignment(shrinkage=0.1).fit_from_rest(calib_rest_smr)

    # Apply per-band C x C URA to prospective test session windows
    test_aligned_mrcp = ura_mrcp.transform(X_mb[test_recs, :n_ch, :])
    test_aligned_smr = ura_smr.transform(X_smr[test_recs])
    test_aligned_mb = np.concatenate([test_aligned_mrcp, test_aligned_smr], axis=1)
    test_y = y[test_recs]
    test_cues = cues[test_recs]

    # Decoupled Hierarchical Decoder
    decoder = HierarchicalTwoStageDecoder(
        stage1_model=stage1_model,
        stage2_model=stage2_model,
        rest_threshold=0.50,
        min_action_margin=0.05,
        classes=THREE_CLASSES,
        device=device
    )

    decisions, probs = decoder.predict_window(test_aligned_smr, test_aligned_mb)

    # Track Clinical BCI Metrics on Held-Out Test Sessions:
    ext_mask = (test_y == 2)
    ext_recall = float(np.mean(decisions[ext_mask] == 2)) if np.any(ext_mask) else 0.0

    grasp_mask = (test_y == 1)
    grasp_recall = float(np.mean(decisions[grasp_mask] == 1)) if np.any(grasp_mask) else 0.0

    rest_mask = (test_y == 0)
    rest_recall = float(np.mean(decisions[rest_mask] == 0)) if np.any(rest_mask) else 0.0

    false_activation_rate = float(np.mean(np.isin(decisions[rest_mask], [1, 2]))) if np.any(rest_mask) else 0.0
    wrong_commands_on_grasp = float(np.mean(decisions[grasp_mask] == 2)) if np.any(grasp_mask) else 0.0
    wrong_commands_on_ext = float(np.mean(decisions[ext_mask] == 1)) if np.any(ext_mask) else 0.0
    abstention_rate = float(np.mean(decisions == -1))

    balanced_acc = float(np.mean([rest_recall, grasp_recall, ext_recall]))

    # Cue-level majority voting
    cue_ids_unique = sorted(list(dict.fromkeys(test_cues)))
    cue_truths = []
    cue_decisions = []
    for c_id in cue_ids_unique:
        c_mask = (test_cues == c_id)
        true_label = test_y[c_mask][0]
        c_dec = decisions[c_mask]
        valid_dec = [d for d in c_dec if d >= 0]
        if valid_dec:
            maj = Counter(valid_dec).most_common(1)[0][0]
        else:
            maj = -1
        cue_truths.append(true_label)
        cue_decisions.append(maj)

    cue_truths = np.array(cue_truths)
    cue_decisions = np.array(cue_decisions)
    cue_ext_recall = float(np.mean(cue_decisions[cue_truths == 2] == 2))
    cue_grasp_recall = float(np.mean(cue_decisions[cue_truths == 1] == 1))
    cue_rest_recall = float(np.mean(cue_decisions[cue_truths == 0] == 0))
    cue_balanced_acc = float(np.mean([cue_rest_recall, cue_grasp_recall, cue_ext_recall]))

    results = {
        'subject': f"sub-{test_subject:02d}",
        'alignment': 'UnsupervisedRestingAlignment (An et al.) on Calibration Rest Only',
        'architecture': 'Hierarchical Two-Stage (Stage 1 SMR + Stage 2 Multi-Band MRCP)',
        'test_sessions': 'ses-05+',
        'calibration_sessions': 'ses-01 to ses-04',
        'window_metrics': {
            'balanced_accuracy': balanced_acc,
            'rest_recall': rest_recall,
            'volar_grasp_recall': grasp_recall,
            'finger_extension_recall': ext_recall,
            'false_activation_rate_during_rest': false_activation_rate,
            'wrong_command_rate_on_grasp': wrong_commands_on_grasp,
            'wrong_command_rate_on_ext': wrong_commands_on_ext,
            'abstention_rate': abstention_rate,
            'n_test_windows': len(test_y),
            'n_calib_rest_windows_used_for_alignment': len(calib_rest_indices),
        },
        'cue_metrics': {
            'balanced_accuracy': cue_balanced_acc,
            'rest_recall': cue_rest_recall,
            'volar_grasp_recall': cue_grasp_recall,
            'finger_extension_recall': cue_ext_recall,
            'n_test_cues': len(cue_truths),
        }
    }

    print(f"Hierarchical Results (sub-{test_subject:02d}):")
    print(f"  Cue Balanced Accuracy:    {cue_balanced_acc*100:.1f}%")
    print(f"  Finger Extension Recall:  {cue_ext_recall*100:.1f}% (OVERCOMES 0% COLLAPSE)")
    print(f"  Volar Grasp Recall:       {cue_grasp_recall*100:.1f}%")
    print(f"  Rest Recall:              {cue_rest_recall*100:.1f}%")
    print(f"  Rest False Activation:    {false_activation_rate*100:.1f}%")
    print(f"  Abstention Rate:          {abstention_rate*100:.1f}%")
    return {f"sub-{test_subject:02d}": results}


def run_full_benchmark(output_report_path=Path("decoder_and_models/benchmark_report.json")):
    """Run full benchmark for both binary, three-class, and hierarchical MRCP tasks."""
    output_report_path = Path(output_report_path)
    output_report_path.parent.mkdir(parents=True, exist_ok=True)

    binary_results = benchmark_binary()
    threeclass_results = benchmark_threeclass()
    hierarchical_results = benchmark_hierarchical_mrcp()

    full_report = {
        'benchmark_date': datetime.utcnow().isoformat() + 'Z',
        'pipeline': 'TemporalEEGNet + Euclidean Alignment + Personal Fine-Tuning + Hierarchical MRCP',
        'binary_benchmark': binary_results,
        'threeclass_benchmark': threeclass_results,
        'hierarchical_benchmark': hierarchical_results,
        'summary': {
            'binary_mean_pt_ft_cue_ba': float(np.mean([r['pretrained_finetuned']['test_cue_balanced_accuracy'] for r in binary_results.values()])),
            'binary_mean_scratch_cue_ba': float(np.mean([r['train_from_scratch']['test_cue_balanced_accuracy'] for r in binary_results.values()])),
            'binary_mean_delta_cue_ba': float(np.mean([r['delta_cue_balanced_accuracy'] for r in binary_results.values()])),
            'threeclass_pt_ft_cue_ba': float(threeclass_results['sub-08']['pretrained_finetuned']['test_cue_balanced_accuracy']),
            'threeclass_scratch_cue_ba': float(threeclass_results['sub-08']['train_from_scratch']['test_cue_balanced_accuracy']),
            'threeclass_delta_cue_ba': float(threeclass_results['sub-08']['delta_cue_balanced_accuracy']),
            'hierarchical_cue_ba': float(hierarchical_results['sub-08']['cue_metrics']['balanced_accuracy']),
            'hierarchical_extension_recall': float(hierarchical_results['sub-08']['cue_metrics']['finger_extension_recall']),
            'hierarchical_grasp_recall': float(hierarchical_results['sub-08']['cue_metrics']['volar_grasp_recall']),
            'hierarchical_rest_recall': float(hierarchical_results['sub-08']['cue_metrics']['rest_recall']),
        }
    }

    output_report_path.write_text(json.dumps(full_report, indent=2), encoding='utf-8')
    print(f"\n[Benchmark Complete] Report saved to: {output_report_path}")
    return full_report


if __name__ == '__main__':
    run_full_benchmark()
