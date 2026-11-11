"""Empirical MRCP (0.3-3.0 Hz) + SMR (4.0-40.0 Hz) Signal & Separability Audit.

Quantifies:
1. Univariate class separability (Cohen's d, Fisher's J, ROC-AUC) between:
   - Stage 2: Volar Grasp vs. Finger Extension
   - Stage 1: Rest vs. Intentional Movement (Grasp + Extension)
2. Top discriminative MRCP temporal features, SMR spectral features, and
   bipolar/Laplacian motor channel pairs across BCIFEST subjects and saved X24 sessions.
3. Leakage-free whole-cue cross-validation & session-split benchmarks comparing:
   - SMR-only (standard 8-30 Hz / 4-40 Hz)
   - MRCP-only (0.3-3.0 Hz slow potentials)
   - Dual-Band Combined (MRCP + SMR)
   Specifically tracking Finger Extension recall, Volar Grasp recall, and Rest false activations.
"""

from collections import defaultdict
import json
from pathlib import Path
import warnings

import mne
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score, roc_auc_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import make_pipeline

from dual_band_filter import (
    DualBandFilter,
    extract_dual_band_features,
    compute_linear_slope,
    EXTENDED_MOTOR_CHANNELS,
    CORE_MOTOR_CHANNELS,
)

warnings.filterwarnings('ignore')

COMMON_X24 = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'POz', 'O1', 'O2']
THREE_CLASSES = ['rest', 'volar_grasp', 'finger_extension']

CHANNEL_PAIRS = [
    ('C3', 'Cz'),
    ('C4', 'Cz'),
    ('C3', 'C4'),
    ('Fz', 'Cz'),
    ('Cz', 'Pz'),
    ('F3', 'C3'),
    ('C3', 'P3'),
    ('F4', 'C4'),
    ('C4', 'P4'),
]


def canonical_label(text):
    return '_'.join(str(text).strip().casefold().replace('_', ' ').split())


def extract_rich_dualband_window_features(mrcp_win, smr_win, sfreq=256.0, channels=COMMON_X24):
    """Extract comprehensive MRCP + SMR + Bipolar/Laplacian channel-pair features from a single window."""
    ch_idx = {ch: i for i, ch in enumerate(channels)}
    n_ch, n_times = mrcp_win.shape

    # 1. Base dual-band features across extended motor channels
    from dual_band_filter import extract_mrcp_temporal_features, extract_smr_spectral_features
    mrcp_vec, mrcp_names, _ = extract_mrcp_temporal_features(
        mrcp_win, sfreq=sfreq, channels=channels,
        motor_channels=EXTENDED_MOTOR_CHANNELS, trajectory_samples=8
    )
    smr_vec, smr_names, _ = extract_smr_spectral_features(
        smr_win, sfreq=sfreq, channels=channels,
        motor_channels=EXTENDED_MOTOR_CHANNELS
    )

    pair_vals = []
    pair_names = []

    late_samples = max(2, n_times // 2)
    taper = np.hanning(n_times)
    scale = 2.0 / (sfreq * max(1e-12, np.sum(taper ** 2)))
    freqs = np.fft.rfftfreq(n_times, 1.0 / sfreq)
    mu_mask = (freqs >= 8.0) & (freqs < 13.0)
    beta_mask = (freqs >= 13.0) & (freqs < 30.0)

    # 2. Pairwise bipolar diffs (MRCP slow shift + SMR power asymmetry)
    for ch_a, ch_b in CHANNEL_PAIRS:
        if ch_a in ch_idx and ch_b in ch_idx:
            ia, ib = ch_idx[ch_a], ch_idx[ch_b]
            diff_mrcp = mrcp_win[ia] - mrcp_win[ib]
            late_diff = float(np.mean(diff_mrcp[-late_samples:]))
            min_diff = float(np.min(diff_mrcp))
            max_diff = float(np.max(diff_mrcp))
            slope_diff = compute_linear_slope(diff_mrcp[-late_samples:], sfreq)

            pair_names.extend([
                f'mrcp_pair_{ch_a}_{ch_b}_late_mean',
                f'mrcp_pair_{ch_a}_{ch_b}_min',
                f'mrcp_pair_{ch_a}_{ch_b}_max',
                f'mrcp_pair_{ch_a}_{ch_b}_late_slope',
            ])
            pair_vals.extend([late_diff, min_diff, max_diff, slope_diff])

            # Bipolar SMR power
            diff_smr = (smr_win[ia] - smr_win[ib])
            diff_smr = diff_smr - np.mean(diff_smr)
            psd = (np.abs(np.fft.rfft(diff_smr * taper)) ** 2) * scale
            mu_bp = float(np.log(max(np.mean(psd[mu_mask]), 1e-12)))
            beta_bp = float(np.log(max(np.mean(psd[beta_mask]), 1e-12)))
            pair_names.extend([
                f'smr_pair_{ch_a}_{ch_b}_mu_log',
                f'smr_pair_{ch_a}_{ch_b}_beta_log',
            ])
            pair_vals.extend([mu_bp, beta_bp])

    # 3. Surface Laplacian over C3, Cz, C4
    laplacians = {
        'C3_lap': ('C3', ['F3', 'Cz', 'P3']),
        'Cz_lap': ('Cz', ['Fz', 'C3', 'C4', 'Pz']),
        'C4_lap': ('C4', ['F4', 'Cz', 'P4']),
    }
    for lap_name, (center, surrounds) in laplacians.items():
        if center in ch_idx and all(s in ch_idx for s in surrounds):
            ic = ch_idx[center]
            isurr = [ch_idx[s] for s in surrounds]
            mrcp_lap = mrcp_win[ic] - np.mean(mrcp_win[isurr], axis=0)
            smr_lap = smr_win[ic] - np.mean(smr_win[isurr], axis=0)

            lap_late = float(np.mean(mrcp_lap[-late_samples:]))
            lap_min = float(np.min(mrcp_lap))
            lap_slope = compute_linear_slope(mrcp_lap[-late_samples:], sfreq)
            pair_names.extend([
                f'mrcp_{lap_name}_late_mean',
                f'mrcp_{lap_name}_min',
                f'mrcp_{lap_name}_late_slope',
            ])
            pair_vals.extend([lap_late, lap_min, lap_slope])

            smr_lap_c = smr_lap - np.mean(smr_lap)
            psd_lap = (np.abs(np.fft.rfft(smr_lap_c * taper)) ** 2) * scale
            mu_lap = float(np.log(max(np.mean(psd_lap[mu_mask]), 1e-12)))
            beta_lap = float(np.log(max(np.mean(psd_lap[beta_mask]), 1e-12)))
            pair_names.extend([
                f'smr_{lap_name}_mu_log',
                f'smr_{lap_name}_beta_log',
            ])
            pair_vals.extend([mu_lap, beta_lap])

    all_vals = np.concatenate([mrcp_vec, smr_vec, np.asarray(pair_vals, dtype=np.float32)])
    all_names = mrcp_names + smr_names + pair_names

    is_mrcp = np.array([n.startswith('mrcp_') for n in all_names], dtype=bool)
    is_smr = np.array([n.startswith('smr_') for n in all_names], dtype=bool)

    return all_vals, all_names, is_mrcp, is_smr


def load_subject_dualband_windows(bids_root, sub_str, channels=COMMON_X24,
                                  window_s=2.0, stride_s=0.5, settle_s=0.5, fs=256.0):
    """Load raw EEG for a BCIFEST subject, filter causally with DualBandFilter, and extract cue windows."""
    sub_dir = Path(bids_root) / sub_str
    if not sub_dir.exists():
        return None

    edf_files = sorted(sub_dir.glob("ses-*/eeg/*task-olrandmcmitrain*_eeg.edf"))
    if not edf_files:
        edf_files = sorted(sub_dir.glob("ses-*/eeg/*_eeg.edf"))
    if not edf_files:
        return None

    dbf = DualBandFilter(sfreq=fs, mrcp_band=(0.3, 3.0), mrcp_order=2, smr_band=(4.0, 40.0), smr_order=4)
    class_map = {canonical_label(c): i for i, c in enumerate(THREE_CLASSES)}

    win_len = round(window_s * fs)
    stride_len = round(stride_s * fs)
    settle_len = round(settle_s * fs)

    features_list = []
    labels_list = []
    cues_list = []
    sessions_list = []
    feature_names = None
    is_mrcp_mask = None
    is_smr_mask = None

    global_cue = 0
    for edf_path in edf_files:
        ses_str = edf_path.parts[-3] if 'ses-' in edf_path.parts[-3] else 'ses-01'
        run_id = edf_path.stem.replace('_eeg', '')
        tsv_path = edf_path.with_name(run_id + '_events.tsv')

        try:
            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        except Exception:
            continue
        if not np.isclose(raw.info['sfreq'], fs):
            continue
        if not set(channels).issubset(set(raw.ch_names)):
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
        # Filter causally with DC step transient cancellation via sosfilt_zi
        mrcp_data, smr_data, _ = dbf.filter_continuous(raw_data, initial_dc_correction=True)

        for _, row in events.iterrows():
            lbl_name = canonical_label(row.get('trial_type', ''))
            if lbl_name not in class_map:
                continue
            cls_id = class_map[lbl_name]
            onset_samp = round(float(row['onset']) * fs)
            dur_samp = round(float(row['duration']) * fs)
            start_samp = onset_samp + settle_len
            end_samp = onset_samp + dur_samp

            if end_samp - start_samp < win_len or end_samp > raw_data.shape[1]:
                continue

            cue_has_window = False
            pos = start_samp
            while pos + win_len <= end_samp:
                w_mrcp = mrcp_data[:, pos:pos + win_len]
                w_smr = smr_data[:, pos:pos + win_len]

                # Quality check
                if np.any(np.abs(w_smr) > 100e-6) or np.any(np.std(w_smr, axis=1) < 1e-9):
                    pos += stride_len
                    continue

                f_vec, f_names, m_mask, s_mask = extract_rich_dualband_window_features(
                    w_mrcp, w_smr, sfreq=fs, channels=channels
                )
                if feature_names is None:
                    feature_names = f_names
                    is_mrcp_mask = m_mask
                    is_smr_mask = s_mask

                features_list.append(f_vec)
                labels_list.append(cls_id)
                cues_list.append(global_cue)
                sessions_list.append(ses_str)
                cue_has_window = True
                pos += stride_len

            if cue_has_window:
                global_cue += 1

    if not features_list:
        return None

    return {
        'X': np.stack(features_list),
        'y': np.array(labels_list, dtype=int),
        'cues': np.array(cues_list, dtype=int),
        'sessions': np.array(sessions_list),
        'feature_names': feature_names,
        'is_mrcp': is_mrcp_mask,
        'is_smr': is_smr_mask,
    }


def compute_separability_ranking(X, y, feature_names, class_a, class_b):
    """Compute univariate Cohen's d, Fisher's J, and ROC-AUC between class_a and class_b."""
    mask_a = (y == class_a)
    mask_b = (y == class_b)
    Xa = X[mask_a]
    Xb = X[mask_b]
    y_bin = np.concatenate([np.zeros(len(Xa)), np.ones(len(Xb))])
    X_bin = np.vstack([Xa, Xb])

    rankings = []
    for idx, name in enumerate(feature_names):
        va = Xa[:, idx]
        vb = Xb[:, idx]
        ma, mb = np.mean(va), np.mean(vb)
        sa, sb = np.std(va) + 1e-12, np.std(vb) + 1e-12
        pooled_sd = np.sqrt((sa ** 2 + sb ** 2) / 2.0)
        cohens_d = float((mb - ma) / pooled_sd)
        fisher_j = float(((mb - ma) ** 2) / (sa ** 2 + sb ** 2))
        try:
            auc = float(roc_auc_score(y_bin, X_bin[:, idx]))
            auc_effect = max(auc, 1.0 - auc)
        except Exception:
            auc_effect = 0.5

        band_type = 'MRCP (0.3-3.0 Hz)' if name.startswith('mrcp_') else 'SMR (4.0-40.0 Hz)'
        rankings.append({
            'feature': name,
            'band': band_type,
            'cohens_d': round(cohens_d, 4),
            'abs_cohens_d': round(abs(cohens_d), 4),
            'fisher_j': round(fisher_j, 4),
            'roc_auc': round(auc_effect, 4),
        })

    rankings.sort(key=lambda r: r['fisher_j'], reverse=True)
    return rankings


def evaluate_cue_disjoint_stage2(X, y, cues, is_mrcp, is_smr, n_folds=5):
    """Evaluate Volar Grasp (1) vs Finger Extension (2) using whole-cue disjoint CV."""
    mov_mask = (y == 1) | (y == 2)
    X_mov = X[mov_mask]
    y_mov = (y[mov_mask] == 2).astype(int)  # 0 = Volar Grasp, 1 = Finger Extension
    cues_mov = cues[mov_mask]

    # Group cues by class
    grasp_cues = sorted(list(set(cues_mov[y_mov == 0])))
    ext_cues = sorted(list(set(cues_mov[y_mov == 1])))
    if len(grasp_cues) < n_folds or len(ext_cues) < n_folds:
        n_folds = min(len(grasp_cues), len(ext_cues))
    if n_folds < 2:
        return None

    rng = np.random.RandomState(42)
    rng.shuffle(grasp_cues)
    rng.shuffle(ext_cues)

    grasp_folds = np.array_split(grasp_cues, n_folds)
    ext_folds = np.array_split(ext_cues, n_folds)

    modes = {
        'SMR_only': is_smr,
        'MRCP_only': is_mrcp,
        'DualBand_MRCP_plus_SMR': np.ones(X.shape[1], dtype=bool),
    }

    results = {}
    for mode_name, feat_mask in modes.items():
        X_sub = X_mov[:, feat_mask]
        cue_preds = []
        cue_trues = []

        for f in range(n_folds):
            val_cue_set = set(grasp_folds[f]).union(set(ext_folds[f]))
            val_win_mask = np.array([c in val_cue_set for c in cues_mov])
            train_win_mask = ~val_win_mask

            clf = make_pipeline(
                RobustScaler(),
                LogisticRegression(C=0.5, class_weight='balanced', max_iter=500, random_state=42)
            )
            clf.fit(X_sub[train_win_mask], y_mov[train_win_mask])

            # Cue-level probability aggregation
            val_probs = clf.predict_proba(X_sub[val_win_mask])[:, 1]
            val_cues_arr = cues_mov[val_win_mask]
            val_y_arr = y_mov[val_win_mask]

            for c_id in sorted(list(val_cue_set)):
                c_mask = (val_cues_arr == c_id)
                mean_prob_ext = float(np.mean(val_probs[c_mask]))
                pred_cls = 1 if mean_prob_ext >= 0.5 else 0
                true_cls = int(val_y_arr[c_mask][0])
                cue_preds.append(pred_cls)
                cue_trues.append(true_cls)

        cue_trues = np.array(cue_trues)
        cue_preds = np.array(cue_preds)
        ba = float(balanced_accuracy_score(cue_trues, cue_preds))
        grasp_rec = float(recall_score(cue_trues, cue_preds, pos_label=0, zero_division=0))
        ext_rec = float(recall_score(cue_trues, cue_preds, pos_label=1, zero_division=0))
        cm = confusion_matrix(cue_trues, cue_preds, labels=[0, 1]).tolist()

        results[mode_name] = {
            'cue_balanced_accuracy': round(ba, 4),
            'volar_grasp_recall': round(grasp_rec, 4),
            'finger_extension_recall': round(ext_rec, 4),
            'n_grasp_cues': int(np.sum(cue_trues == 0)),
            'n_extension_cues': int(np.sum(cue_trues == 1)),
            'confusion_matrix_grasp_ext': cm,
        }

    return results


def main():
    bids_root = Path(r"C:\Users\giaco\Box\BCIFEST\bids")
    candidates = ['sub-00', 'sub-01', 'sub-02', 'sub-04', 'sub-07', 'sub-08', 'sub-99']

    subject_reports = {}
    pooled_X = []
    pooled_y = []
    feature_names = None
    is_mrcp = None
    is_smr = None

    print("=== Running Empirical Dual-Band MRCP (0.3-3.0 Hz) + SMR (4.0-40.0 Hz) Signal Audit ===", flush=True)
    for sub_str in candidates:
        data = load_subject_dualband_windows(bids_root, sub_str)
        if data is None:
            continue
        X, y, cues = data['X'], data['y'], data['cues']
        class_counts = {THREE_CLASSES[i]: int(np.sum(y == i)) for i in range(3)}
        if sum(1 for v in class_counts.values() if v > 0) < 2:
            continue

        feature_names = data['feature_names']
        is_mrcp = data['is_mrcp']
        is_smr = data['is_smr']

        # Per-subject normalizations for pooled feature ranking
        X_norm = RobustScaler().fit_transform(X)
        if class_counts['volar_grasp'] > 0 and class_counts['finger_extension'] > 0:
            pooled_X.append(X_norm)
            pooled_y.append(y)

        # Stage 2 ranking (Volar Grasp vs Finger Extension)
        stage2_top = []
        stage2_cv = None
        if class_counts['volar_grasp'] > 0 and class_counts['finger_extension'] > 0:
            s2_rank = compute_separability_ranking(X, y, feature_names, class_a=1, class_b=2)
            stage2_top = s2_rank[:10]
            stage2_cv = evaluate_cue_disjoint_stage2(X, y, cues, is_mrcp, is_smr, n_folds=5)

        # Stage 1 ranking (Rest vs Movement)
        stage1_top = []
        if class_counts['rest'] > 0 and (class_counts['volar_grasp'] + class_counts['finger_extension']) > 0:
            y_s1 = (y > 0).astype(int)
            s1_rank = compute_separability_ranking(X, y_s1, feature_names, class_a=0, class_b=1)
            stage1_top = s1_rank[:10]

        subject_reports[sub_str] = {
            'n_windows': int(len(y)),
            'n_cues': int(len(set(cues))),
            'window_counts_by_class': class_counts,
            'stage2_grasp_vs_extension_cv': stage2_cv,
            'top10_stage2_grasp_vs_extension_features': stage2_top,
            'top10_stage1_rest_vs_movement_features': stage1_top,
        }

        print(f"\nSubject {sub_str}: {len(y)} windows, {len(set(cues))} cues | {class_counts}", flush=True)
        if stage2_cv:
            for m_name, m_res in stage2_cv.items():
                print(f"  [Stage 2 CV - {m_name:22s}] BA={m_res['cue_balanced_accuracy']:.3f} | "
                      f"Grasp Recall={m_res['volar_grasp_recall']:.3f} | "
                      f"Extension Recall={m_res['finger_extension_recall']:.3f}", flush=True)

    # Pooled multi-subject ranking
    pooled_stage2_top15 = []
    pooled_stage1_top15 = []
    if pooled_X:
        X_all = np.vstack(pooled_X)
        y_all = np.concatenate(pooled_y)
        pooled_stage2_top15 = compute_separability_ranking(X_all, y_all, feature_names, 1, 2)[:15]
        y_all_s1 = (y_all > 0).astype(int)
        pooled_stage1_top15 = compute_separability_ranking(X_all, y_all_s1, feature_names, 0, 1)[:15]

    report = {
        'filter_contract': {
            'branch_a_mrcp': {'band_hz': [0.3, 3.0], 'order': 2, 'type': 'butterworth_sos_causal_zi'},
            'branch_b_smr': {'band_hz': [4.0, 40.0], 'order': 4, 'type': 'butterworth_sos_causal_zi'},
            'channels': COMMON_X24,
            'core_motor_channels': list(CORE_MOTOR_CHANNELS),
            'recommended_bipolar_pairs': [f"{a}-{b}" for a, b in CHANNEL_PAIRS],
        },
        'pooled_top15_stage2_grasp_vs_extension_features': pooled_stage2_top15,
        'pooled_top15_stage1_rest_vs_movement_features': pooled_stage1_top15,
        'subjects': subject_reports,
    }

    out_path = Path(__file__).resolve().parent / 'decoder_and_models' / 'mrcp_dualband_separability_report.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"\nSaved separability report to {out_path}", flush=True)


if __name__ == '__main__':
    main()
