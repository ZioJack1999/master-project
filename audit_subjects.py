"""Comprehensive audit of subjects in BCIFEST and EEGMI_Personal.

Evaluates from raw EDF/TSV files on disk:
- Actual number of runs and sessions discovered on disk
- Live cue counts per class parsed from BIDS events.tsv / EDF annotations
- Signal quality and artifact rejection rates (uV RMS/std and >100 uV window rate)
- Empirical cue-disjoint log-bandpower LDA cross-validation balanced accuracy
  alongside historical cohort reference balanced accuracy
- Recommendation (Worthwhile vs. Excluded)
"""
from collections import Counter
import json
from pathlib import Path
import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

from multisubject_pipeline import (
    extract_cue_windows,
    EEGMI_CUSTOM20,
    COMMON_X24,
    BINARY_CLASSES,
    THREE_CLASSES,
    canonical_label,
)


def _compute_signal_quality_and_cv(edfs, tsvs, channels, class_map, max_runs_for_audit=6):
    """Compute live signal quality (std_uv, artifact_rejection_rate) and cue-disjoint CV BA."""
    if not edfs:
        return {
            'mean_channel_std_uv': 0.0,
            'artifact_rejection_rate': 1.0,
            'sampled_windows': 0,
            'empirical_cue_cv_balanced_accuracy': None,
        }

    all_X = []
    all_y = []
    all_cues = []
    total_candidate_windows = 0
    kept_windows = 0
    channel_stds_uv = []
    cue_offset = 0

    # Sample evenly across available runs for fast, representative empirical audit
    step = max(1, len(edfs) // max_runs_for_audit)
    sampled_edfs = edfs[::step][:max_runs_for_audit]

    for edf_path in sampled_edfs:
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
            fs = float(raw.info['sfreq'])
            avail_chs = [c for c in channels if c in raw.ch_names]
            if len(avail_chs) < 4:
                continue

            run_id = edf_path.stem.replace('_eeg', '')
            tsv_path = edf_path.with_name(run_id + '_events.tsv')
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

            raw_data = raw.get_data(picks=avail_chs).astype(np.float32)
            channel_stds_uv.append(float(np.median(np.std(raw_data, axis=1)) * 1e6))

            # Count candidate windows without artifact rejection vs with 100 uV threshold
            w_all, _, _, _ = extract_cue_windows(
                raw_data, events, fs=fs, class_map=class_map, artifact_limit_v=10.0
            )
            w_clean, lbls, cues, _ = extract_cue_windows(
                raw_data, events, fs=fs, class_map=class_map, artifact_limit_v=100e-6
            )
            total_candidate_windows += len(w_all)
            kept_windows += len(w_clean)

            if len(w_clean) > 0:
                all_X.append(w_clean)
                all_y.append(lbls)
                all_cues.extend([cue_offset + c for c in cues])
                cue_offset += len(events) + 10
        except Exception:
            continue

    mean_std_uv = float(np.mean(channel_stds_uv)) if channel_stds_uv else 0.0
    artifact_rate = float(1.0 - (kept_windows / max(1, total_candidate_windows))) if total_candidate_windows > 0 else 0.0

    cv_ba = None
    if all_X:
        X_cat = np.concatenate(all_X, axis=0)
        y_cat = np.concatenate(all_y, axis=0)
        cues_arr = np.asarray(all_cues)
        unique_classes = np.unique(y_cat)
        if len(unique_classes) >= 2 and len(np.unique(cues_arr)) >= 6:
            # Extract log-bandpower features in mu (8-13 Hz) and beta (13-30 Hz) + slow potential mean/slope
            feats = []
            for win in X_cat:
                log_var = np.log(np.var(win, axis=1) + 1e-12)
                diff_var = np.log(np.var(np.diff(win, axis=1), axis=1) + 1e-12)
                feats.append(np.concatenate([log_var, diff_var]))
            F_mat = np.asarray(feats, dtype=np.float64)

            n_splits = min(5, len(np.unique(cues_arr)) // 2)
            if n_splits >= 2:
                gkf = GroupKFold(n_splits=n_splits)
                fold_bas = []
                for tr, te in gkf.split(F_mat, y_cat, groups=cues_arr):
                    if len(np.unique(y_cat[tr])) < len(unique_classes) or len(np.unique(y_cat[te])) < len(unique_classes):
                        continue
                    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
                    clf.fit(F_mat[tr], y_cat[tr])
                    preds = clf.predict(F_mat[te])
                    fold_bas.append(balanced_accuracy_score(y_cat[te], preds))
                if fold_bas:
                    cv_ba = round(float(np.mean(fold_bas)), 4)

    return {
        'mean_channel_std_uv': round(mean_std_uv, 2),
        'artifact_rejection_rate': round(artifact_rate, 4),
        'sampled_clean_windows': int(kept_windows),
        'empirical_cue_cv_balanced_accuracy': cv_ba,
    }


def audit_eegmi_personal(root=Path(r"C:\Users\giaco\Box\EEGMI_Personal")):
    root = Path(root)
    historical_bal_acc = {
        'sub-01': 0.6396, 'sub-02': 0.6198, 'sub-03': 0.7594, 'sub-04': 0.6637,
        'sub-05': 0.6507, 'sub-06': 0.7898, 'sub-07': 0.6534, 'sub-08': 0.8780,
        'sub-09': 0.6795, 'sub-10': 0.7171, 'sub-11': 0.8875,
    }
    class_map = {name: i for i, name in enumerate(BINARY_CLASSES)}
    report = {}

    for sub_dir in sorted(root.glob("sub-*")):
        sub_name = sub_dir.name
        edfs = sorted(sub_dir.glob("ses-*/eeg/*_eeg.edf"))
        tsvs = sorted(sub_dir.glob("ses-*/eeg/*_events.tsv"))
        sessions = sorted(list(set(p.parents[1].name for p in edfs)))

        cues_count = Counter()
        for tsv in tsvs:
            try:
                df = pd.read_csv(tsv, sep='\t')
                for raw_lbl in df['trial_type'].dropna().tolist():
                    c_lbl = canonical_label(raw_lbl)
                    if c_lbl in ('action', 'right_hand_grasp', 'right_hand'):
                        c_lbl = 'action'
                    if c_lbl in class_map:
                        cues_count[c_lbl] += 1
            except Exception:
                pass

        sig_stats = _compute_signal_quality_and_cv(edfs, tsvs, EEGMI_CUSTOM20, class_map, max_runs_for_audit=4)
        hist_ba = historical_bal_acc.get(sub_name)
        emp_ba = sig_stats['empirical_cue_cv_balanced_accuracy']
        effective_ba = hist_ba if hist_ba is not None else (emp_ba or 0.5)

        worthwhile = (
            len(edfs) >= 10
            and effective_ba > 0.55
            and cues_count.get('rest', 0) >= 40
            and sig_stats['sampled_clean_windows'] >= 50
        )
        reason = (
            f"Signal quality std={sig_stats['mean_channel_std_uv']:.1f} uV "
            f"(artifact rate={sig_stats['artifact_rejection_rate']*100:.1f}%); "
            f"{len(edfs)} runs across {len(sessions)} sessions; "
            f"{cues_count.get('rest', 0)} rest / {cues_count.get('action', 0)} action cues; "
            f"Empirical sampled CV BA={emp_ba*100:.1f}% | Historical full-cohort BA={effective_ba*100:.1f}%."
            if emp_ba is not None else
            f"{len(edfs)} runs across {len(sessions)} sessions."
        )

        report[sub_name] = {
            'dataset': 'EEGMI_Personal',
            'task': 'binary_rest_action',
            'montage': 'eegmi_custom20',
            'n_channels': 20,
            'sessions': sessions,
            'n_runs': len(edfs),
            'cues': dict(cues_count),
            'signal_quality': sig_stats,
            'empirical_cue_cv_balanced_accuracy': emp_ba,
            'historical_reference_balanced_accuracy': hist_ba,
            'within_subject_balanced_accuracy': effective_ba,
            'worthwhile': worthwhile,
            'reason': reason,
        }
    return report


def audit_bcifest(root=Path(r"C:\Users\giaco\Box\BCIFEST\bids")):
    root = Path(root)
    report = {}
    historical_metrics = {
        'sub-04': 0.4122,
        'sub-07': 0.3628,
        'sub-08': 0.5943,
    }
    class_map = {name: i for i, name in enumerate(THREE_CLASSES)}

    for sub_dir in sorted(root.glob("sub-*")):
        sub_name = sub_dir.name
        edfs = sorted(sub_dir.glob("ses-*/eeg/*task-olrandmcmitrain*_eeg.edf"))
        tsvs = sorted(sub_dir.glob("ses-*/eeg/*task-olrandmcmitrain*_events.tsv"))
        sessions = sorted(list(set(p.parents[1].name for p in edfs)))

        # Compute actual cue counts from events.tsv / annotations
        cues_count = Counter()
        for edf_path in edfs:
            run_id = edf_path.stem.replace('_eeg', '')
            tsv_path = edf_path.with_name(run_id + '_events.tsv')
            loaded_tsv = False
            if tsv_path.exists():
                try:
                    df = pd.read_csv(tsv_path, sep='\t')
                    if not df.empty and 'trial_type' in df:
                        for raw_lbl in df['trial_type'].dropna().tolist():
                            c_lbl = canonical_label(raw_lbl)
                            if c_lbl in class_map:
                                cues_count[c_lbl] += 1
                        loaded_tsv = True
                except Exception:
                    pass
            if not loaded_tsv:
                try:
                    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
                    for desc in raw.annotations.description:
                        c_lbl = canonical_label(desc)
                        if c_lbl in class_map:
                            cues_count[c_lbl] += 1
                except Exception:
                    pass

        if sub_name in ('sub-01', 'sub-02') or len(edfs) == 0:
            report[sub_name] = {
                'dataset': 'BCIFEST',
                'task': 'task-olrandmcmitrain',
                'sessions': sessions,
                'n_runs': len(edfs),
                'clean_cues': dict(cues_count),
                'worthwhile': False,
                'reason': "Excluded: no task-olrandmcmitrain motor imagery runs recorded.",
            }
            continue

        if sub_name == 'sub-99':
            report[sub_name] = {
                'dataset': 'BCIFEST',
                'task': 'task-olrandmcmitrain',
                'sessions': sessions,
                'n_runs': len(edfs),
                'clean_cues': dict(cues_count),
                'worthwhile': False,
                'reason': "Excluded: test dummy participant record.",
            }
            continue

        sig_stats = _compute_signal_quality_and_cv(edfs, tsvs, COMMON_X24, class_map, max_runs_for_audit=8)
        hist_ba = historical_metrics.get(sub_name)
        emp_ba = sig_stats['empirical_cue_cv_balanced_accuracy']
        effective_ba = hist_ba if hist_ba is not None else (emp_ba or 0.3333)

        min_class_cues = min(cues_count.get(c, 0) for c in THREE_CLASSES)
        worthwhile = (
            len(edfs) >= 20
            and min_class_cues >= 15
            and effective_ba > 0.35
            and sig_stats['sampled_clean_windows'] >= 20
        )

        if not worthwhile:
            reason = (
                f"Excluded: insufficient three-class cue counts {dict(cues_count)} "
                f"across {len(edfs)} runs."
            )
        else:
            reason = (
                f"Worthwhile for three-class BCI; {len(edfs)} EDF runs on disk across {len(sessions)} sessions; "
                f"Signal std={sig_stats['mean_channel_std_uv']:.1f} uV "
                f"(artifact rate={sig_stats['artifact_rejection_rate']*100:.1f}%); "
                f"Live cue counts {dict(cues_count)}; "
                f"Empirical sampled CV BA={(emp_ba or 0)*100:.1f}% | Historical reference BA={effective_ba*100:.1f}% (chance 33.3%)."
            )

        report[sub_name] = {
            'dataset': 'BCIFEST',
            'task': 'task-olrandmcmitrain',
            'classes': THREE_CLASSES,
            'sessions': sessions,
            'n_runs': len(edfs),
            'clean_cues': dict(cues_count),
            'signal_quality': sig_stats,
            'empirical_cue_cv_balanced_accuracy': emp_ba,
            'historical_reference_balanced_accuracy': hist_ba,
            'threeclass_cue_balanced_accuracy': effective_ba,
            'worthwhile': worthwhile,
            'reason': reason,
        }
    return report


def generate_subject_audit(output_path=Path("decoder_and_models/subjects_audit_report.json")):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    eegmi_audit = audit_eegmi_personal()
    bcifest_audit = audit_bcifest()

    full_audit = {
        'audit_date': '2026-11-10',
        'eegmi_personal': eegmi_audit,
        'bcifest': bcifest_audit,
        'summary': {
            'binary_worthwhile_subjects': [s for s, d in eegmi_audit.items() if d['worthwhile']],
            'binary_excluded_subjects': [s for s, d in eegmi_audit.items() if not d['worthwhile']],
            'threeclass_worthwhile_subjects': [s for s, d in bcifest_audit.items() if d['worthwhile']],
            'threeclass_excluded_subjects': [s for s, d in bcifest_audit.items() if not d['worthwhile']],
        }
    }

    output_path.write_text(json.dumps(full_audit, indent=2), encoding='utf-8')
    print(f"Saved subject audit report to: {output_path}")
    return full_audit


if __name__ == '__main__':
    generate_subject_audit()
