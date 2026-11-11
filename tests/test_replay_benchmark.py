"""Unit and Integration Tests for X24 Replay Benchmark Harness."""

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from replay_benchmark import (
    locate_x24_recordings,
    compute_clinical_metrics,
    ReplayBenchmarkHarness,
    CLASS_NAMES,
)
from hierarchical_classifier import UnsupervisedRestingAlignment
from multisubject_pipeline import (
    DualBandTemporalEEGNet,
    HierarchicalTwoStageDecoder,
    COMMON_X24,
)
from temporal_networks import TemporalEEGNet


def test_locate_x24_recordings():
    """Verify recording locator finds structured records."""
    recs = locate_x24_recordings(workspace_dir=".")
    assert isinstance(recs, list)
    for r in recs:
        assert 'type' in r
        assert 'path' in r


def test_clinical_metrics_computation():
    """Verify accurate calculation of clinical BCI metrics with synthetic ground truth."""
    # 0 = rest, 1 = volar_grasp, 2 = finger_extension
    # 30 cues / windows: 10 of each
    y_true = np.array([0]*10 + [1]*10 + [2]*10)
    # Perfect predictions
    y_pred = np.array([0]*10 + [1]*10 + [2]*10)
    cue_ids = list(range(30))

    metrics = compute_clinical_metrics(y_true, y_pred, cue_ids=cue_ids, window_stride_s=0.5)
    assert metrics.balanced_accuracy == 1.0
    assert metrics.finger_extension_recall == 1.0
    assert metrics.volar_grasp_recall == 1.0
    assert metrics.rest_recall == 1.0
    assert metrics.rest_false_activation_rate_per_min == 0.0
    assert metrics.abstention_rate == 0.0
    assert metrics.n_abstain_windows == 0

    # Test with Extension collapse (0 extension correct)
    y_pred_collapsed = np.array([0]*10 + [1]*10 + [1]*10)  # extensions misclassified as grasp
    m_col = compute_clinical_metrics(y_true, y_pred_collapsed, cue_ids=cue_ids, window_stride_s=0.5)
    assert m_col.finger_extension_recall == 0.0
    assert m_col.volar_grasp_recall == 1.0
    assert m_col.rest_recall == 1.0
    assert np.isclose(m_col.balanced_accuracy, 2.0 / 3.0)

    # Test with Rest false activations
    # 2 false activations in 10 rest windows (5 seconds = 5/60 min = 0.0833 min)
    # FAR = 2 / (5/60) = 24.0 per min
    y_pred_fa = np.array([0]*8 + [1, 2] + [1]*10 + [2]*10)
    m_fa = compute_clinical_metrics(y_true, y_pred_fa, cue_ids=cue_ids, window_stride_s=0.5)
    assert m_fa.false_activation_count == 2
    assert np.isclose(m_fa.rest_false_activation_rate_per_min, 24.0)

    # Test with Abstentions (-1)
    y_pred_abs = np.array([0]*10 + [1]*8 + [-1, -1] + [2]*10)
    m_abs = compute_clinical_metrics(y_true, y_pred_abs, cue_ids=cue_ids, window_stride_s=0.5)
    assert m_abs.n_abstain_windows == 2
    assert np.isclose(m_abs.abstention_rate, 2.0 / 30.0)
    assert np.isclose(m_abs.volar_grasp_recall, 0.8)  # 8 / 10


def test_canonical_unsupervised_resting_alignment():
    """Verify canonical resting alignment matrix and covariance normalization."""
    rng = np.random.RandomState(42)
    n_ch = 12
    n_samples = 256 * 10
    raw_rest = rng.randn(n_ch, n_samples).astype(np.float32)

    ea = UnsupervisedRestingAlignment(shrinkage=0.05).fit_resting_eeg(raw_rest, fs=256.0)
    assert ea.alignment_matrix_ is not None
    assert ea.alignment_matrix_.shape == (n_ch, n_ch)

    aligned = ea.transform(raw_rest)
    assert aligned.shape == (n_ch, n_samples)
    # Check transformed covariance is well-conditioned
    cov = (aligned @ aligned.T) / (n_samples - 1)
    diag = np.diag(cov)
    assert np.all(diag > 0.5) and np.all(diag < 2.0)


def test_replay_harness_synthetic():
    """Verify replay harness on synthetic models and test windows with per-band URA."""
    n_channels = len(COMMON_X24)
    n_times = 512
    n_windows = 15

    # Create dummy models
    base_model = TemporalEEGNet(n_channels=n_channels, n_classes=3, window_standardization=True)
    s1_model = TemporalEEGNet(n_channels=n_channels, n_classes=2, window_standardization=True)
    s2_model = DualBandTemporalEEGNet(n_channels=n_channels, n_classes=2, window_standardization=True)

    hier_decoder = HierarchicalTwoStageDecoder(
        stage1_model=s1_model,
        stage2_model=s2_model,
        rest_threshold=0.5,
        min_action_margin=0.05
    )

    X_smr = np.random.randn(n_windows, n_channels, n_times).astype(np.float32) * 10e-6
    X_mb = np.random.randn(n_windows, 2 * n_channels, n_times).astype(np.float32) * 10e-6
    y = np.array([0, 1, 2] * 5)
    cues = list(range(n_windows))

    # Fit per-band URA on rest windows
    rest_mask = (y == 0)
    ura_mrcp = UnsupervisedRestingAlignment(shrinkage=0.1).fit_from_rest(X_mb[rest_mask, :n_channels, :])
    ura_smr = UnsupervisedRestingAlignment(shrinkage=0.1).fit_from_rest(X_smr[rest_mask])

    harness = ReplayBenchmarkHarness(channels=COMMON_X24, fs=256.0, device='cpu')
    res = harness.replay_dataset_windows(
        X_mb, X_smr, y, cues,
        baseline_model=base_model,
        hierarchical_decoder=hier_decoder,
        ura_mrcp=ura_mrcp,
        ura_smr=ura_smr
    )

    assert 'baseline' in res
    assert 'hierarchical' in res
    assert 'delta' in res
    assert 'balanced_accuracy' in res['hierarchical']['metrics']
    assert 'finger_extension_recall' in res['hierarchical']['metrics']
    assert 'volar_grasp_recall' in res['hierarchical']['metrics']
    assert 'rest_false_activation_rate_per_min' in res['hierarchical']['metrics']


def test_continuous_streaming_replay_synthetic(tmp_path):
    """Verify continuous chunk-by-chunk streaming replay through DualBandRingBuffer."""
    n_ch = len(COMMON_X24)
    sfreq = 256.0
    n_samples = int(sfreq * 6.0)  # 6 seconds

    rng = np.random.RandomState(42)
    raw = rng.randn(n_ch, n_samples).astype(np.float32) * 15e-6

    # Save temporary npz
    fake_npz = tmp_path / "eeg.npz"
    np.savez(fake_npz, raw=raw, sfreq=sfreq)
    fake_meta = tmp_path / "metadata.json"
    fake_meta.write_text(json.dumps({'channels': COMMON_X24}), encoding='utf-8')

    s1_model = TemporalEEGNet(n_channels=n_ch, n_classes=2, window_standardization=True)
    s2_model = DualBandTemporalEEGNet(n_channels=n_ch, n_classes=2, window_standardization=True)
    decoder = HierarchicalTwoStageDecoder(stage1_model=s1_model, stage2_model=s2_model, rest_threshold=0.5)

    harness = ReplayBenchmarkHarness(channels=COMMON_X24, fs=sfreq, device='cpu')
    res = harness.replay_continuous_x24_session(fake_npz, decoder, chunk_size=32, stride_samples=64)

    assert res['n_windows_evaluated'] > 0
    assert res['sum_to_one_calibrated'] is True
    assert res['mean_latency_ms'] >= 0.0
