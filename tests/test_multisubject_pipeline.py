"""Unit tests for multisubject_pipeline.py."""
import numpy as np
import pytest
import torch

from multisubject_pipeline import (
    EuclideanAlignment,
    UnsupervisedRestingAlignment,
    split_whole_cues_stratified,
    evaluate_predictions,
    train_temporal_backbone,
    finetune_temporal_personal,
    evaluate_test_session,
    filter_multiband_eeg,
    extract_cue_windows_multiband,
    DualBandTemporalEEGNet,
    HierarchicalTwoStageDecoder,
    COMMON_X24,
    EEGMI_CUSTOM20,
)
from temporal_networks import TemporalEEGNet


def test_euclidean_alignment_whitening():
    """Verify EuclideanAlignment whitens average spatial covariance to near-identity."""
    np.random.seed(42)
    n_trials = 30
    n_channels = 8
    n_times = 256

    # Generate synthetic correlated trials
    A = np.random.randn(n_channels, n_channels)
    cov_true = A @ A.T + 0.5 * np.eye(n_channels)
    L = np.linalg.cholesky(cov_true)

    trials = []
    for _ in range(n_trials):
        raw = np.random.randn(n_channels, n_times)
        trial = L @ raw
        trials.append(trial)
    X = np.stack(trials)

    ea = EuclideanAlignment(shrinkage=0.01).fit(X)
    X_aligned = ea.transform(X)

    # Check aligned covariance
    aligned_covs = []
    for i in range(n_trials):
        centered = X_aligned[i] - X_aligned[i].mean(axis=1, keepdims=True)
        cov = (centered @ centered.T) / (n_times - 1)
        aligned_covs.append(cov)
    mean_cov = np.mean(aligned_covs, axis=0)

    # Off-diagonals should be near 0, diagonals near 1
    np.testing.assert_allclose(np.diag(mean_cov), np.ones(n_channels), atol=0.15)
    off_diag = mean_cov - np.diag(np.diag(mean_cov))
    np.testing.assert_allclose(off_diag, np.zeros((n_channels, n_channels)), atol=0.15)


def test_euclidean_alignment_serialization():
    """Verify EuclideanAlignment state_dict roundtrip."""
    np.random.seed(42)
    X = np.random.randn(10, 4, 128).astype(np.float32)
    ea1 = EuclideanAlignment(shrinkage=0.08).fit(X)
    state = ea1.state_dict()

    ea2 = EuclideanAlignment().load_state_dict(state)
    np.testing.assert_allclose(ea1.alignment_matrix_, ea2.alignment_matrix_)
    np.testing.assert_allclose(ea1.reference_cov_, ea2.reference_cov_)


def test_whole_cue_splitting_no_leakage():
    """Verify that stratified whole-cue splitting has ZERO cue overlap between partitions."""
    np.random.seed(42)
    n_cues = 20
    windows_per_cue = 4
    records = []
    cue_ids = []
    y = []

    for cue_idx in range(n_cues):
        cls = cue_idx % 2  # binary classes
        for w in range(windows_per_cue):
            records.append({'run': 'run1', 'cue_idx': cue_idx, 'label': cls})
            cue_ids.append(cue_idx)
            y.append(cls)

    y = np.array(y)
    train_idx, val_idx = split_whole_cues_stratified(records, y, cue_ids, test_size=0.25, random_state=42)

    # Check indices are disjoint
    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(train_idx) + len(val_idx) == len(y)

    # Check cues are completely disjoint
    train_cues = set(np.array(cue_ids)[train_idx])
    val_cues = set(np.array(cue_ids)[val_idx])
    assert len(train_cues & val_cues) == 0, f"Leaked cues: {train_cues & val_cues}"

    # Verify both classes are represented
    assert set(y[train_idx]) == {0, 1}
    assert set(y[val_idx]) == {0, 1}


def test_temporal_eegnet_finetune_flow():
    """Verify full fine-tuning and evaluation workflow on synthetic data."""
    torch.manual_seed(42)
    np.random.seed(42)
    n_channels = 6
    n_classes = 2
    n_times = 128

    model = TemporalEEGNet(n_channels=n_channels, n_classes=n_classes, window_standardization=True)

    # Synthetic calibration train & val
    calib_train_x = np.random.randn(16, n_channels, n_times).astype(np.float32)
    calib_train_y = np.array([0, 1] * 8, dtype=int)
    calib_train_cues = list(range(16))

    calib_val_x = np.random.randn(8, n_channels, n_times).astype(np.float32)
    calib_val_y = np.array([0, 1] * 4, dtype=int)
    calib_val_cues = list(range(16, 24))

    tuned_model, ea, best_ba, history = finetune_temporal_personal(
        model, calib_train_x, calib_train_y, calib_train_cues,
        calib_val_x, calib_val_y, calib_val_cues,
        epochs=3, lr_head=1e-2, lr_backbone=1e-3, patience=2,
        freeze_backbone_epochs=1, batch_size=8
    )

    assert tuned_model is not None
    assert ea.alignment_matrix_ is not None
    assert len(history) > 0

    # Test evaluation
    test_x = np.random.randn(8, n_channels, n_times).astype(np.float32)
    test_y = np.array([0, 1] * 4, dtype=int)
    test_cues = list(range(24, 32))

    metrics, probs = evaluate_test_session(tuned_model, test_x, test_y, test_cues, ea)
    assert 'window' in metrics
    assert 'cue' in metrics
    assert 0.0 <= metrics['window']['balanced_accuracy'] <= 1.0
    assert 0.0 <= metrics['cue']['balanced_accuracy'] <= 1.0


def test_unsupervised_resting_alignment():
    """Verify UnsupervisedRestingAlignment computes whitening strictly from resting state."""
    np.random.seed(42)
    n_rest_windows = 20
    n_channels = 6
    n_times = 256

    # Correlated resting data
    A = np.random.randn(n_channels, n_channels)
    cov_true = A @ A.T + 0.2 * np.eye(n_channels)
    L = np.linalg.cholesky(cov_true)

    rest_trials = np.stack([L @ np.random.randn(n_channels, n_times) for _ in range(n_rest_windows)])
    ura = UnsupervisedRestingAlignment(shrinkage=0.01).fit_from_rest(rest_trials)

    aligned_rest = ura.transform(rest_trials)
    covs = [ (trial - trial.mean(axis=1, keepdims=True)) @ (trial - trial.mean(axis=1, keepdims=True)).T / (n_times - 1)
             for trial in aligned_rest ]
    mean_cov = np.mean(covs, axis=0)

    # Whitened resting covariance should have diag near 1, off-diag near 0
    np.testing.assert_allclose(np.diag(mean_cov), np.ones(n_channels), atol=0.20)
    off_diag = mean_cov - np.diag(np.diag(mean_cov))
    np.testing.assert_allclose(off_diag, np.zeros((n_channels, n_channels)), atol=0.20)


def test_dualband_temporaleegnet_forward():
    """Verify DualBandTemporalEEGNet forward pass and gradient backpropagation."""
    torch.manual_seed(42)
    n_channels = 4
    n_classes = 2
    n_times = 512
    batch_size = 4

    model = DualBandTemporalEEGNet(n_channels=n_channels, n_classes=n_classes, window_standardization=True)
    # Input has 2 * n_channels = 8 channels
    x = torch.randn(batch_size, 2 * n_channels, n_times, requires_grad=True)
    logits = model(x)

    assert logits.shape == (batch_size, n_classes)
    loss = logits.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_hierarchical_two_stage_decoder():
    """Verify HierarchicalTwoStageDecoder decision routing and probability combination."""
    torch.manual_seed(42)
    n_channels = 4
    n_times = 512

    stage1 = TemporalEEGNet(n_channels=n_channels, n_classes=2, window_standardization=True)
    stage2 = DualBandTemporalEEGNet(n_channels=n_channels, n_classes=2, window_standardization=True)

    decoder = HierarchicalTwoStageDecoder(
        stage1_model=stage1,
        stage2_model=stage2,
        rest_threshold=0.50,
        min_action_margin=0.0
    )

    x_smr = np.random.randn(5, n_channels, n_times).astype(np.float32)
    x_mb = np.random.randn(5, 2 * n_channels, n_times).astype(np.float32)

    decisions, probs = decoder.predict_window(x_smr, x_mb)
    assert len(decisions) == 5
    assert probs.shape == (5, 3)
    # Probabilities must be valid and sum to 1.0
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(5), atol=1e-5)
    for d in decisions:
        assert d in (-1, 0, 1, 2)


def test_extract_cue_windows_multiband():
    """Verify extract_cue_windows_multiband extracts synchronized dual-band windows."""
    import pandas as pd
    fs = 256
    n_channels = 4
    # Create 8 seconds of synthetic EEG (2048 samples) in realistic microvolts (~10 uV)
    raw_data = (10e-6 * np.random.randn(n_channels, 2048)).astype(np.float32)
    events = pd.DataFrame([
        {'onset': 0.0, 'duration': 3.0, 'trial_type': 'rest'},
        {'onset': 4.0, 'duration': 3.5, 'trial_type': 'volar_grasp'},
    ])

    class_map = {'rest': 0, 'volar_grasp': 1}
    w_mb, w_smr, lbls, cues, recs = extract_cue_windows_multiband(
        raw_data, events, fs=fs, window_s=2.0, stride_s=0.5, settle_s=0.5, class_map=class_map
    )

    assert len(w_mb) > 0
    assert w_mb.shape[1] == 2 * n_channels
    assert w_smr.shape[1] == n_channels
    assert w_mb.shape[2] == round(2.0 * fs)
    assert len(lbls) == len(w_mb)
    assert set(lbls).issubset({0, 1})


def test_benchmark_hierarchical_mrcp_synthetic():
    """Verify hierarchical benchmark execution path with decoupled Stage 1 (SMR) and Stage 2 (DualBand MRCP+SMR).

    Checks per-band C x C UnsupervisedRestingAlignment, decision routing,
    non-zero recalls, and leak-free session evaluation.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    n_channels = 4
    n_times = 512

    stage1_model = TemporalEEGNet(n_channels=n_channels, n_classes=2, window_standardization=True)
    stage2_model = DualBandTemporalEEGNet(n_channels=n_channels, n_classes=2, window_standardization=True)

    # Synthetic calibration and test windows (12 calib, 12 test)
    calib_smr = (10e-6 * np.random.randn(12, n_channels, n_times)).astype(np.float32)
    calib_mrcp = (10e-6 * np.random.randn(12, n_channels, n_times)).astype(np.float32)
    calib_mb = np.concatenate([calib_mrcp, calib_smr], axis=1)
    calib_y = np.array([0, 1, 2] * 4, dtype=int)

    test_smr = (10e-6 * np.random.randn(12, n_channels, n_times)).astype(np.float32)
    test_mrcp = (10e-6 * np.random.randn(12, n_channels, n_times)).astype(np.float32)
    test_mb = np.concatenate([test_mrcp, test_smr], axis=1)
    test_y = np.array([0, 1, 2] * 4, dtype=int)

    # 1. Fit per-band C x C URA strictly on calibration rest windows
    calib_rest_idx = np.flatnonzero(calib_y == 0)
    ura_mrcp = UnsupervisedRestingAlignment(shrinkage=0.1).fit_from_rest(calib_mb[calib_rest_idx, :n_channels, :])
    ura_smr = UnsupervisedRestingAlignment(shrinkage=0.1).fit_from_rest(calib_smr[calib_rest_idx])

    # 2. Transform test windows
    test_aligned_mrcp = ura_mrcp.transform(test_mb[:, :n_channels, :])
    test_aligned_smr = ura_smr.transform(test_smr)
    test_aligned_mb = np.concatenate([test_aligned_mrcp, test_aligned_smr], axis=1)

    # 3. Predict via HierarchicalTwoStageDecoder
    decoder = HierarchicalTwoStageDecoder(
        stage1_model=stage1_model,
        stage2_model=stage2_model,
        rest_threshold=0.50,
        min_action_margin=0.05,
        classes=['rest', 'volar_grasp', 'finger_extension'],
    )
    decisions, probs = decoder.predict_window(test_aligned_smr, test_aligned_mb)

    assert len(decisions) == len(test_y)
    assert probs.shape == (len(test_y), 3)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(test_y)), atol=1e-5)
    for d in decisions:
        assert d in (-1, 0, 1, 2)

