"""Unit tests for hierarchical_classifier.py (Two-Stage Classifier & Unsupervised Resting Alignment)."""

import numpy as np
import pytest
import torch

from hierarchical_classifier import (
    UnsupervisedRestingAlignment,
    Stage1GatingNetwork,
    Stage2DirectionalNetwork,
    HierarchicalTwoStageEEGNet,
    HierarchicalEnsemble,
    compute_hierarchical_metrics,
    train_hierarchical_two_stage,
    finetune_hierarchical_personal,
    THREE_CLASSES,
)


def test_unsupervised_resting_alignment_whitening():
    """Verify that UnsupervisedRestingAlignment whitens resting covariance to identity."""
    np.random.seed(42)
    n_channels = 6
    n_times = 512
    n_epochs = 20

    # Synthetic resting EEG with correlated spatial covariance
    A = np.random.randn(n_channels, n_channels)
    cov_true = A @ A.T + 0.3 * np.eye(n_channels)
    L = np.linalg.cholesky(cov_true)

    resting_epochs = []
    for _ in range(n_epochs):
        white = np.random.randn(n_channels, n_times)
        resting_epochs.append(L @ white)
    X_rest = np.stack(resting_epochs)

    ura = UnsupervisedRestingAlignment(shrinkage=0.01).fit(X_rest)
    aligned_rest = ura.transform(X_rest)

    # Compute covariance of aligned resting data
    aligned_covs = []
    for i in range(n_epochs):
        c = aligned_rest[i] - aligned_rest[i].mean(axis=1, keepdims=True)
        cov = (c @ c.T) / (n_times - 1)
        aligned_covs.append(cov)
    mean_aligned_cov = np.mean(aligned_covs, axis=0)

    # Diagonal should be ~ 1.0, off-diagonal ~ 0.0
    diag = np.diag(mean_aligned_cov)
    off_diag = mean_aligned_cov - np.diag(diag)
    np.testing.assert_allclose(diag, np.ones(n_channels), atol=0.20)
    np.testing.assert_allclose(off_diag, np.zeros((n_channels, n_channels)), atol=0.20)


def test_unsupervised_resting_alignment_serialization_and_online_update():
    """Verify serialization roundtrip and online EMA covariance adaptation."""
    np.random.seed(42)
    X_rest = np.random.randn(8, 4, 256)
    ura = UnsupervisedRestingAlignment(shrinkage=0.05).fit(X_rest)
    state = ura.state_dict()

    ura_restored = UnsupervisedRestingAlignment().load_state_dict(state)
    np.testing.assert_allclose(ura.alignment_matrix_, ura_restored.alignment_matrix_)
    np.testing.assert_allclose(ura.reference_cov_, ura_restored.reference_cov_)

    # Online update with new resting chunk
    new_chunk = np.random.randn(4, 128)
    W_before = ura.alignment_matrix_.copy()
    ura.update_online(new_chunk, alpha_ema=0.1)
    assert not np.allclose(W_before, ura.alignment_matrix_)
    assert ura.alignment_matrix_.shape == (4, 4)


def test_unsupervised_resting_alignment_fit_from_dataset():
    """Verify fit_from_dataset extracts rest trials (y == 0)."""
    np.random.seed(42)
    X = np.random.randn(30, 4, 128)
    y = np.array([0, 1, 2] * 10)  # rest, grasp, extension

    ura = UnsupervisedRestingAlignment().fit_from_dataset(X, y, rest_label=0)
    assert ura.n_samples_fitted_ == 10 * 128
    aligned = ura.transform(X)
    assert aligned.shape == X.shape


def test_stage1_gating_network_forward():
    """Verify Stage 1 Gating Network outputs 2 logits for Rest vs Movement."""
    torch.manual_seed(42)
    n_ch = 6
    n_times = 256
    batch = torch.randn(4, n_ch, n_times)

    gating_net = Stage1GatingNetwork(n_channels=n_ch, window_standardization=True)
    logits = gating_net(batch)

    assert logits.shape == (4, 2)
    assert torch.isfinite(logits).all()


def test_stage2_directional_network_forward():
    """Verify Stage 2 Directional Network outputs 2 logits for Grasp vs Extension."""
    torch.manual_seed(42)
    n_ch = 6
    n_times = 256
    smr_batch = torch.randn(4, n_ch, n_times)
    mrcp_batch = torch.randn(4, n_ch, n_times)

    dir_net = Stage2DirectionalNetwork(n_channels=n_ch, window_standardization=True)
    logits = dir_net(smr_batch, mrcp_batch)

    assert logits.shape == (4, 2)
    assert torch.isfinite(logits).all()


def test_hierarchical_two_stage_joint_probabilities():
    """Verify HierarchicalTwoStageEEGNet outputs calibrated sum-to-one probabilities."""
    torch.manual_seed(42)
    n_ch = 6
    n_times = 256
    model = HierarchicalTwoStageEEGNet(n_channels=n_ch, window_standardization=True)

    smr_batch = torch.randn(5, n_ch, n_times)
    mrcp_batch = torch.randn(5, n_ch, n_times)

    # 1. Forward log-probabilities
    log_p = model(smr_batch, mrcp_batch)
    assert log_p.shape == (5, 3)
    probs = torch.exp(log_p).detach().numpy()
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(5), atol=1e-5)

    # 2. Predict proba
    probs_np = model.predict_proba(smr_batch.numpy(), mrcp_batch.numpy())
    assert probs_np.shape == (5, 3)
    np.testing.assert_allclose(probs_np.sum(axis=1), np.ones(5), atol=1e-5)

    # 3. Stacked channels input (B, 2*C, T)
    stacked = torch.cat([mrcp_batch, smr_batch], dim=1)
    log_p_stacked = model(stacked)
    assert log_p_stacked.shape == (5, 3)

    # 4. Gated predictions
    preds = model.predict(smr_batch.numpy(), mrcp_batch.numpy(), gate_threshold=0.60)
    assert preds.shape == (5,)
    assert set(preds).issubset({0, 1, 2})


def test_hierarchical_predict_step_streaming():
    """Verify single-step streaming inference with gate diagnostics."""
    torch.manual_seed(42)
    n_ch = 4
    n_times = 256
    model = HierarchicalTwoStageEEGNet(n_channels=n_ch)

    win_smr = np.random.randn(n_ch, n_times).astype(np.float32)
    win_mrcp = np.random.randn(n_ch, n_times).astype(np.float32)

    step_res = model.predict_step(win_smr, win_mrcp, gate_threshold=0.50, min_margin=0.05)

    assert 'class_id' in step_res
    assert 'class_name' in step_res
    assert 'probabilities' in step_res
    assert 'p_gate_movement' in step_res
    assert 'p_dir_grasp' in step_res
    assert 'p_dir_extension' in step_res
    assert 'gated' in step_res

    probs = np.array(step_res['probabilities'])
    assert len(probs) == 3
    assert np.isclose(probs.sum(), 1.0, atol=1e-5)


def test_hierarchical_ensemble():
    """Verify HierarchicalEnsemble averages probabilities across member models."""
    torch.manual_seed(42)
    n_ch = 4
    m1 = HierarchicalTwoStageEEGNet(n_channels=n_ch)
    m2 = HierarchicalTwoStageEEGNet(n_channels=n_ch)
    ensemble = HierarchicalEnsemble(n_channels=n_ch, member_states=[m1.state_dict(), m2.state_dict()])

    x_smr = torch.randn(3, n_ch, 128)
    x_mrcp = torch.randn(3, n_ch, 128)

    log_p = ensemble(x_smr, x_mrcp)
    assert log_p.shape == (3, 3)
    probs = torch.exp(log_p).detach().numpy()
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(3), atol=1e-5)


def test_compute_hierarchical_metrics():
    """Verify metric computation including Extension recall and Rest false activation rate."""
    y_true = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    # Simulated probabilities: [p_rest, p_grasp, p_ext]
    y_prob = np.array([
        [0.9, 0.05, 0.05],  # 0 -> 0 (correct rest)
        [0.8, 0.10, 0.10],  # 0 -> 0 (correct rest)
        [0.3, 0.60, 0.10],  # 0 -> 1 (false activation!)
        [0.1, 0.80, 0.10],  # 1 -> 1 (correct grasp)
        [0.2, 0.70, 0.10],  # 1 -> 1 (correct grasp)
        [0.1, 0.20, 0.70],  # 2 -> 2 (correct extension)
        [0.1, 0.10, 0.80],  # 2 -> 2 (correct extension)
        [0.2, 0.60, 0.20],  # 2 -> 1 (extension misclassified as grasp)
    ])
    cue_ids = [0, 0, 1, 2, 2, 3, 3, 4]

    metrics = compute_hierarchical_metrics(y_true, y_prob, cue_ids=cue_ids, gate_threshold=0.50)

    assert 'window' in metrics
    assert 'cue' in metrics
    win = metrics['window']
    # 3 rest cues: 2 correct, 1 false activation -> false activation rate = 1/3 = 0.333
    assert np.isclose(win['rest_false_activation_rate'], 1.0 / 3.0, atol=1e-2)
    # Extension recall: 2 correct out of 3 = 0.667
    assert np.isclose(win['finger_extension_recall'], 2.0 / 3.0, atol=1e-2)
    # Grasp recall: 2 correct out of 2 = 1.0
    assert np.isclose(win['volar_grasp_recall'], 1.0, atol=1e-2)


def test_train_and_finetune_hierarchical_flow():
    """Verify end-to-end two-stage training and personal calibration fine-tuning."""
    torch.manual_seed(42)
    np.random.seed(42)
    n_ch = 4
    n_times = 128

    model = HierarchicalTwoStageEEGNet(n_channels=n_ch, window_standardization=True)

    # 1. Synthetic pre-training data
    n_train = 18
    train_x_smr = np.random.randn(n_train, n_ch, n_times).astype(np.float32)
    train_x_mrcp = np.random.randn(n_train, n_ch, n_times).astype(np.float32)
    train_y = np.array([0, 1, 2] * 6, dtype=int)

    n_val = 9
    val_x_smr = np.random.randn(n_val, n_ch, n_times).astype(np.float32)
    val_x_mrcp = np.random.randn(n_val, n_ch, n_times).astype(np.float32)
    val_y = np.array([0, 1, 2] * 3, dtype=int)
    val_cues = list(range(n_val))

    trained_model, best_ba, history = train_hierarchical_two_stage(
        model, train_x_smr, train_x_mrcp, train_y,
        val_x_smr, val_x_mrcp, val_y, val_cue_ids=val_cues,
        epochs=3, lr=1e-2, patience=2, batch_size=6
    )

    assert trained_model is not None
    assert len(history) > 0
    assert 0.0 <= best_ba <= 1.0

    # 2. Personal calibration fine-tuning flow
    calib_raw_train = np.random.randn(12, n_ch, n_times).astype(np.float32)
    calib_y_train = np.array([0, 1, 2] * 4, dtype=int)
    calib_cues_train = list(range(12))

    calib_raw_val = np.random.randn(6, n_ch, n_times).astype(np.float32)
    calib_y_val = np.array([0, 1, 2] * 2, dtype=int)
    calib_cues_val = list(range(12, 18))

    tuned_model, ura, diag = finetune_hierarchical_personal(
        trained_model, calib_raw_train, calib_y_train, calib_cues_train,
        calib_raw_val, calib_y_val, calib_cues_val,
        epochs=3, lr_head=1e-2, lr_backbone=1e-3, patience=2, batch_size=6
    )

    assert tuned_model is not None
    assert ura.alignment_matrix_ is not None
    assert diag['best_val_balanced_accuracy'] >= 0.0


def test_unsupervised_resting_alignment_fit_dual_band_and_serialization():
    """Verify fit_dual_band produces separate C x C matrices with zero cross-band mixing."""
    np.random.seed(42)
    n_ch = 5
    n_times = 128
    n_trials = 16

    # Create stacked resting data: 2*C = 10 channels
    rest_stacked = np.random.randn(n_trials, 2 * n_ch, n_times).astype(np.float32)

    ura = UnsupervisedRestingAlignment(shrinkage=0.05, n_base_channels=n_ch).fit_dual_band(rest_stacked)

    assert ura.n_channels_ == n_ch
    assert ura.alignment_matrix_mrcp_ is not None
    assert ura.alignment_matrix_smr_ is not None
    assert ura.alignment_matrix_mrcp_.shape == (n_ch, n_ch)
    assert ura.alignment_matrix_smr_.shape == (n_ch, n_ch)

    # Test serialization round-trip
    state = ura.state_dict()
    assert 'alignment_matrix_mrcp' in state
    assert 'alignment_matrix_smr' in state

    ura_restored = UnsupervisedRestingAlignment().load_state_dict(state)
    np.testing.assert_allclose(ura.alignment_matrix_mrcp_, ura_restored.alignment_matrix_mrcp_)
    np.testing.assert_allclose(ura.alignment_matrix_smr_, ura_restored.alignment_matrix_smr_)

    # Test transform_dual_band
    mrcp_raw = rest_stacked[:, :n_ch, :]
    smr_raw = rest_stacked[:, n_ch:, :]
    aligned_mrcp, aligned_smr = ura.transform_dual_band(mrcp_raw, smr_raw)
    assert aligned_mrcp.shape == mrcp_raw.shape
    assert aligned_smr.shape == smr_raw.shape

    # Test stacked transform
    aligned_stacked = ura.transform(rest_stacked)
    assert aligned_stacked.shape == rest_stacked.shape
    np.testing.assert_allclose(aligned_stacked[:, :n_ch, :], aligned_mrcp, atol=1e-5)
    np.testing.assert_allclose(aligned_stacked[:, n_ch:, :], aligned_smr, atol=1e-5)


def test_multisubject_pipeline_deduplication():
    """Verify multisubject_pipeline uses canonical UnsupervisedRestingAlignment from hierarchical_classifier."""
    from multisubject_pipeline import UnsupervisedRestingAlignment as URA_multi
    from hierarchical_classifier import UnsupervisedRestingAlignment as URA_hier
    assert URA_multi is URA_hier


def test_hierarchical_model_prefiltered_support():
    """Verify HierarchicalTwoStageEEGNet handles prefiltered separate and stacked inputs."""
    torch.manual_seed(42)
    n_ch = 6
    n_times = 128
    b = 4

    model = HierarchicalTwoStageEEGNet(n_channels=n_ch)
    model.eval()
    x_smr = torch.randn(b, n_ch, n_times)
    x_mrcp = torch.randn(b, n_ch, n_times)

    with torch.no_grad():
        # Separate inputs
        log_p1 = model(x_smr, x_mrcp, prefiltered=True)
        assert log_p1.shape == (b, 3)

        # Stacked inputs [MRCP; SMR]
        x_stacked = torch.cat([x_mrcp, x_smr], dim=1)
        log_p2 = model(x_stacked, prefiltered=True)
        assert log_p2.shape == (b, 3)
        np.testing.assert_allclose(log_p1.cpu().numpy(), log_p2.cpu().numpy(), atol=1e-5)

