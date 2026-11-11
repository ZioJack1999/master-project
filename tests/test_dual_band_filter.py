"""Unit tests for dual_band_filter module."""

import numpy as np
import pytest

from dual_band_filter import (
    DualBandFilter,
    DualBandFilterState,
    DualBandRingBuffer,
    extract_mrcp_temporal_features,
    extract_smr_spectral_features,
    extract_dual_band_features,
    prepare_dual_band_tensor,
    resolve_channel_indices,
    compute_linear_slope,
)


def test_channel_resolution():
    channels = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4']
    resolved = resolve_channel_indices(channels, ('C3', 'Cz', 'C4'))
    assert resolved['C3'] == 3
    assert resolved['Cz'] == 4
    assert resolved['C4'] == 5

    # Case insensitive
    resolved_lower = resolve_channel_indices(['c3', 'cz', 'c4'], ('C3', 'Cz', 'C4'))
    assert resolved_lower['C3'] == 0
    assert resolved_lower['Cz'] == 1
    assert resolved_lower['C4'] == 2

    # None channels
    resolved_none = resolve_channel_indices(None, ('C3', 'Cz', 'C4'))
    assert resolved_none['C3'] == 0
    assert resolved_none['Cz'] == 1
    assert resolved_none['C4'] == 2


def test_linear_slope_computation():
    fs = 256.0
    n = 256  # 1.0 second
    t = np.arange(n) / fs
    target_slope = -2.5  # -2.5 V/s
    y = target_slope * t + 0.1  # with intercept

    computed = compute_linear_slope(y, fs)
    assert np.isclose(computed, target_slope, rtol=1e-3)


def test_dual_band_filter_chunk_continuity():
    """Verify that chunk-by-chunk causal filtering matches continuous filtering."""
    fs = 256.0
    n_ch = 4
    n_samples = 1024
    np.random.seed(42)
    raw = np.random.randn(n_ch, n_samples).astype(np.float32) * 20e-6

    dbf = DualBandFilter(sfreq=fs)

    # 1. Continuous filtering
    state_cont = dbf.create_state(n_ch, initial_sample=raw[:, 0])
    mrcp_cont, smr_cont = dbf.filter_chunk(raw, state_cont)

    # 2. Chunk-by-chunk filtering (e.g. 64 samples per chunk)
    chunk_size = 64
    state_chunk = dbf.create_state(n_ch, initial_sample=raw[:, 0])
    mrcp_chunks = []
    smr_chunks = []

    for pos in range(0, n_samples, chunk_size):
        chunk = raw[:, pos:pos + chunk_size]
        m_c, s_c = dbf.filter_chunk(chunk, state_chunk)
        mrcp_chunks.append(m_c)
        smr_chunks.append(s_c)

    mrcp_chunked = np.concatenate(mrcp_chunks, axis=1)
    smr_chunked = np.concatenate(smr_chunks, axis=1)

    # Must be identical within floating point tolerance
    np.testing.assert_allclose(mrcp_chunked, mrcp_cont, atol=1e-6)
    np.testing.assert_allclose(smr_chunked, smr_cont, atol=1e-6)


def test_initial_dc_correction():
    """Verify that initial DC step transient is suppressed with sosfilt_zi scaling."""
    fs = 256.0
    n_ch = 3
    n_samples = 512
    dc_offset = 80e-6  # 80 microvolts DC
    raw = np.ones((n_ch, n_samples), dtype=np.float32) * dc_offset

    dbf = DualBandFilter(sfreq=fs)

    # Without DC correction (zi=0)
    state_uninit = dbf.create_state(n_ch, initial_sample=None)
    mrcp_uninit, _ = dbf.filter_chunk(raw, state_uninit)
    max_err_uninit = np.max(np.abs(mrcp_uninit))

    # With DC correction (zi scaled by initial sample)
    state_init = dbf.create_state(n_ch, initial_sample=raw[:, 0])
    mrcp_init, _ = dbf.filter_chunk(raw, state_init)
    max_err_init = np.max(np.abs(mrcp_init))

    # Initialized state suppresses transient by several orders of magnitude
    assert max_err_init < 1e-10
    assert max_err_uninit > 1e-5


def test_dual_band_ring_buffer():
    fs = 256.0
    n_ch = 8
    win_len = 512
    ring = DualBandRingBuffer(n_ch, win_len)
    assert not ring.is_full

    chunk = np.random.randn(n_ch, 128).astype(np.float32) * 10e-6
    mrcp_c, smr_c = ring.push_chunk(chunk)
    assert mrcp_c.shape == (n_ch, 128)
    assert smr_c.shape == (n_ch, 128)
    assert not ring.is_full

    # Push remaining to fill buffer
    for _ in range(3):
        ring.push_chunk(chunk)
    assert ring.is_full

    wins = ring.get_windows()
    assert wins['raw'].shape == (n_ch, win_len)
    assert wins['mrcp'].shape == (n_ch, win_len)
    assert wins['smr'].shape == (n_ch, win_len)
    assert wins['bad'].shape == (1, win_len)
    assert wins['quality'] in ('ok', 'artifact', 'flat_signal')


def test_mrcp_feature_extraction():
    fs = 256.0
    channels = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4']
    n_times = 512
    # Simulate a slow negative Bereitschaftspotential on C3
    t = np.arange(n_times) / fs
    mrcp_window = np.zeros((len(channels), n_times), dtype=np.float32)
    # C3 is index 3: negative dip building towards the end
    mrcp_window[3] = -15e-6 * (t / t[-1])**2

    feats, names, fdict = extract_mrcp_temporal_features(
        mrcp_window, sfreq=fs, channels=channels, motor_channels=('C3', 'Cz', 'C4'),
        trajectory_samples=16
    )

    assert isinstance(feats, np.ndarray)
    assert feats.ndim == 1
    assert len(feats) == len(names)
    assert 'C3' in fdict
    assert 'Cz' in fdict
    assert 'C4' in fdict
    assert fdict['C3']['min_amplitude'] < 0.0
    assert fdict['C3']['slope_late'] < 0.0
    assert len(fdict['C3']['trajectory']) == 16
    assert 'diff_c3_c4_late' in fdict


def test_smr_feature_extraction():
    fs = 256.0
    channels = ['C3', 'Cz', 'C4']
    n_times = 512
    t = np.arange(n_times) / fs
    smr_window = np.zeros((len(channels), n_times), dtype=np.float32)
    # Put strong 10 Hz mu on C4 (suppressed on C3 -> left motor imagery / right hand)
    smr_window[0] = 2e-6 * np.sin(2 * np.pi * 10 * t)  # C3
    smr_window[2] = 10e-6 * np.sin(2 * np.pi * 10 * t) # C4

    feats, names, fdict = extract_smr_spectral_features(
        smr_window, sfreq=fs, channels=channels, motor_channels=('C3', 'Cz', 'C4')
    )

    assert isinstance(feats, np.ndarray)
    assert len(feats) == len(names)
    assert 'C3' in fdict
    assert 'C4' in fdict
    assert 'mu_laterality_c3_c4' in fdict
    # Mu power C3 < C4 => laterality = (C3 - C4) / (C3 + C4) should be negative!
    assert fdict['mu_laterality_c3_c4'] < 0.0


def test_extract_dual_band_features_from_raw():
    fs = 256.0
    channels = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4']
    raw_window = np.random.randn(len(channels), 512).astype(np.float32) * 20e-6

    vec, names, meta = extract_dual_band_features(
        raw_window, sfreq=fs, channels=channels, motor_channels=('C3', 'Cz', 'C4')
    )

    assert isinstance(vec, np.ndarray)
    assert len(vec) == len(names)
    assert 'mrcp' in meta
    assert 'smr' in meta


def test_prepare_dual_band_tensor():
    fs = 256.0
    n_trials = 8
    n_ch = 12
    n_times = 256
    raw_batch = np.random.randn(n_trials, n_ch, n_times).astype(np.float32) * 15e-6

    from dual_band_filter import slice_continuous_dual_band_windows

    # Test stacked channels (N, 2*C, T)
    tensor_stacked = prepare_dual_band_tensor(raw_batch, sfreq=fs, mode='stacked_channels')
    assert tensor_stacked.shape == (n_trials, 2 * n_ch, n_times)

    # Test dual branch (N, 2, C, T)
    tensor_branch = prepare_dual_band_tensor(raw_batch, sfreq=fs, mode='dual_branch')
    assert tensor_branch.shape == (n_trials, 2, n_ch, n_times)

    # Test prefiltered pass-through
    tensor_pre = prepare_dual_band_tensor(tensor_stacked, sfreq=fs, mode='dual_branch', prefiltered=True)
    assert tensor_pre.shape == (n_trials, 2, n_ch, n_times)
    np.testing.assert_allclose(tensor_pre[:, 0], tensor_stacked[:, :n_ch])
    np.testing.assert_allclose(tensor_pre[:, 1], tensor_stacked[:, n_ch:])


def test_ring_buffer_matches_continuous_slices_r_gt_0999():
    """Verify DualBandRingBuffer streamed windows match filter_continuous slices (r > 0.999)."""
    from dual_band_filter import slice_continuous_dual_band_windows

    fs = 256.0
    n_ch = 6
    win_samples = 512  # 2.0 s window
    step_samples = 64  # 0.25 s step
    total_samples = 2560  # 10.0 s recording

    rng = np.random.RandomState(123)
    t = np.arange(total_samples) / fs
    raw = np.zeros((n_ch, total_samples), dtype=np.float32)
    for c in range(n_ch):
        # Mix of DC offset, slow 0.8 Hz MRCP wave, 1.8 Hz wave, 10 Hz mu, 22 Hz beta, and noise
        raw[c] = (
            45e-6 * (c + 1)
            + 12e-6 * np.sin(2 * np.pi * 0.8 * t + 0.4 * c)
            + 8e-6 * np.cos(2 * np.pi * 1.8 * t - 0.3 * c)
            + 10e-6 * np.sin(2 * np.pi * 10.5 * t + 0.2 * c)
            + 6e-6 * np.sin(2 * np.pi * 22.0 * t + 0.7 * c)
            + 3e-6 * rng.randn(total_samples)
        ).astype(np.float32)

    dbf = DualBandFilter(sfreq=fs)
    mrcp_cont, smr_cont, _ = dbf.filter_continuous(raw, initial_dc_correction=True)

    ring = DualBandRingBuffer(n_ch, win_samples, dual_filter=dbf)

    # Stream in irregular chunk sizes (e.g. 25, 39, 64 samples) and check every prediction step
    pos = 0
    next_pred = win_samples
    starts_checked = []

    while pos < total_samples:
        chunk_len = min(total_samples - pos, next_pred - pos, 37)
        ring.push_chunk(raw[:, pos:pos + chunk_len])
        pos += chunk_len
        if pos == next_pred:
            wins = ring.get_windows()
            w_start = pos - win_samples
            starts_checked.append(w_start)

            expected_mrcp = mrcp_cont[:, w_start:pos]
            expected_smr = smr_cont[:, w_start:pos]

            # Correlation must exceed 0.999 across all channels
            for c in range(n_ch):
                r_mrcp = np.corrcoef(wins['mrcp'][c], expected_mrcp[c])[0, 1]
                r_smr = np.corrcoef(wins['smr'][c], expected_smr[c])[0, 1]
                assert r_mrcp > 0.99999, f"Channel {c} MRCP correlation {r_mrcp} <= 0.999"
                assert r_smr > 0.99999, f"Channel {c} SMR correlation {r_smr} <= 0.999"

            np.testing.assert_allclose(wins['mrcp'], expected_mrcp, atol=1e-6)
            np.testing.assert_allclose(wins['smr'], expected_smr, atol=1e-6)
            np.testing.assert_allclose(
                ring.get_stacked_window(),
                np.concatenate([expected_mrcp, expected_smr], axis=0),
                atol=1e-6
            )
            next_pred += step_samples

    # Also verify slice_continuous_dual_band_windows matches exact continuous slices
    sliced_stacked = slice_continuous_dual_band_windows(
        raw, starts_checked, win_samples, dual_filter=dbf, sfreq=fs, mode='stacked_channels'
    )
    assert sliced_stacked.shape == (len(starts_checked), 2 * n_ch, win_samples)
    for i, s in enumerate(starts_checked):
        np.testing.assert_allclose(sliced_stacked[i, :n_ch], mrcp_cont[:, s:s + win_samples], atol=1e-6)
        np.testing.assert_allclose(sliced_stacked[i, n_ch:], smr_cont[:, s:s + win_samples], atol=1e-6)

