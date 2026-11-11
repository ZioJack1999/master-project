"""Verify MulticlassDecoder integration with HierarchicalTwoStageEEGNet."""

import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch

from hierarchical_classifier import HierarchicalTwoStageEEGNet, UnsupervisedRestingAlignment, THREE_CLASSES
from multiclass_decoder import MulticlassDecoder


def test_multiclass_decoder_with_hierarchical_checkpoint(tmp_path):
    """Verify MulticlassDecoder loads and executes HierarchicalTwoStageEEGNet."""
    torch.manual_seed(42)
    n_ch = 4
    sfreq = 256
    n_times = 512
    channels = ['C3', 'Cz', 'C4', 'Pz']

    # 1. Build and save hierarchical model
    model = HierarchicalTwoStageEEGNet(n_channels=n_ch, classes=THREE_CLASSES, sfreq=sfreq)
    ura = UnsupervisedRestingAlignment().fit(np.random.randn(n_ch, 512))

    ckpt_path = tmp_path / "hierarchical_test.pt"
    json_path = tmp_path / "hierarchical_test.pt.json"

    torch.save({
        'arch': 'hierarchical_two_stage',
        'model_state': model.state_dict(),
        'resting_alignment': ura.state_dict(),
        'n_channels': n_ch,
        'n_classes': 3,
        'n_times': n_times,
        'channels': channels,
        'classes': THREE_CLASSES,
        'window_standardization': True,
        'train_mean': [0.0] * n_ch,
        'train_std': [1.0] * n_ch,
    }, ckpt_path)

    sha256 = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()

    contract = {
        'schema_version': 2,
        'task_mode': 'multiclass',
        'channels': channels,
        'sfreq': sfreq,
        'classes': THREE_CLASSES,
        'n_times': n_times,
        'normalization': 'unsupervised_resting_alignment',
        'artifact_limit_v': 100e-6,
        'min_signal_std_v': 1e-9,
        'sha256': sha256,
        'filter': {'mode': 'causal', 'order': 4, 'fmin': 8.0, 'fmax': 30.0},
    }
    json_path.write_text(json.dumps(contract, indent=2), encoding='utf-8')

    # 2. Instantiate MulticlassDecoder
    decoder = MulticlassDecoder(checkpoint=str(ckpt_path))
    assert decoder.arch == 'hierarchical_two_stage'
    assert decoder.alignment is not None

    # 3. Stream 4 seconds of raw EEG (1024 samples) in chunks of 64 samples
    raw_stream = np.random.randn(n_ch, 1024).astype(np.float32) * 10e-6  # 10 uV EEG
    chunk_size = 64
    all_outputs = []

    for pos in range(0, 1024, chunk_size):
        chunk = raw_stream[:, pos:pos + chunk_size]
        outputs = decoder.process_chunk_all(chunk)
        all_outputs.extend(outputs)

    # After initial window (512 samples), predictions should be produced every step_samples (64 samples)
    assert len(all_outputs) > 0
    for out in all_outputs:
        assert 'class_id' in out
        assert 'class_name' in out
        assert 'probabilities_raw' in out
        assert len(out['probabilities_raw']) == 3
        assert np.isclose(sum(out['probabilities_raw']), 1.0, atol=1e-4)
        assert out['quality'] == 'ok'
        assert 'stage1_diagnostics' in out
        assert 'stage2_diagnostics' in out
        assert 'hierarchical_diagnostics' in out
        assert out['stage1_diagnostics']['gate_threshold'] == 0.50
        assert 'p_movement' in out['stage1_diagnostics']
        assert 'p_grasp_conditional' in out['stage2_diagnostics']
        assert out['hierarchical_diagnostics']['active'] is True


def test_multiclass_decoder_hierarchical_dual_band_arch(tmp_path):
    """Verify MulticlassDecoder loads and executes with arch='hierarchical_dual_band'."""
    torch.manual_seed(42)
    n_ch = 4
    sfreq = 256
    n_times = 512
    channels = ['C3', 'Cz', 'C4', 'Pz']

    model = HierarchicalTwoStageEEGNet(n_channels=n_ch, classes=THREE_CLASSES, sfreq=sfreq)
    ura = UnsupervisedRestingAlignment().fit(np.random.randn(n_ch, 512))

    ckpt_path = tmp_path / "hierarchical_dual_band.pt"
    json_path = tmp_path / "hierarchical_dual_band.pt.json"

    torch.save({
        'arch': 'hierarchical_dual_band',
        'model_state': model.state_dict(),
        'resting_alignment': ura.state_dict(),
        'n_channels': n_ch,
        'n_classes': 3,
        'n_times': n_times,
        'channels': channels,
        'classes': THREE_CLASSES,
        'window_standardization': True,
        'train_mean': [0.0] * n_ch,
        'train_std': [1.0] * n_ch,
    }, ckpt_path)

    sha256 = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()

    contract = {
        'schema_version': 2,
        'task_mode': 'multiclass',
        'channels': channels,
        'sfreq': sfreq,
        'classes': THREE_CLASSES,
        'n_times': n_times,
        'normalization': 'unsupervised_resting_alignment',
        'artifact_limit_v': 100e-6,
        'min_signal_std_v': 1e-9,
        'sha256': sha256,
        'filter': {'mode': 'causal', 'order': 4, 'fmin': 8.0, 'fmax': 30.0},
    }
    json_path.write_text(json.dumps(contract, indent=2), encoding='utf-8')

    decoder = MulticlassDecoder(checkpoint=str(ckpt_path))
    assert decoder.is_hierarchical is True
    assert decoder.is_dual_band is True

    raw_stream = np.random.randn(n_ch, 1024).astype(np.float32) * 10e-6
    outputs = decoder.process_chunk_all(raw_stream)
    assert len(outputs) > 0
    first = outputs[0]
    assert 'hierarchical_diagnostics' in first
    assert first['hierarchical_diagnostics']['active'] is True
    assert 'stage1' in first['hierarchical_diagnostics']
    assert 'stage2' in first['hierarchical_diagnostics']


def test_realtime_bci_decoder_with_hierarchical_checkpoint(tmp_path):
    """Verify RealtimeBCIDecoder executes with hierarchical_dual_band and emits stage diagnostics."""
    from realtime_bci_decoder import RealtimeBCIDecoder
    torch.manual_seed(42)
    n_ch = 4
    sfreq = 256
    n_times = 512
    channels = ['C3', 'Cz', 'C4', 'Pz']

    model = HierarchicalTwoStageEEGNet(n_channels=n_ch, classes=THREE_CLASSES, sfreq=sfreq)
    ura = UnsupervisedRestingAlignment().fit(np.random.randn(n_ch, 512))

    ckpt_path = tmp_path / "rt_hierarchical.pt"
    json_path = tmp_path / "rt_hierarchical.pt.json"

    torch.save({
        'arch': 'hierarchical_dual_band',
        'model_state': model.state_dict(),
        'resting_alignment': ura.state_dict(),
        'n_channels': n_ch,
        'n_classes': 3,
        'n_times': n_times,
        'channels': channels,
        'classes': ['rest', 'action'],
        'window_standardization': True,
        'train_mean': [0.0] * n_ch,
        'train_std': [1.0] * n_ch,
    }, ckpt_path)

    sha256 = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()

    contract = {
        'schema_version': 2,
        'task_mode': 'binary',
        'channels': channels,
        'sfreq': sfreq,
        'classes': ['rest', 'action'],
        'n_times': n_times,
        'normalization': 'unsupervised_resting_alignment',
        'artifact_limit_v': 100e-6,
        'min_signal_std_v': 1e-9,
        'verified': True,
        'sha256': sha256,
        'filter': {'mode': 'causal', 'order': 4, 'fmin': 8.0, 'fmax': 30.0},
    }
    json_path.write_text(json.dumps(contract, indent=2), encoding='utf-8')

    rt_decoder = RealtimeBCIDecoder(str(ckpt_path), send_udp=False)
    assert rt_decoder.is_hierarchical is True
    assert rt_decoder.is_dual_band is True

    raw_stream = np.random.randn(n_ch, 1024).astype(np.float32) * 10e-6
    outputs = rt_decoder.process_chunk_all(raw_stream, emit=False)
    assert len(outputs) > 0
    packet = outputs[0]
    assert 'hierarchical_diagnostics' in packet
    assert packet['hierarchical_diagnostics']['active'] is True
    assert 'stage1_diagnostics' in packet
    assert 'stage2_diagnostics' in packet


def test_threeclass_studio_surfaces_hierarchical_diagnostics(tmp_path):
    """Verify ThreeClassStudio surfaces stage1/stage2/hierarchical diagnostics in /api/state."""
    from threeclass_studio import ThreeClassStudio
    torch.manual_seed(42)
    n_ch = 20
    channels = ['Fp1','Fp2','F7','F3','Fz','F4','F8','T3','C3','Cz','C4','T4','T5','P3','Pz','P4','T6','O1','POz','O2']
    sfreq = 256
    n_times = 512

    model = HierarchicalTwoStageEEGNet(n_channels=n_ch, classes=THREE_CLASSES, sfreq=sfreq)
    ura = UnsupervisedRestingAlignment().fit(np.random.randn(n_ch, 512))

    ckpt_path = tmp_path / "studio_hierarchical.pt"
    json_path = tmp_path / "studio_hierarchical.pt.json"

    torch.save({
        'arch': 'hierarchical_dual_band',
        'model_state': model.state_dict(),
        'resting_alignment': ura.state_dict(),
        'n_channels': n_ch,
        'n_classes': 3,
        'n_times': n_times,
        'channels': channels,
        'classes': THREE_CLASSES,
        'window_standardization': True,
        'train_mean': [0.0] * n_ch,
        'train_std': [1.0] * n_ch,
    }, ckpt_path)

    sha256 = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()

    contract = {
        'schema_version': 2,
        'task_mode': 'multiclass',
        'channels': channels,
        'sfreq': sfreq,
        'classes': THREE_CLASSES,
        'n_times': n_times,
        'normalization': 'unsupervised_resting_alignment',
        'artifact_limit_v': 100e-6,
        'min_signal_std_v': 1e-9,
        'verified': True,
        'sha256': sha256,
        'filter': {'mode': 'causal', 'order': 4, 'fmin': 8.0, 'fmax': 30.0},
    }
    json_path.write_text(json.dumps(contract, indent=2), encoding='utf-8')

    studio = ThreeClassStudio(dict(checkpoint=str(ckpt_path), channels=channels), root=tmp_path, clock=lambda: 100.0)
    state = studio.snapshot()
    assert 'hierarchical_diagnostics' in state
    assert 'stage1_diagnostics' in state
    assert 'stage2_diagnostics' in state

    # Start stream and ingest chunks to test live updates
    studio.state['connected'] = True
    studio.action('/api/start')
    raw = np.random.randn(n_ch, 640).astype(np.float32) * 5e-6
    ts = 100.0 + np.arange(640) / 256.0
    studio.ingest(raw, ts)

    state = studio.snapshot()
    assert state['hierarchical_diagnostics'] is not None
    assert state['hierarchical_diagnostics']['active'] is True
    assert 'p_gate_movement' in state['hierarchical_diagnostics']
    studio.close()


def test_bci_studio_gui_surfaces_hierarchical_diagnostics(tmp_path):
    """Verify Studio in bci_studio_gui surfaces stage1/stage2/hierarchical diagnostics."""
    from bci_studio_gui import Studio
    torch.manual_seed(42)
    n_ch = 4
    channels = ['C3', 'Cz', 'C4', 'Pz']
    sfreq = 256
    n_times = 512

    model = HierarchicalTwoStageEEGNet(n_channels=n_ch, classes=['rest', 'action'], sfreq=sfreq)
    ura = UnsupervisedRestingAlignment().fit(np.random.randn(n_ch, 512))

    ckpt_path = tmp_path / "bci_studio_hierarchical.pt"
    json_path = tmp_path / "bci_studio_hierarchical.pt.json"

    torch.save({
        'arch': 'hierarchical_dual_band',
        'model_state': model.state_dict(),
        'resting_alignment': ura.state_dict(),
        'n_channels': n_ch,
        'n_classes': 2,
        'n_times': n_times,
        'channels': channels,
        'classes': ['rest', 'action'],
        'window_standardization': True,
        'train_mean': [0.0] * n_ch,
        'train_std': [1.0] * n_ch,
    }, ckpt_path)

    sha256 = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()

    contract = {
        'schema_version': 2,
        'task_mode': 'binary',
        'channels': channels,
        'sfreq': sfreq,
        'classes': ['rest', 'action'],
        'n_times': n_times,
        'normalization': 'unsupervised_resting_alignment',
        'artifact_limit_v': 100e-6,
        'min_signal_std_v': 1e-9,
        'verified': True,
        'sha256': sha256,
        'filter': {'mode': 'causal', 'order': 4, 'fmin': 8.0, 'fmax': 30.0},
    }
    json_path.write_text(json.dumps(contract, indent=2), encoding='utf-8')

    studio = Studio(dict(checkpoint=str(ckpt_path), channels=channels), root=tmp_path, clock=lambda: 100.0)
    state = studio.snapshot()
    assert 'hierarchical_diagnostics' in state
    assert 'stage1_diagnostics' in state
    assert 'stage2_diagnostics' in state

    studio.state['connected'] = True
    studio.state['streaming'] = True
    raw = np.random.randn(n_ch, 640).astype(np.float32) * 5e-6
    ts = 100.0 + np.arange(640) / 256.0
    studio.ingest(raw, ts, 100.0)

    state = studio.snapshot()
    assert state['hierarchical_diagnostics'] is not None
    assert state['hierarchical_diagnostics']['active'] is True
    assert 'stage1' in state['hierarchical_diagnostics']
    studio.pause('test_done')


def test_standard_model_backwards_compatible(tmp_path):
    """Verify standard EEGNet models produce None for hierarchical diagnostics."""
    from networks import EEGNet
    from realtime_bci_decoder import RealtimeBCIDecoder
    from multiclass_decoder import MulticlassDecoder
    n_ch = 4
    sfreq = 256
    n_times = 512
    channels = ['C3', 'Cz', 'C4', 'Pz']

    model = EEGNet(n_ch=n_ch, n_t=n_times, n_cls=2)
    ckpt_path = tmp_path / "standard_eegnet.pt"
    json_path = tmp_path / "standard_eegnet.pt.json"

    torch.save({
        'arch': 'eegnet',
        'model_state': model.state_dict(),
        'n_channels': n_ch,
        'n_classes': 2,
        'n_times': n_times,
        'channels': channels,
        'classes': ['rest', 'action'],
        'train_mean': [0.0] * n_ch,
        'train_std': [1.0] * n_ch,
    }, ckpt_path)

    sha256 = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    contract = {
        'schema_version': 2,
        'task_mode': 'binary',
        'channels': channels,
        'sfreq': sfreq,
        'classes': ['rest', 'action'],
        'n_times': n_times,
        'normalization': 'fixed',
        'artifact_limit_v': 100e-6,
        'min_signal_std_v': 1e-9,
        'verified': True,
        'sha256': sha256,
        'filter': {'mode': 'causal', 'order': 4, 'fmin': 8.0, 'fmax': 30.0},
    }
    json_path.write_text(json.dumps(contract, indent=2), encoding='utf-8')

    multi_dec = MulticlassDecoder(str(ckpt_path))
    assert multi_dec.is_hierarchical is False
    out = multi_dec.process_chunk_all(np.random.randn(n_ch, 640).astype(np.float32) * 5e-6)
    assert 'hierarchical_diagnostics' not in out[0]

    rt_dec = RealtimeBCIDecoder(str(ckpt_path), send_udp=False)
    assert rt_dec.is_hierarchical is False
    out_rt = rt_dec.process_chunk_all(np.random.randn(n_ch, 640).astype(np.float32) * 5e-6, emit=False)
    assert 'hierarchical_diagnostics' not in out_rt[0]


def test_multiclass_decoder_with_dual_band_ura(tmp_path):
    """Verify MulticlassDecoder executes seamlessly with per-band C x C UnsupervisedRestingAlignment."""
    torch.manual_seed(42)
    n_ch = 4
    sfreq = 256
    n_times = 512
    channels = ['C3', 'Cz', 'C4', 'Pz']

    # Build model & per-band URA
    model = HierarchicalTwoStageEEGNet(n_channels=n_ch, classes=THREE_CLASSES, sfreq=sfreq)
    rest_stacked = np.random.randn(10, 2 * n_ch, 256)
    ura = UnsupervisedRestingAlignment(shrinkage=0.05, n_base_channels=n_ch).fit_dual_band(rest_stacked)

    ckpt_path = tmp_path / "hierarchical_dual_ura.pt"
    json_path = tmp_path / "hierarchical_dual_ura.pt.json"

    torch.save({
        'arch': 'hierarchical_two_stage',
        'model_state': model.state_dict(),
        'resting_alignment': ura.state_dict(),
        'n_channels': n_ch,
        'n_classes': 3,
        'n_times': n_times,
        'channels': channels,
        'classes': THREE_CLASSES,
        'window_standardization': True,
        'train_mean': [0.0] * n_ch,
        'train_std': [1.0] * n_ch,
    }, ckpt_path)

    sha256 = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    contract = {
        'schema_version': 2,
        'task_mode': 'multiclass',
        'channels': channels,
        'sfreq': sfreq,
        'classes': THREE_CLASSES,
        'n_times': n_times,
        'normalization': 'unsupervised_resting_alignment',
        'artifact_limit_v': 100e-6,
        'min_signal_std_v': 1e-9,
        'sha256': sha256,
        'filter': {'mode': 'causal', 'order': 4, 'fmin': 8.0, 'fmax': 30.0},
    }
    json_path.write_text(json.dumps(contract, indent=2), encoding='utf-8')

    decoder = MulticlassDecoder(checkpoint=str(ckpt_path))
    assert decoder.alignment.alignment_matrix_mrcp_ is not None
    assert decoder.alignment.alignment_matrix_smr_ is not None

    # Stream chunks of 64 samples
    raw_stream = np.random.randn(n_ch, 1024).astype(np.float32) * 10e-6
    all_outputs = []
    for pos in range(0, 1024, 64):
        out = decoder.process_chunk_all(raw_stream[:, pos:pos + 64])
        all_outputs.extend(out)

    assert len(all_outputs) > 0
    for out in all_outputs:
        assert out['quality'] == 'ok'
        assert len(out['probabilities_raw']) == 3
        assert np.isclose(sum(out['probabilities_raw']), 1.0, atol=1e-4)
        assert out['hierarchical_diagnostics']['active'] is True

