"""Two-Stage Hierarchical EEG Classifier & Unsupervised Resting Alignment.

Implements:
1. Unsupervised Resting Alignment (Sion An et al. style):
   - Spatial covariance whitening R_rest^(-1/2) * X using resting EEG.
   - Normalizes inter-subject skull conductivity, electrode impedances, and head geometry.
   - Requires zero task-labeled calibration trials before inference.
   - Online recursive streaming covariance update during rest intervals.

2. Two-Stage Hierarchical Architecture:
   - Stage 1 (Movement Gating): Binary classifier (Rest vs. Intentional Movement)
     operating on spectral sensorimotor rhythms (SMR, 4.0-40.0 Hz).
     Prevents false activations during rest.
   - Stage 2 (Directional Discrimination): Binary classifier (Volar Grasp vs. Finger Extension)
     combining slow Movement-Related Cortical Potentials (MRCPs, 0.3-3.0 Hz) and SMR.
     Resolves the historical 0/20 Extension collapse by utilizing low-frequency
     waveform dynamics (Bereitschaftspotential and motor potential slope/polarity).

3. Probability Composition & Decoupled Inference:
   - P(rest) = P_gate(rest)
   - P(volar_grasp) = P_gate(movement) * P_dir(volar_grasp | movement)
   - P(finger_extension) = P_gate(movement) * P_dir(finger_extension | movement)
   - Guaranteed sum-to-one calibrated probabilities.
   - Tunable gating threshold theta_gate to strictly control false activations during rest.

4. End-to-end training and personal calibration fine-tuning routines
   with whole-cue held-out partitions and zero window leakage.
"""

from collections import Counter, defaultdict
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.signal import butter, sosfilt
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from dual_band_filter import (
        DualBandFilter,
        DualBandFilterState,
        DualBandRingBuffer,
        prepare_dual_band_tensor,
    )
except ImportError:
    from .dual_band_filter import (
        DualBandFilter,
        DualBandFilterState,
        DualBandRingBuffer,
        prepare_dual_band_tensor,
    )


# Standard BCI class names
THREE_CLASSES = ['rest', 'volar_grasp', 'finger_extension']
BINARY_CLASSES = ['rest', 'action']


# =====================================================================
# 1. Sion An et al. Unsupervised Resting Alignment
# =====================================================================

class UnsupervisedRestingAlignment:
    """Unsupervised Euclidean Alignment via Resting EEG Covariance Whitening.

    Reference: Sion An et al., 'Unsupervised Subject Adaptation using Resting-State
    EEG for Motor Imagery Brain-Computer Interfaces'.

    Projects subject spatial covariance to unit-variance identity I_C based solely on resting-state EEG:
        R_rest = 1/T_rest * (X_rest - mu)(X_rest - mu)^T
        R_reg = (1 - shrinkage) * (R_rest / (trace(R_rest)/C)) + shrinkage * I_C
        W_rest = (trace(R_rest)/C)^(-1/2) * R_reg^(-1/2)
        X_aligned = W_rest @ X

    Advantages:
    - Zero task labels required: can be fitted from a brief resting baseline (5-30s)
      or from explicit rest trials/intervals.
    - Completely eliminates inter-subject head volume conduction differences, skull
      thickness variations, and channel impedance imbalances.
    - Once aligned, resting EEG covariance is standardized to unit-variance identity I_C,
      and intentional motor commands (grasp and extension) appear as directional deviations.
    - For dual-band (MRCP + SMR) stacked inputs, fits and applies per-band C x C
      whitening matrices independently so MRCP and SMR channels never mix.
    """

    def __init__(self, shrinkage: float = 0.05, sfreq: float = 256.0,
                 n_base_channels: Optional[int] = None):
        self.shrinkage = float(shrinkage)
        self.sfreq = float(sfreq)
        self.n_base_channels = int(n_base_channels) if n_base_channels is not None else None
        self.reference_cov_: Optional[np.ndarray] = None
        self.alignment_matrix_: Optional[np.ndarray] = None
        self.alignment_matrix_mrcp_: Optional[np.ndarray] = None
        self.alignment_matrix_smr_: Optional[np.ndarray] = None
        self.n_channels_: Optional[int] = None
        self.n_samples_fitted_: int = 0

    @staticmethod
    def _compute_whitening_from_array(resting_eeg: np.ndarray, shrinkage: float) -> Tuple[np.ndarray, np.ndarray, int, int]:
        if resting_eeg.ndim == 2:
            n_channels, n_samples = resting_eeg.shape
            centered = resting_eeg - resting_eeg.mean(axis=1, keepdims=True)
            cov = (centered @ centered.T) / max(1, (n_samples - 1))
            total_samples = n_samples
        elif resting_eeg.ndim == 3:
            n_trials, n_channels, n_times = resting_eeg.shape
            covs = []
            for i in range(n_trials):
                trial = resting_eeg[i]
                centered = trial - trial.mean(axis=1, keepdims=True)
                c = (centered @ centered.T) / max(1, (n_times - 1))
                covs.append(c)
            cov = np.mean(covs, axis=0)
            total_samples = n_trials * n_times
        else:
            raise ValueError(f"Expected 2D or 3D array for resting_eeg, got shape {resting_eeg.shape}")

        trace_mean = np.trace(cov) / n_channels
        if trace_mean < 1e-18:
            trace_mean = 1.0
        cov_norm = cov / trace_mean
        reg_cov = (1.0 - shrinkage) * cov_norm + shrinkage * np.eye(n_channels)

        vals, vecs = np.linalg.eigh(reg_cov)
        vals = np.maximum(vals, 1e-12)
        inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
        # Scale by 1/sqrt(trace_mean) to standardize resting covariance to unit-variance identity I_C
        inv_sqrt = inv_sqrt / np.sqrt(trace_mean)
        return reg_cov.astype(np.float32), inv_sqrt.astype(np.float32), int(n_channels), int(total_samples)

    def fit(self, resting_eeg: Union[np.ndarray, List[np.ndarray]],
            n_base_channels: Optional[int] = None) -> 'UnsupervisedRestingAlignment':
        """Fit whitening matrix from resting EEG signal.

        If n_base_channels=C is set and resting_eeg has 2*C channels (stacked [MRCP, SMR]),
        fits separate C x C whitening matrices per band without cross-band mixing.
        """
        if isinstance(resting_eeg, list):
            resting_eeg = np.stack(resting_eeg)
        resting_eeg = np.asarray(resting_eeg, dtype=np.float64)

        base_c = n_base_channels or self.n_base_channels
        ch_dim = resting_eeg.shape[0] if resting_eeg.ndim == 2 else resting_eeg.shape[1]

        if base_c is not None and ch_dim == 2 * base_c:
            return self.fit_dual_band(resting_eeg, n_base_channels=base_c)

        reg_cov, inv_sqrt, n_ch, total_samples = self._compute_whitening_from_array(resting_eeg, self.shrinkage)
        self.n_channels_ = n_ch
        self.n_samples_fitted_ = total_samples
        self.reference_cov_ = reg_cov
        self.alignment_matrix_ = inv_sqrt
        self.alignment_matrix_mrcp_ = None
        self.alignment_matrix_smr_ = None
        return self

    def fit_dual_band(self, resting_mb: Union[np.ndarray, List[np.ndarray]],
                      n_base_channels: Optional[int] = None) -> 'UnsupervisedRestingAlignment':
        """Fit independent C x C whitening matrices on stacked (MRCP, SMR) resting EEG.

        Guarantees zero cross-band mixing between slow MRCP (0..C-1) and SMR (C..2C-1).
        """
        if isinstance(resting_mb, list):
            resting_mb = np.stack(resting_mb)
        resting_mb = np.asarray(resting_mb, dtype=np.float64)

        if resting_mb.ndim == 2:
            total_ch = resting_mb.shape[0]
            c = n_base_channels or (total_ch // 2)
            rest_mrcp = resting_mb[:c, :]
            rest_smr = resting_mb[c:2 * c, :]
        elif resting_mb.ndim == 3:
            total_ch = resting_mb.shape[1]
            c = n_base_channels or (total_ch // 2)
            rest_mrcp = resting_mb[:, :c, :]
            rest_smr = resting_mb[:, c:2 * c, :]
        else:
            raise ValueError(f"Expected 2D or 3D stacked array, got {resting_mb.shape}")

        cov_mrcp, w_mrcp, _, n_samp = self._compute_whitening_from_array(rest_mrcp, self.shrinkage)
        cov_smr, w_smr, _, _ = self._compute_whitening_from_array(rest_smr, self.shrinkage)

        self.n_channels_ = int(c)
        self.n_base_channels = int(c)
        self.n_samples_fitted_ = int(n_samp)
        self.reference_cov_ = cov_smr
        self.alignment_matrix_ = w_smr
        self.alignment_matrix_mrcp_ = w_mrcp
        self.alignment_matrix_smr_ = w_smr
        return self

    def fit_from_rest(self, resting_eeg: Union[np.ndarray, List[np.ndarray]],
                      n_base_channels: Optional[int] = None) -> 'UnsupervisedRestingAlignment':
        """Alias for fit() specifically taking resting EEG trials/baseline."""
        return self.fit(resting_eeg, n_base_channels=n_base_channels)

    def fit_resting_eeg(self, resting_eeg: Union[np.ndarray, List[np.ndarray]],
                        fs: float = 256.0, window_s: float = 2.0,
                        n_base_channels: Optional[int] = None) -> 'UnsupervisedRestingAlignment':
        """Convenience alias for fit() on continuous or windowed resting EEG."""
        return self.fit(resting_eeg, n_base_channels=n_base_channels)

    def fit_from_dataset(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                         rest_label: int = 0,
                         n_base_channels: Optional[int] = None) -> 'UnsupervisedRestingAlignment':
        """Fit alignment using resting trials in dataset.

        If y is provided, extracts all windows where y == rest_label.
        If no rest trials are present or y is None, falls back to all windows (standard EA).
        """
        X = np.asarray(X, dtype=np.float32)
        if y is not None:
            y = np.asarray(y)
            mask = (y == rest_label)
            if np.any(mask):
                return self.fit(X[mask], n_base_channels=n_base_channels)
        return self.fit(X, n_base_channels=n_base_channels)

    def transform(self, X: Union[np.ndarray, List[np.ndarray]],
                  band: Optional[str] = None) -> np.ndarray:
        """Apply resting alignment whitening to trials or streaming windows.

        Parameters:
            X: (C, T), (2*C, T), (N, C, T), or (N, 2*C, T) array in Volts.
            band: Optional explicit band selector ('mrcp' or 'smr'). If None,
                  uses stacked logic or default alignment matrix.
        Returns:
            X_aligned: array of same shape, whitened per C-channel band without cross-band mixing.
        """
        if self.alignment_matrix_ is None:
            raise RuntimeError("UnsupervisedRestingAlignment has not been fitted.")
        if isinstance(X, list):
            X = np.stack(X)
        X = np.asarray(X, dtype=np.float32)

        if band == 'mrcp' and self.alignment_matrix_mrcp_ is not None:
            w = self.alignment_matrix_mrcp_
        elif band == 'smr' and self.alignment_matrix_smr_ is not None:
            w = self.alignment_matrix_smr_
        else:
            w = self.alignment_matrix_

        w_mrcp = self.alignment_matrix_mrcp_ if self.alignment_matrix_mrcp_ is not None else w
        w_smr = self.alignment_matrix_smr_ if self.alignment_matrix_smr_ is not None else w

        if X.ndim == 2:
            if band is None and self.n_channels_ is not None and X.shape[0] == 2 * self.n_channels_:
                mrcp = (w_mrcp @ X[:self.n_channels_]).astype(np.float32)
                smr = (w_smr @ X[self.n_channels_:]).astype(np.float32)
                return np.concatenate([mrcp, smr], axis=0)
            return (w @ X).astype(np.float32)
        elif X.ndim == 3:
            if band is None and self.n_channels_ is not None and X.shape[1] == 2 * self.n_channels_:
                # Dual-band stacked channels: align MRCP and SMR independently with C x C matrices
                mrcp = np.einsum('ij,njk->nik', w_mrcp, X[:, :self.n_channels_, :])
                smr = np.einsum('ij,njk->nik', w_smr, X[:, self.n_channels_:, :])
                return np.concatenate([mrcp, smr], axis=1).astype(np.float32)
            # (N, C, T) -> W @ X_i
            return np.einsum('ij,njk->nik', w, X).astype(np.float32)
        elif X.ndim == 4:
            # Multi-branch tensor (N, B, C, T) -> align channels C
            return np.einsum('ij,nbjk->nbik', w, X).astype(np.float32)
        else:
            raise ValueError(f"Unsupported array shape {X.shape}")

    def transform_dual_band(self, mrcp: np.ndarray, smr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Align MRCP and SMR arrays independently with their respective C x C whitening matrices."""
        aligned_mrcp = self.transform(mrcp, band='mrcp')
        aligned_smr = self.transform(smr, band='smr')
        return aligned_mrcp, aligned_smr

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                      rest_label: int = 0) -> np.ndarray:
        return self.fit_from_dataset(X, y, rest_label=rest_label).transform(X)

    def update_online(self, resting_chunk: np.ndarray, alpha_ema: float = 0.05) -> None:
        """Recursively adapt resting covariance estimate during detected rest.

        Allows zero-shot wearable continuous tracking as electrode impedances drift.
        """
        if self.reference_cov_ is None or self.alignment_matrix_ is None:
            self.fit(resting_chunk)
            return

        resting_chunk = np.asarray(resting_chunk, dtype=np.float64)
        if resting_chunk.ndim != 2 or resting_chunk.shape[0] != self.n_channels_:
            return

        n_ch, n_samples = resting_chunk.shape
        centered = resting_chunk - resting_chunk.mean(axis=1, keepdims=True)
        chunk_cov = (centered @ centered.T) / max(1, (n_samples - 1))
        trace_mean = np.trace(chunk_cov) / n_ch
        if trace_mean > 1e-18:
            chunk_cov = chunk_cov / trace_mean

        # EMA blend
        new_cov = (1.0 - alpha_ema) * self.reference_cov_ + alpha_ema * chunk_cov
        vals, vecs = np.linalg.eigh(new_cov)
        vals = np.maximum(vals, 1e-12)
        self.alignment_matrix_ = (vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T).astype(np.float32)
        self.reference_cov_ = new_cov.astype(np.float32)

    def state_dict(self) -> Dict[str, Union[float, int, list, None]]:
        return {
            'shrinkage': self.shrinkage,
            'sfreq': self.sfreq,
            'n_channels': self.n_channels_,
            'n_base_channels': self.n_base_channels,
            'n_samples_fitted': self.n_samples_fitted_,
            'reference_cov': self.reference_cov_.tolist() if self.reference_cov_ is not None else None,
            'alignment_matrix': self.alignment_matrix_.tolist() if self.alignment_matrix_ is not None else None,
            'alignment_matrix_mrcp': self.alignment_matrix_mrcp_.tolist() if self.alignment_matrix_mrcp_ is not None else None,
            'alignment_matrix_smr': self.alignment_matrix_smr_.tolist() if self.alignment_matrix_smr_ is not None else None,
        }

    def load_state_dict(self, state: Dict) -> 'UnsupervisedRestingAlignment':
        self.shrinkage = float(state.get('shrinkage', 0.05))
        self.sfreq = float(state.get('sfreq', 256.0))
        self.n_channels_ = state.get('n_channels')
        self.n_base_channels = state.get('n_base_channels')
        self.n_samples_fitted_ = int(state.get('n_samples_fitted', 0))
        if state.get('reference_cov') is not None:
            self.reference_cov_ = np.array(state['reference_cov'], dtype=np.float32)
        if state.get('alignment_matrix') is not None:
            self.alignment_matrix_ = np.array(state['alignment_matrix'], dtype=np.float32)
        if state.get('alignment_matrix_mrcp') is not None:
            self.alignment_matrix_mrcp_ = np.array(state['alignment_matrix_mrcp'], dtype=np.float32)
        if state.get('alignment_matrix_smr') is not None:
            self.alignment_matrix_smr_ = np.array(state['alignment_matrix_smr'], dtype=np.float32)
        return self


# =====================================================================
# 2. Stage 1: Movement Gating Network (Rest vs. Intentional Movement)
# =====================================================================

class Stage1GatingNetwork(nn.Module):
    """Stage 1 Movement Gating Network.

    Binary classifier: Rest (0) vs. Intentional Movement (1, Grasp or Extension).
    Operates on Sensorimotor Rhythms (SMR, 4.0-40.0 Hz).
    Utilizes multi-scale temporal convolutions and spatial depthwise filtering to detect
    bilateral and contralateral Event-Related Desynchronization (ERD) in mu and beta bands.
    """

    def __init__(self, n_channels: int, window_standardization: bool = True):
        super().__init__()
        self.n_channels = int(n_channels)
        self.window_standardization = bool(window_standardization)

        # Multi-scale temporal filter bank for oscillatory bursts (theta, mu, beta)
        self.temporal_branches = nn.ModuleList([
            nn.Conv2d(1, 4, (1, width), padding=(0, width // 2), bias=False)
            for width in (17, 33, 65)
        ])
        total_temp_channels = 4 * len(self.temporal_branches)  # 12

        # Depthwise spatial filter across electrodes
        self.spatial = nn.Sequential(
            nn.BatchNorm2d(total_temp_channels),
            nn.Conv2d(total_temp_channels, 24, (self.n_channels, 1),
                      groups=total_temp_channels, bias=False),
            nn.BatchNorm2d(24),
            nn.ELU(),
        )

        # Temporal power pooling
        self.power_pool = nn.AvgPool1d(16)

        # Dilated contextual temporal convolutions
        self.context = nn.Sequential(
            nn.Conv1d(24, 24, 5, padding=2, bias=False),
            nn.BatchNorm1d(24),
            nn.ELU(),
            nn.Conv1d(24, 24, 5, dilation=2, padding=4, bias=False),
            nn.BatchNorm1d(24),
            nn.ELU(),
        )

        # Binary Gating Head: [logit_rest, logit_movement]
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(2),
            nn.Flatten(),
            nn.Dropout(0.40),
            nn.Linear(48, 2),
        )

    def forward(self, x_smr: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters:
            x_smr: (B, C, T) tensor in SMR band (4-40 Hz).
        Returns:
            logits: (B, 2) where index 0 is Rest, index 1 is Movement.
        """
        if self.window_standardization:
            x_smr = x_smr - x_smr.mean(dim=1, keepdim=True)
            channel_rms = x_smr.square().mean(dim=2).sqrt()
            scale = channel_rms.median(dim=1).values.clamp_min(1e-6)
            x_smr = x_smr / scale[:, None, None]

        x_4d = x_smr.unsqueeze(1)  # (B, 1, C, T)
        temp_features = torch.cat([b(x_4d) for b in self.temporal_branches], dim=1)
        spatial_features = self.spatial(temp_features).squeeze(2)  # (B, 24, T)

        # Log-power representation
        power = torch.log1p(self.power_pool(spatial_features.square()))
        context_features = power + self.context(power)
        return self.head(context_features)


# =====================================================================
# 3. Stage 2: Directional Discrimination Network (Grasp vs. Extension)
# =====================================================================

class Stage2DirectionalNetwork(nn.Module):
    """Stage 2 Directional Discrimination Network.

    Binary classifier: Volar Grasp (0) vs. Finger Extension (1).
    Dual-Band Architecture combining:
    - Branch A: Slow Movement-Related Cortical Potentials (MRCP, 0.3-3.0 Hz)
      Captures waveform morphology, Bereitschaftspotential polarity, and negative slope.
    - Branch B: Sensorimotor Rhythms (SMR, 4.0-40.0 Hz)
      Captures high-frequency power envelopes and contralateral laterality.
    """

    def __init__(self, n_channels: int, window_standardization: bool = True):
        super().__init__()
        self.n_channels = int(n_channels)
        self.window_standardization = bool(window_standardization)

        # -------------------------------------------------------------
        # Branch A: Slow MRCP Subnet (Large temporal receptive fields)
        # Note: We do NOT square activations here; polarity and slope matter!
        # -------------------------------------------------------------
        self.mrcp_temporal = nn.ModuleList([
            # Kernel lengths 65 and 129 (~250ms and 500ms at 256 Hz)
            nn.Conv2d(1, 4, (1, width), padding=(0, width // 2), bias=False)
            for width in (65, 129)
        ])
        mrcp_temp_ch = 4 * len(self.mrcp_temporal)  # 8

        self.mrcp_spatial = nn.Sequential(
            nn.BatchNorm2d(mrcp_temp_ch),
            nn.Conv2d(mrcp_temp_ch, 16, (self.n_channels, 1), groups=mrcp_temp_ch, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
        )

        self.mrcp_pool = nn.AvgPool1d(16)
        self.mrcp_context = nn.Sequential(
            nn.Conv1d(16, 16, 5, padding=2, bias=False),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.Conv1d(16, 16, 5, dilation=2, padding=4, bias=False),
            nn.BatchNorm1d(16),
            nn.ELU(),
        )

        # -------------------------------------------------------------
        # Branch B: SMR Subnet (Oscillatory Power Envelopes)
        # -------------------------------------------------------------
        self.smr_temporal = nn.ModuleList([
            nn.Conv2d(1, 4, (1, width), padding=(0, width // 2), bias=False)
            for width in (17, 33, 65)
        ])
        smr_temp_ch = 4 * len(self.smr_temporal)  # 12

        self.smr_spatial = nn.Sequential(
            nn.BatchNorm2d(smr_temp_ch),
            nn.Conv2d(smr_temp_ch, 24, (self.n_channels, 1), groups=smr_temp_ch, bias=False),
            nn.BatchNorm2d(24),
            nn.ELU(),
        )

        self.smr_pool = nn.AvgPool1d(16)
        self.smr_context = nn.Sequential(
            nn.Conv1d(24, 24, 5, padding=2, bias=False),
            nn.BatchNorm1d(24),
            nn.ELU(),
            nn.Conv1d(24, 24, 5, dilation=2, padding=4, bias=False),
            nn.BatchNorm1d(24),
            nn.ELU(),
        )

        # -------------------------------------------------------------
        # Fusion and Directional Head: Volar Grasp vs. Finger Extension
        # -------------------------------------------------------------
        self.pool_mrcp = nn.AdaptiveAvgPool1d(2)  # 16 * 2 = 32
        self.pool_smr = nn.AdaptiveAvgPool1d(2)   # 24 * 2 = 48

        # 32 (MRCP) + 48 (SMR) = 80 combined
        self.fusion = nn.Sequential(
            nn.Linear(80, 32),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(0.40),
            nn.Linear(32, 2),  # [logit_grasp, logit_extension]
        )

    def forward(self, x_smr: torch.Tensor, x_mrcp: torch.Tensor) -> torch.Tensor:
        """Forward pass for directional discrimination.

        Parameters:
            x_smr: (B, C, T) tensor in SMR band (4-40 Hz).
            x_mrcp: (B, C, T) tensor in MRCP band (0.3-3.0 Hz).
        Returns:
            logits: (B, 2) where index 0 is Volar Grasp, index 1 is Finger Extension.
        """
        if self.window_standardization:
            # Standardize SMR
            x_smr = x_smr - x_smr.mean(dim=1, keepdim=True)
            scale_smr = x_smr.square().mean(dim=2).sqrt().median(dim=1).values.clamp_min(1e-6)
            x_smr = x_smr / scale_smr[:, None, None]

            # Standardize MRCP preserving waveform baseline
            x_mrcp = x_mrcp - x_mrcp.mean(dim=2, keepdim=True)
            scale_mrcp = x_mrcp.square().mean(dim=2).sqrt().median(dim=1).values.clamp_min(1e-6)
            x_mrcp = x_mrcp / scale_mrcp[:, None, None]

        # 1. MRCP processing (raw amplitude & slope preserved)
        mrcp_4d = x_mrcp.unsqueeze(1)
        mrcp_t = torch.cat([b(mrcp_4d) for b in self.mrcp_temporal], dim=1)
        mrcp_s = self.mrcp_spatial(mrcp_t).squeeze(2)
        mrcp_pooled = self.mrcp_pool(mrcp_s)
        mrcp_feat = mrcp_pooled + self.mrcp_context(mrcp_pooled)
        mrcp_vec = self.pool_mrcp(mrcp_feat).flatten(1)  # (B, 32)

        # 2. SMR processing (power envelope)
        smr_4d = x_smr.unsqueeze(1)
        smr_t = torch.cat([b(smr_4d) for b in self.smr_temporal], dim=1)
        smr_s = self.smr_spatial(smr_t).squeeze(2)
        smr_pow = torch.log1p(self.smr_pool(smr_s.square()))
        smr_feat = smr_pow + self.smr_context(smr_pow)
        smr_vec = self.pool_smr(smr_feat).flatten(1)  # (B, 32)

        # 3. Multimodal fusion
        combined = torch.cat([mrcp_vec, smr_vec], dim=1)  # (B, 64)
        return self.fusion(combined)


# =====================================================================
# 4. Hierarchical Two-Stage Classifier (Combined Architecture)
# =====================================================================

class HierarchicalTwoStageEEGNet(nn.Module):
    """End-to-End Two-Stage Hierarchical BCI Classifier.

    Stage 1: Movement Gating (Rest vs. Intentional Movement) via SMR (4-40 Hz).
    Stage 2: Directional Discrimination (Volar Grasp vs. Finger Extension)
             via Dual-Band MRCP (0.3-3 Hz) + SMR (4-40 Hz).

    Produces calibrated 3-class probability distribution:
        P(rest) = P_gate(rest)
        P(volar_grasp) = P_gate(movement) * P_dir(volar_grasp | movement)
        P(finger_extension) = P_gate(movement) * P_dir(finger_extension | movement)
    """

    def __init__(self, n_channels: int, classes: Optional[List[str]] = None,
                 window_standardization: bool = True, sfreq: float = 256.0):
        super().__init__()
        self.n_channels = int(n_channels)
        self.classes = list(classes or THREE_CLASSES)
        self.window_standardization = bool(window_standardization)
        self.sfreq = float(sfreq)

        self.stage1_gating = Stage1GatingNetwork(
            n_channels=self.n_channels,
            window_standardization=self.window_standardization
        )
        self.stage2_directional = Stage2DirectionalNetwork(
            n_channels=self.n_channels,
            window_standardization=self.window_standardization
        )

        # Internal causal dual-band filter for single-stream raw EEG inputs
        self.dual_filter = DualBandFilter(sfreq=self.sfreq)

    def forward(self, x_smr: torch.Tensor, x_mrcp: Optional[torch.Tensor] = None,
                prefiltered: bool = False) -> torch.Tensor:
        """Compute joint 3-class log-probabilities.

        Parameters:
            x_smr:
                - If x_mrcp is provided: (B, C, T) tensor in SMR band.
                - If x_mrcp is None and x_smr has 2*C channels: stacked [MRCP; SMR] (B, 2*C, T).
                - If x_mrcp is None and x_smr has C channels: raw EEG (causally filtered).
            x_mrcp: optional (B, C, T) tensor in MRCP band.
            prefiltered: whether inputs are already pre-filtered into dual bands.

        Returns:
            log_probs: (B, 3) log-probabilities [log P(rest), log P(grasp), log P(ext)].
        """
        x_smr, x_mrcp = self._resolve_inputs(x_smr, x_mrcp, prefiltered=prefiltered)

        logits_gate = self.stage1_gating(x_smr)                    # (B, 2)
        logits_dir = self.stage2_directional(x_smr, x_mrcp)        # (B, 2)

        probs_gate = F.softmax(logits_gate, dim=1)                 # [P(rest), P(movement)]
        probs_dir = F.softmax(logits_dir, dim=1)                   # [P(grasp|move), P(ext|move)]

        p_rest = probs_gate[:, 0:1]
        p_move = probs_gate[:, 1:2]

        p_grasp = p_move * probs_dir[:, 0:1]
        p_extension = p_move * probs_dir[:, 1:2]

        probs_3class = torch.cat([p_rest, p_grasp, p_extension], dim=1)  # (B, 3)
        return torch.log(probs_3class.clamp_min(1e-12))

    def predict_proba(self, x_smr: Union[np.ndarray, torch.Tensor],
                      x_mrcp: Optional[Union[np.ndarray, torch.Tensor]] = None,
                      prefiltered: bool = False) -> np.ndarray:
        """Compute calibrated probabilities over [rest, volar_grasp, finger_extension]."""
        self.eval()
        with torch.no_grad():
            if not isinstance(x_smr, torch.Tensor):
                x_smr = torch.from_numpy(np.asarray(x_smr, dtype=np.float32))
            if x_mrcp is not None and not isinstance(x_mrcp, torch.Tensor):
                x_mrcp = torch.from_numpy(np.asarray(x_mrcp, dtype=np.float32))

            log_p = self.forward(x_smr, x_mrcp, prefiltered=prefiltered)
            return torch.exp(log_p).cpu().numpy()

    def predict(self, x_smr: Union[np.ndarray, torch.Tensor],
                x_mrcp: Optional[Union[np.ndarray, torch.Tensor]] = None,
                gate_threshold: float = 0.50,
                prefiltered: bool = False) -> np.ndarray:
        """Predict class index 0 (rest), 1 (volar_grasp), or 2 (finger_extension).

        Applies strict gating: if P(movement) < gate_threshold, outputs Rest (0).
        """
        probs = self.predict_proba(x_smr, x_mrcp, prefiltered=prefiltered)
        p_movement = probs[:, 1] + probs[:, 2]

        preds = np.zeros(len(probs), dtype=int)
        for i in range(len(probs)):
            if p_movement[i] < gate_threshold:
                preds[i] = 0
            else:
                # Discriminate grasp (1) vs extension (2)
                preds[i] = 1 if probs[i, 1] >= probs[i, 2] else 2
        return preds

    def predict_step(self, x_smr_window: np.ndarray,
                     x_mrcp_window: Optional[np.ndarray] = None,
                     gate_threshold: float = 0.50,
                     min_margin: float = 0.05,
                     prefiltered: bool = False) -> Dict[str, Union[int, float, str, list]]:
        """Single-step inference for real-time BCI engines and decoders.

        Returns decision dictionary with fine-grained gating diagnostics.
        """
        self.eval()
        with torch.no_grad():
            if x_smr_window.ndim == 2:
                x_smr_window = x_smr_window[None, ...]
            if x_mrcp_window is not None and x_mrcp_window.ndim == 2:
                x_mrcp_window = x_mrcp_window[None, ...]

            probs = self.predict_proba(x_smr_window, x_mrcp_window, prefiltered=prefiltered)[0]

        p_rest = float(probs[0])
        p_grasp = float(probs[1])
        p_ext = float(probs[2])
        p_move = float(p_grasp + p_ext)

        # Conditional directional probabilities
        cond_grasp = float(p_grasp / max(1e-12, p_move))
        cond_ext = float(p_ext / max(1e-12, p_move))

        if p_move < gate_threshold:
            decision = 0
            decision_name = self.classes[0]
            confidence = p_rest
            margin = float(p_rest - p_move)
            gated = True
        else:
            gated = False
            if p_grasp >= p_ext:
                decision = 1
                decision_name = self.classes[1]
                confidence = p_grasp
                margin = float(cond_grasp - cond_ext)
            else:
                decision = 2
                decision_name = self.classes[2]
                confidence = p_ext
                margin = float(cond_ext - cond_grasp)

            if margin < min_margin:
                decision = -1
                decision_name = 'uncertain'

        return {
            'class_id': int(decision),
            'class_name': str(decision_name),
            'confidence': float(confidence),
            'margin': float(margin),
            'probabilities': [p_rest, p_grasp, p_ext],
            'p_gate_movement': p_move,
            'p_gate_rest': p_rest,
            'p_dir_grasp': cond_grasp,
            'p_dir_extension': cond_ext,
            'gated': gated,
        }

    def _resolve_inputs(self, x_smr: torch.Tensor,
                        x_mrcp: Optional[torch.Tensor] = None,
                        prefiltered: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resolves inputs whether provided as separate tensors, stacked channels, or raw EEG."""
        device = x_smr.device
        if x_mrcp is not None:
            return x_smr, x_mrcp

        # Case 1: Stacked channels (B, 2*C, T) -> [MRCP; SMR]
        if x_smr.shape[1] == 2 * self.n_channels:
            mrcp_part = x_smr[:, :self.n_channels, :]
            smr_part = x_smr[:, self.n_channels:, :]
            return smr_part, mrcp_part

        # Case 2: Raw broadband EEG (B, C, T) -> filter causally using internal dual filter
        if x_smr.shape[1] == self.n_channels:
            if prefiltered:
                # If caller asserts prefiltered with single band, default mrcp to smr for safety
                return x_smr, x_smr
            x_np = x_smr.detach().cpu().numpy()
            mrcp_batch = np.empty_like(x_np)
            smr_batch = np.empty_like(x_np)
            for i in range(len(x_np)):
                m, s, _ = self.dual_filter.filter_continuous(x_np[i], initial_dc_correction=True)
                mrcp_batch[i] = m
                smr_batch[i] = s
            return (
                torch.from_numpy(smr_batch).to(device, dtype=torch.float32),
                torch.from_numpy(mrcp_batch).to(device, dtype=torch.float32),
            )

        raise ValueError(
            f"Input shape {x_smr.shape} does not match expected n_channels={self.n_channels} "
            f"or stacked 2*n_channels={2 * self.n_channels}"
        )


# =====================================================================
# 5. Ensemble of Hierarchical Classifiers
# =====================================================================

class HierarchicalEnsemble(nn.ModuleList):
    """Ensemble of HierarchicalTwoStageEEGNet models for robust multi-checkpoint decoding."""

    def __init__(self, n_channels: int, member_states: List[Dict],
                 classes: Optional[List[str]] = None, window_standardization: bool = True):
        super().__init__([
            HierarchicalTwoStageEEGNet(n_channels, classes=classes, window_standardization=window_standardization)
            for _ in member_states
        ])
        for model, state in zip(self, member_states):
            model.load_state_dict(state)
        self.classes = list(classes or THREE_CLASSES)

    def forward(self, x_smr: torch.Tensor, x_mrcp: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Ensemble log-probabilities by averaging member probability outputs."""
        all_probs = torch.stack([torch.exp(m(x_smr, x_mrcp)) for m in self], dim=0)
        mean_probs = all_probs.mean(dim=0).clamp_min(1e-12)
        return torch.log(mean_probs)

    def predict_proba(self, x_smr: Union[np.ndarray, torch.Tensor],
                      x_mrcp: Optional[Union[np.ndarray, torch.Tensor]] = None) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            if not isinstance(x_smr, torch.Tensor):
                x_smr = torch.from_numpy(np.asarray(x_smr, dtype=np.float32))
            if x_mrcp is not None and not isinstance(x_mrcp, torch.Tensor):
                x_mrcp = torch.from_numpy(np.asarray(x_mrcp, dtype=np.float32))
            log_p = self.forward(x_smr, x_mrcp)
            return torch.exp(log_p).cpu().numpy()


# =====================================================================
# 6. Training & Evaluation Routines
# =====================================================================

def compute_hierarchical_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray,
                                 cue_ids: Optional[List] = None,
                                 class_names: Optional[List[str]] = None,
                                 gate_threshold: float = 0.50) -> Dict:
    """Compute comprehensive window-level and cue-level metrics for hierarchical models.

    Specifically evaluates:
    - Balanced accuracy and Macro-F1 across all 3 classes
    - Finger Extension recall (targeting recovery from 0/20)
    - Volar Grasp recall
    - Rest recall and False Activation Rate during rest (fraction of true rest windows/cues misclassified as movement, target < 3 per minute)
    - 3x3 Confusion matrix
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_prob = np.asarray(y_pred_prob, dtype=np.float32)
    n_classes = len(class_names or THREE_CLASSES)
    class_names = list(class_names or THREE_CLASSES)

    # Window predictions
    win_p_move = y_pred_prob[:, 1] + y_pred_prob[:, 2]
    win_preds = np.zeros(len(y_true), dtype=int)
    for i in range(len(y_true)):
        if win_p_move[i] < gate_threshold:
            win_preds[i] = 0
        else:
            win_preds[i] = 1 if y_pred_prob[i, 1] >= y_pred_prob[i, 2] else 2

    win_ba = float(balanced_accuracy_score(y_true, win_preds))
    win_f1 = float(f1_score(y_true, win_preds, average='macro', zero_division=0))
    win_cm = confusion_matrix(y_true, win_preds, labels=list(range(n_classes))).tolist()

    # Per-class recalls
    recalls = recall_score(y_true, win_preds, average=None, labels=list(range(n_classes)), zero_division=0)
    rest_recall = float(recalls[0]) if len(recalls) > 0 else 0.0
    grasp_recall = float(recalls[1]) if len(recalls) > 1 else 0.0
    ext_recall = float(recalls[2]) if len(recalls) > 2 else 0.0

    # False activation rate during rest: fraction of true rest trials misclassified as movement
    rest_mask = (y_true == 0)
    rest_false_activations = int(np.sum((win_preds[rest_mask] > 0))) if np.any(rest_mask) else 0
    rest_total = int(np.sum(rest_mask))
    rest_false_rate = float(rest_false_activations / max(1, rest_total))

    metrics = {
        'window': {
            'balanced_accuracy': win_ba,
            'macro_f1': win_f1,
            'rest_recall': rest_recall,
            'volar_grasp_recall': grasp_recall,
            'finger_extension_recall': ext_recall,
            'rest_false_activation_rate': rest_false_rate,
            'confusion_matrix': win_cm,
            'n_windows': len(y_true),
        }
    }

    # Cue-level metrics
    if cue_ids is not None and len(cue_ids) == len(y_true):
        cues_dict = defaultdict(list)
        cue_labels = {}
        for c_id, p, y in zip(cue_ids, y_pred_prob, y_true):
            cues_dict[c_id].append(p)
            cue_labels[c_id] = y

        cue_trues = []
        cue_preds = []
        for c_id, prob_list in cues_dict.items():
            mean_p = np.mean(prob_list, axis=0)
            p_move = mean_p[1] + mean_p[2]
            if p_move < gate_threshold:
                pred = 0
            else:
                pred = 1 if mean_p[1] >= mean_p[2] else 2
            cue_trues.append(cue_labels[c_id])
            cue_preds.append(pred)

        cue_trues = np.array(cue_trues)
        cue_preds = np.array(cue_preds)

        cue_ba = float(balanced_accuracy_score(cue_trues, cue_preds))
        cue_f1 = float(f1_score(cue_trues, cue_preds, average='macro', zero_division=0))
        cue_cm = confusion_matrix(cue_trues, cue_preds, labels=list(range(n_classes))).tolist()
        c_recalls = recall_score(cue_trues, cue_preds, average=None, labels=list(range(n_classes)), zero_division=0)

        c_rest_recall = float(c_recalls[0]) if len(c_recalls) > 0 else 0.0
        c_grasp_recall = float(c_recalls[1]) if len(c_recalls) > 1 else 0.0
        c_ext_recall = float(c_recalls[2]) if len(c_recalls) > 2 else 0.0

        c_rest_mask = (cue_trues == 0)
        c_false_acts = int(np.sum(cue_preds[c_rest_mask] > 0)) if np.any(c_rest_mask) else 0
        c_rest_total = int(np.sum(c_rest_mask))

        metrics['cue'] = {
            'balanced_accuracy': cue_ba,
            'macro_f1': cue_f1,
            'rest_recall': c_rest_recall,
            'volar_grasp_recall': c_grasp_recall,
            'finger_extension_recall': c_ext_recall,
            'rest_false_activation_rate': float(c_false_acts / max(1, c_rest_total)),
            'confusion_matrix': cue_cm,
            'n_cues': len(cue_trues),
        }

    return metrics


def train_hierarchical_two_stage(
    model: HierarchicalTwoStageEEGNet,
    train_x_smr: np.ndarray,
    train_x_mrcp: np.ndarray,
    train_y: np.ndarray,
    val_x_smr: np.ndarray,
    val_x_mrcp: np.ndarray,
    val_y: np.ndarray,
    val_cue_ids: Optional[List] = None,
    epochs: int = 35,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    patience: int = 10,
    batch_size: int = 32,
    device: str = 'cpu',
) -> Tuple[HierarchicalTwoStageEEGNet, float, List[Dict]]:
    """Train HierarchicalTwoStageEEGNet with decoupled stage-wise and joint losses.

    Stage 1 Loss: Binary Cross-Entropy on Rest (0) vs Movement (1, pooling grasp + extension).
    Stage 2 Loss: Directional Binary Cross-Entropy on Movement trials (Grasp: 0 vs Extension: 1).
    Total Loss: Loss_stage1 + lambda_dir * Loss_stage2.
    """
    model = model.to(device)

    # Convert labels
    # 3-class: 0 = rest, 1 = grasp, 2 = extension
    # Stage 1: 0 = rest, 1 = movement (1 or 2)
    # Stage 2: 0 = grasp (from 1), 1 = extension (from 2), -1 = ignore (from 0)
    y_gate_train = (train_y > 0).astype(int)
    y_gate_val = (val_y > 0).astype(int)

    y_dir_train = np.where(train_y == 1, 0, np.where(train_y == 2, 1, -1))
    y_dir_val = np.where(val_y == 1, 0, np.where(val_y == 2, 1, -1))

    # PyTorch Datasets
    train_ds = TensorDataset(
        torch.from_numpy(train_x_smr).float(),
        torch.from_numpy(train_x_mrcp).float(),
        torch.from_numpy(train_y).long(),
        torch.from_numpy(y_gate_train).long(),
        torch.from_numpy(y_dir_train).long(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(val_x_smr).float(),
        torch.from_numpy(val_x_mrcp).float(),
        torch.from_numpy(val_y).long(),
        torch.from_numpy(y_gate_val).long(),
        torch.from_numpy(y_dir_val).long(),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Class weighting for gating
    gate_counts = Counter(y_gate_train)
    total_g = len(y_gate_train)
    gate_w = [total_g / (2.0 * max(1, gate_counts[c])) for c in (0, 1)]
    loss_gate_fn = nn.CrossEntropyLoss(weight=torch.tensor(gate_w, dtype=torch.float32, device=device))

    # Class weighting for directional discrimination (movement only)
    dir_mask = (y_dir_train >= 0)
    dir_counts = Counter(y_dir_train[dir_mask])
    total_d = max(1, int(np.sum(dir_mask)))
    dir_w = [total_d / (2.0 * max(1, dir_counts.get(c, 1))) for c in (0, 1)]
    loss_dir_fn = nn.CrossEntropyLoss(weight=torch.tensor(dir_w, dtype=torch.float32, device=device), ignore_index=-1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_ba = -1.0
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for b_smr, b_mrcp, _, b_gate, b_dir in train_loader:
            b_smr = b_smr.to(device)
            b_mrcp = b_mrcp.to(device)
            b_gate = b_gate.to(device)
            b_dir = b_dir.to(device)

            optimizer.zero_grad()
            logits_gate = model.stage1_gating(b_smr)
            logits_dir = model.stage2_directional(b_smr, b_mrcp)

            loss_g = loss_gate_fn(logits_gate, b_gate)
            loss_d = loss_dir_fn(logits_dir, b_dir) if torch.any(b_dir >= 0) else torch.tensor(0.0, device=device)

            loss = loss_g + 1.2 * loss_d
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        train_loss = float(np.mean(train_losses))

        # Validation evaluation
        model.eval()
        val_probs = []
        with torch.no_grad():
            for b_smr, b_mrcp, _, _, _ in val_loader:
                b_smr = b_smr.to(device)
                b_mrcp = b_mrcp.to(device)
                log_p = model(b_smr, b_mrcp)
                val_probs.append(torch.exp(log_p).cpu().numpy())

        cat_val_probs = np.concatenate(val_probs, axis=0) if val_probs else np.empty((0, 3))
        metrics = compute_hierarchical_metrics(val_y, cat_val_probs, cue_ids=val_cue_ids)
        val_ba = metrics.get('cue', {}).get('balanced_accuracy', metrics['window']['balanced_accuracy'])
        ext_recall = metrics.get('cue', {}).get('finger_extension_recall', metrics['window']['finger_extension_recall'])
        grasp_recall = metrics.get('cue', {}).get('volar_grasp_recall', metrics['window']['volar_grasp_recall'])

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_balanced_accuracy': float(val_ba),
            'val_finger_extension_recall': float(ext_recall),
            'val_volar_grasp_recall': float(grasp_recall),
            'window_metrics': metrics['window'],
            'cue_metrics': metrics.get('cue'),
        })

        if val_ba > best_val_ba:
            best_val_ba = val_ba
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_ba, history


def finetune_hierarchical_personal(
    pretrained_model: HierarchicalTwoStageEEGNet,
    calib_train_raw: np.ndarray,
    calib_train_y: np.ndarray,
    calib_train_cues: List,
    calib_val_raw: np.ndarray,
    calib_val_y: np.ndarray,
    calib_val_cues: List,
    dual_filter: Optional[DualBandFilter] = None,
    epochs: int = 25,
    lr_head: float = 1e-3,
    lr_backbone: float = 1e-4,
    freeze_backbone_epochs: int = 3,
    patience: int = 8,
    batch_size: int = 16,
    device: str = 'cpu',
    prefiltered: bool = False,
    n_base_channels: Optional[int] = None,
) -> Tuple[HierarchicalTwoStageEEGNet, UnsupervisedRestingAlignment, Dict]:
    """Personal Calibration Fine-Tuning Routine for HierarchicalTwoStageEEGNet.

    Strict Leakage-Free Protocol:
    1. Fit UnsupervisedRestingAlignment (Sion An et al.) exclusively on the wearer's
       resting trials (calib_train_y == 0) from calib_train_raw.
    2. Whitens spatial covariance of both calib_train and calib_val per band (C x C).
    3. Causally extracts dual-band MRCP and SMR representations with filter state
       continuity across stream chunks (no isolated-window startup transient).
    4. Fine-tunes classification heads with early stopping on whole-cue validation
       accuracy (monitoring both Finger Extension and Volar Grasp recall).
    """
    filt = dual_filter or DualBandFilter()
    calib_train_raw = np.asarray(calib_train_raw, dtype=np.float32)
    calib_val_raw = np.asarray(calib_val_raw, dtype=np.float32)
    calib_train_y = np.asarray(calib_train_y, dtype=int)
    calib_val_y = np.asarray(calib_val_y, dtype=int)

    model_ch = int(pretrained_model.n_channels)
    if prefiltered or calib_train_raw.shape[1] == 2 * model_ch:
        is_stacked = True
        n_ch = model_ch
    elif calib_train_raw.shape[1] == model_ch:
        is_stacked = False
        n_ch = model_ch
    else:
        raise ValueError(
            f"calib_train_raw shape {calib_train_raw.shape} channels {calib_train_raw.shape[1]} "
            f"does not match model n_channels={model_ch} or stacked 2*n_channels={2 * model_ch}"
        )

    if is_stacked:
        # 1. Fit per-band C x C resting alignment on calib_train rest trials
        ura = UnsupervisedRestingAlignment(shrinkage=0.05, n_base_channels=n_ch).fit_from_dataset(
            calib_train_raw, calib_train_y, rest_label=0, n_base_channels=n_ch
        )
        aligned_train = ura.transform(calib_train_raw)
        aligned_val = ura.transform(calib_val_raw)

        train_mrcp = aligned_train[:, :n_ch, :]
        train_smr = aligned_train[:, n_ch:, :]
        val_mrcp = aligned_val[:, :n_ch, :]
        val_smr = aligned_val[:, n_ch:, :]
    else:
        n_ch = calib_train_raw.shape[1]
        n_train, _, n_times = calib_train_raw.shape
        n_val = calib_val_raw.shape[0]

        # To prevent isolated 2s window 0.3 Hz Butterworth startup transients,
        # concatenate windows with a settling pad and causal filter continuity
        train_flat = calib_train_raw.transpose(1, 0, 2).reshape(n_ch, -1)
        pad_len = min(n_times, train_flat.shape[1])
        train_padded = np.concatenate([train_flat[:, :pad_len], train_flat], axis=1)
        mrcp_c, smr_c, _ = filt.filter_continuous(train_padded, initial_dc_correction=True)
        mrcp_tr = mrcp_c[:, pad_len:].reshape(n_ch, n_train, n_times).transpose(1, 0, 2).astype(np.float32)
        smr_tr = smr_c[:, pad_len:].reshape(n_ch, n_train, n_times).transpose(1, 0, 2).astype(np.float32)

        val_flat = calib_val_raw.transpose(1, 0, 2).reshape(n_ch, -1)
        val_pad = min(n_times, val_flat.shape[1])
        val_padded = np.concatenate([val_flat[:, :val_pad], val_flat], axis=1)
        mrcp_vc, smr_vc, _ = filt.filter_continuous(val_padded, initial_dc_correction=True)
        mrcp_va = mrcp_vc[:, val_pad:].reshape(n_ch, n_val, n_times).transpose(1, 0, 2).astype(np.float32)
        smr_va = smr_vc[:, val_pad:].reshape(n_ch, n_val, n_times).transpose(1, 0, 2).astype(np.float32)

        stacked_tr = np.concatenate([mrcp_tr, smr_tr], axis=1)
        stacked_va = np.concatenate([mrcp_va, smr_va], axis=1)

        ura = UnsupervisedRestingAlignment(shrinkage=0.05, n_base_channels=n_ch).fit_from_dataset(
            stacked_tr, calib_train_y, rest_label=0, n_base_channels=n_ch
        )
        aligned_tr = ura.transform(stacked_tr)
        aligned_va = ura.transform(stacked_va)

        train_mrcp = aligned_tr[:, :n_ch, :]
        train_smr = aligned_tr[:, n_ch:, :]
        val_mrcp = aligned_va[:, :n_ch, :]
        val_smr = aligned_va[:, n_ch:, :]

    # 4. Fine-tune model
    model = copy.deepcopy(pretrained_model).to(device)

    # Parameter groups
    head_params = list(model.stage1_gating.head.parameters()) + list(model.stage2_directional.fusion.parameters())
    backbone_params = [
        p for n, p in model.named_parameters()
        if not n.startswith('stage1_gating.head') and not n.startswith('stage2_directional.fusion')
    ]

    optimizer = torch.optim.AdamW([
        {'params': head_params, 'lr': lr_head},
        {'params': backbone_params, 'lr': 0.0 if freeze_backbone_epochs > 0 else lr_backbone},
    ], weight_decay=1e-2)

    # Gating and directional targets
    y_gate_tr = (calib_train_y > 0).astype(int)
    y_dir_tr = np.where(calib_train_y == 1, 0, np.where(calib_train_y == 2, 1, -1))

    train_ds = TensorDataset(
        torch.from_numpy(train_smr).float(),
        torch.from_numpy(train_mrcp).float(),
        torch.from_numpy(calib_train_y).long(),
        torch.from_numpy(y_gate_tr).long(),
        torch.from_numpy(y_dir_tr).long(),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    loss_gate_fn = nn.CrossEntropyLoss()
    loss_dir_fn = nn.CrossEntropyLoss(ignore_index=-1)

    best_ba = -1.0
    best_state = None
    patience_cnt = 0
    history = []

    for epoch in range(1, epochs + 1):
        if epoch == freeze_backbone_epochs + 1 and freeze_backbone_epochs > 0:
            optimizer.param_groups[1]['lr'] = lr_backbone

        model.train()
        for b_smr, b_mrcp, _, b_gate, b_dir in train_loader:
            b_smr = b_smr.to(device)
            b_mrcp = b_mrcp.to(device)
            b_gate = b_gate.to(device)
            b_dir = b_dir.to(device)

            optimizer.zero_grad()
            lg = model.stage1_gating(b_smr)
            ld = model.stage2_directional(b_smr, b_mrcp)

            loss = loss_gate_fn(lg, b_gate)
            if torch.any(b_dir >= 0):
                loss = loss + 1.2 * loss_dir_fn(ld, b_dir)

            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            v_smr_t = torch.from_numpy(val_smr).float().to(device)
            v_mrcp_t = torch.from_numpy(val_mrcp).float().to(device)
            log_p = model(v_smr_t, v_mrcp_t)
            v_probs = torch.exp(log_p).cpu().numpy()

        val_metrics = compute_hierarchical_metrics(calib_val_y, v_probs, cue_ids=calib_val_cues)
        cue_ba = val_metrics.get('cue', {}).get('balanced_accuracy', val_metrics['window']['balanced_accuracy'])

        history.append({
            'epoch': epoch,
            'val_cue_ba': cue_ba,
            'val_metrics': val_metrics,
        })

        if cue_ba > best_ba:
            best_ba = cue_ba
            best_state = copy.deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    diagnostics = {
        'best_val_balanced_accuracy': float(best_ba),
        'epochs_trained': len(history),
        'history': history,
        'resting_alignment_samples': ura.n_samples_fitted_,
    }
    return model, ura, diagnostics
