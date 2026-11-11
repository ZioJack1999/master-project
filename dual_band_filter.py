"""Causal Dual-Band EEG Filter and Feature Extraction Module.

Designed specifically for Slow Movement-Related Cortical Potentials (MRCP)
and Sensorimotor Rhythms (SMR) operating directly on raw pre-filtered X24 EEG
streamed and saved in Volts.

Architecture:
- Branch A (MRCP): Slow movement-related cortical potentials (0.3–3.0 Hz / 0.5–4.0 Hz),
  2nd or 4th order Butterworth SOS causal filter with state-preserving sosfilt_zi
  and DC step-transient cancellation.
- Branch B (SMR): Sensorimotor rhythms (4.0–40.0 Hz / 8.0–30.0 Hz),
  4th order Butterworth SOS causal filter with state-preserving sosfilt_zi.
- Causal Streaming Buffer: DualBandRingBuffer manages continuous raw streaming,
  synchronized multi-branch state preservation, and artifact tracking.
- Feature Extraction:
  * Temporal MRCP: negative peak amplitude (Bereitschaftspotential), peak latency,
    slope (full-window & late-phase intention building), AUC, downsampled trajectory,
    and spatial motor gradients (C3, Cz, C4).
  * Spectral SMR: log band powers (theta, mu, beta_low, beta_high, gamma),
    relative powers, and contralateral laterality indices.
  * Combined feature vector and deep learning dual-band tensor preparation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


# Default configurations
DEFAULT_SFREQ = 256.0
DEFAULT_MRCP_BAND = (0.3, 3.0)
DEFAULT_MRCP_ORDER = 2
DEFAULT_SMR_BAND = (4.0, 40.0)
DEFAULT_SMR_ORDER = 4

# Standard Motor Montages
CORE_MOTOR_CHANNELS = ('C3', 'Cz', 'C4')
EXTENDED_MOTOR_CHANNELS = ('F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4')

SMR_BANDS = {
    'theta': (4.0, 8.0),
    'mu': (8.0, 13.0),
    'beta_low': (13.0, 20.0),
    'beta_high': (20.0, 30.0),
    'gamma_low': (30.0, 40.0),
}


class RingBuffer:
    """Channels x samples ring buffer for continuous EEG streaming."""

    def __init__(self, n_ch: int, n_samples: int):
        self.n_ch = int(n_ch)
        self.n_samples = int(n_samples)
        self.buf = np.zeros((self.n_ch, self.n_samples), dtype=np.float32)
        self.idx = 0

    def push_chunk(self, chunk: np.ndarray) -> None:
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[0] != self.n_ch:
            raise ValueError(f"Expected channels x samples ({self.n_ch} channels), got {chunk.shape}")
        n = chunk.shape[1]
        if n >= self.n_samples:
            self.idx += n
            self.buf[:] = np.roll(chunk[:, -self.n_samples:], self.idx % self.n_samples, axis=1)
            return
        i = self.idx % self.n_samples
        first = min(n, self.n_samples - i)
        self.buf[:, i:i + first] = chunk[:, :first]
        self.buf[:, :n - first] = chunk[:, first:]
        self.idx += n

    def push(self, sample: np.ndarray) -> None:
        self.push_chunk(np.asarray(sample)[:, None])

    def get(self) -> np.ndarray:
        i = self.idx % self.n_samples
        return np.concatenate((self.buf[:, i:], self.buf[:, :i]), axis=1)

    def reset(self) -> None:
        self.buf.fill(0.0)
        self.idx = 0

    @property
    def is_full(self) -> bool:
        return self.idx >= self.n_samples


class DualBandFilterState:
    """Preserves causal filter states (zi) across streaming chunks or segments."""

    def __init__(self, sos_mrcp: np.ndarray, sos_smr: np.ndarray, n_channels: int,
                 initial_sample: Optional[np.ndarray] = None):
        self.sos_mrcp = np.asarray(sos_mrcp, dtype=np.float64)
        self.sos_smr = np.asarray(sos_smr, dtype=np.float64)
        self.n_channels = int(n_channels)
        self.base_zi_mrcp = sosfilt_zi(self.sos_mrcp)  # (n_sections_mrcp, 2)
        self.base_zi_smr = sosfilt_zi(self.sos_smr)    # (n_sections_smr, 2)

        self.zi_mrcp = np.zeros((len(self.sos_mrcp), self.n_channels, 2), dtype=np.float64)
        self.zi_smr = np.zeros((len(self.sos_smr), self.n_channels, 2), dtype=np.float64)
        if initial_sample is not None:
            self.reset(initial_sample)

    def reset(self, initial_sample: Optional[np.ndarray] = None) -> None:
        """Reset filter states.

        When initial_sample is provided (shape (n_channels,) or (n_channels, 1)),
        initializes zi with steady-state response scaled by initial_sample.
        This completely eliminates initial step response transients and DC offset ringing.
        """
        if initial_sample is not None:
            init = np.asarray(initial_sample, dtype=np.float64).reshape(self.n_channels)
            # zi shape: (n_sections, n_channels, 2)
            self.zi_mrcp = np.repeat(self.base_zi_mrcp[:, None, :], self.n_channels, axis=1) * init[None, :, None]
            self.zi_smr = np.repeat(self.base_zi_smr[:, None, :], self.n_channels, axis=1) * init[None, :, None]
        else:
            self.zi_mrcp.fill(0.0)
            self.zi_smr.fill(0.0)

    def clone(self) -> 'DualBandFilterState':
        new_state = DualBandFilterState(self.sos_mrcp, self.sos_smr, self.n_channels)
        new_state.zi_mrcp = self.zi_mrcp.copy()
        new_state.zi_smr = self.zi_smr.copy()
        return new_state


class DualBandFilter:
    """Causal Dual-Band Butterworth SOS Filter.

    Branch A: MRCP (0.3–3.0 Hz, 2nd or 4th order SOS)
    Branch B: SMR (4.0–40.0 Hz, 4th order SOS)
    """

    def __init__(self, sfreq: float = DEFAULT_SFREQ,
                 mrcp_band: Tuple[float, float] = DEFAULT_MRCP_BAND,
                 mrcp_order: int = DEFAULT_MRCP_ORDER,
                 smr_band: Tuple[float, float] = DEFAULT_SMR_BAND,
                 smr_order: int = DEFAULT_SMR_ORDER):
        self.sfreq = float(sfreq)
        self.mrcp_band = tuple(float(x) for x in mrcp_band)
        self.mrcp_order = int(mrcp_order)
        self.smr_band = tuple(float(x) for x in smr_band)
        self.smr_order = int(smr_order)

        # Validate bands
        nyquist = self.sfreq / 2.0
        if not (0 < self.mrcp_band[0] < self.mrcp_band[1] < nyquist):
            raise ValueError(f"Invalid MRCP band {self.mrcp_band} for sfreq {self.sfreq}")
        if not (0 < self.smr_band[0] < self.smr_band[1] < nyquist):
            raise ValueError(f"Invalid SMR band {self.smr_band} for sfreq {self.sfreq}")

        self.sos_mrcp = butter(self.mrcp_order, list(self.mrcp_band), btype='bandpass',
                               fs=self.sfreq, output='sos')
        self.sos_smr = butter(self.smr_order, list(self.smr_band), btype='bandpass',
                              fs=self.sfreq, output='sos')

    def create_state(self, n_channels: int, initial_sample: Optional[np.ndarray] = None) -> DualBandFilterState:
        """Create a state-preserving tracker for streaming or continuous blocks."""
        return DualBandFilterState(self.sos_mrcp, self.sos_smr, n_channels, initial_sample)

    def filter_chunk(self, chunk: np.ndarray, state: DualBandFilterState) -> Tuple[np.ndarray, np.ndarray]:
        """Causally filter a single streaming chunk (channels x samples) in Volts.

        Updates state.zi_mrcp and state.zi_smr in-place.
        Returns:
            mrcp_chunk: (n_channels, n_samples) np.float32
            smr_chunk: (n_channels, n_samples) np.float32
        """
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[0] != state.n_channels:
            raise ValueError(f"Expected ({state.n_channels}, samples), got {chunk.shape}")

        mrcp_out, state.zi_mrcp = sosfilt(self.sos_mrcp, chunk, axis=1, zi=state.zi_mrcp)
        smr_out, state.zi_smr = sosfilt(self.sos_smr, chunk, axis=1, zi=state.zi_smr)

        return mrcp_out.astype(np.float32), smr_out.astype(np.float32)

    def filter_continuous(self, raw: np.ndarray, state: Optional[DualBandFilterState] = None,
                          initial_dc_correction: bool = True) -> Tuple[np.ndarray, np.ndarray, DualBandFilterState]:
        """Filter a continuous EEG recording or segment.

        Parameters:
            raw: (n_channels, n_samples) in Volts.
            state: optional existing filter state.
            initial_dc_correction: if state is None, initialize state using first sample
                                   to eliminate step response transient.
        Returns:
            mrcp: (n_channels, n_samples) np.float32
            smr: (n_channels, n_samples) np.float32
            state: updated DualBandFilterState
        """
        raw = np.asarray(raw, dtype=np.float64)
        if raw.ndim != 2:
            raise ValueError(f"Expected 2D array (channels x samples), got {raw.shape}")
        n_ch, n_samples = raw.shape

        if state is None:
            init_val = raw[:, 0] if (initial_dc_correction and n_samples > 0) else None
            state = self.create_state(n_ch, initial_sample=init_val)

        mrcp, smr = self.filter_chunk(raw, state)
        return mrcp, smr, state


class DualBandRingBuffer:
    """Synchronized causal ring buffer holding Raw, MRCP, SMR, and Artifact masks."""

    def __init__(self, n_channels: int, window_samples: int,
                 dual_filter: Optional[DualBandFilter] = None,
                 artifact_limit_v: float = 100e-6,
                 min_signal_std_v: float = 1e-9):
        self.n_channels = int(n_channels)
        self.window_samples = int(window_samples)
        self.dual_filter = dual_filter or DualBandFilter()
        self.artifact_limit_v = float(artifact_limit_v)
        self.min_signal_std_v = float(min_signal_std_v)

        self.raw_buf = RingBuffer(self.n_channels, self.window_samples)
        self.mrcp_buf = RingBuffer(self.n_channels, self.window_samples)
        self.smr_buf = RingBuffer(self.n_channels, self.window_samples)
        self.bad_buf = RingBuffer(1, self.window_samples)

        self.state = self.dual_filter.create_state(self.n_channels)
        self.samples_seen = 0

    def reset(self, initial_sample: Optional[np.ndarray] = None) -> None:
        self.raw_buf.reset()
        self.mrcp_buf.reset()
        self.smr_buf.reset()
        self.bad_buf.reset()
        self.state.reset(initial_sample)
        self.samples_seen = 0

    def push_chunk(self, raw_chunk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Ingest raw pre-filtered EEG chunk, filter causally, update buffers.

        Returns (mrcp_chunk, smr_chunk).
        """
        raw_chunk = np.asarray(raw_chunk, dtype=np.float32)
        if raw_chunk.ndim != 2 or raw_chunk.shape[0] != self.n_channels:
            raise ValueError(f"Expected ({self.n_channels}, samples), got {raw_chunk.shape}")

        if self.samples_seen == 0 and raw_chunk.shape[1] > 0:
            # Initialize filter states with steady-state DC level of the first sample
            self.state.reset(raw_chunk[:, 0])

        mrcp_chunk, smr_chunk = self.dual_filter.filter_chunk(raw_chunk, self.state)

        # Artifact detection: either SMR or MRCP exceeding physiological limits
        bad = ((np.abs(smr_chunk) > self.artifact_limit_v).any(axis=0)[None, :] |
               (np.abs(mrcp_chunk) > self.artifact_limit_v * 2.0).any(axis=0)[None, :]).astype(np.float32)

        self.raw_buf.push_chunk(raw_chunk)
        self.mrcp_buf.push_chunk(mrcp_chunk)
        self.smr_buf.push_chunk(smr_chunk)
        self.bad_buf.push_chunk(bad)
        self.samples_seen += raw_chunk.shape[1]

        return mrcp_chunk, smr_chunk

    def get_windows(self) -> Dict[str, Union[np.ndarray, str]]:
        """Retrieve current aligned windows and quality evaluation."""
        raw_win = self.raw_buf.get()
        mrcp_win = self.mrcp_buf.get()
        smr_win = self.smr_buf.get()
        bad_win = self.bad_buf.get()

        if bad_win.any():
            quality = 'artifact'
        elif np.any(np.std(smr_win, axis=1) < self.min_signal_std_v):
            quality = 'flat_signal'
        else:
            quality = 'ok'

        return {
            'raw': raw_win,
            'mrcp': mrcp_win,
            'smr': smr_win,
            'stacked': np.concatenate([mrcp_win, smr_win], axis=0),
            'bad': bad_win,
            'quality': quality,
        }

    def get_stacked_window(self) -> np.ndarray:
        """Return pre-filtered steady-state (2*C, W) stacked [MRCP; SMR] window."""
        return np.concatenate([self.mrcp_buf.get(), self.smr_buf.get()], axis=0)

    @property
    def is_full(self) -> bool:
        return self.raw_buf.is_full


# =====================================================================
# Channel Resolution Utilities
# =====================================================================

def resolve_channel_indices(channels: Optional[List[str]],
                            target_channels: Tuple[str, ...]) -> Dict[str, int]:
    """Resolve channel name case-insensitively to array indices.

    If channels is None, maps target_channels to consecutive indices 0..k-1.
    """
    if channels is None:
        return {ch: i for i, ch in enumerate(target_channels)}

    lookup = {str(name).strip().casefold(): i for i, name in enumerate(channels)}
    resolved = {}
    for target in target_channels:
        norm = str(target).strip().casefold()
        if norm in lookup:
            resolved[target] = lookup[norm]
    return resolved


# =====================================================================
# Feature Extraction: MRCP Temporal Features
# =====================================================================

def compute_linear_slope(y: np.ndarray, sfreq: float) -> float:
    """Compute linear regression slope dv/dt (Volts/sec) over 1D series."""
    n = len(y)
    if n < 2:
        return 0.0
    i = np.arange(n, dtype=np.float64)
    weights = (i - (n - 1) / 2.0) / (n * (n**2 - 1) / 12.0)
    return float(np.dot(y.astype(np.float64), weights) * sfreq)


def extract_mrcp_temporal_features(
    mrcp_window: np.ndarray,
    sfreq: float = DEFAULT_SFREQ,
    channels: Optional[List[str]] = None,
    motor_channels: Tuple[str, ...] = CORE_MOTOR_CHANNELS,
    trajectory_samples: int = 16,
    late_fraction: float = 0.5,
) -> Tuple[np.ndarray, List[str], Dict[str, Union[float, np.ndarray]]]:
    """Extract temporal MRCP features across motor channels.

    Parameters:
        mrcp_window: (n_channels, n_times) array in Volts, filtered 0.3-3 Hz.
        sfreq: sampling frequency (Hz).
        channels: list of channel names in mrcp_window.
        motor_channels: channels of interest (e.g. C3, Cz, C4).
        trajectory_samples: number of points to downsample readiness potential waveform.
        late_fraction: fraction of window designated as late intention/execution phase.

    Returns:
        feature_vector: 1D np.ndarray (float32) of all extracted features.
        feature_names: list of corresponding feature names.
        feature_dict: dictionary structured by channel and metric.
    """
    mrcp_window = np.asarray(mrcp_window, dtype=np.float32)
    if mrcp_window.ndim != 2:
        raise ValueError(f"Expected 2D array (channels x times), got {mrcp_window.shape}")
    n_ch, n_times = mrcp_window.shape

    ch_map = resolve_channel_indices(channels, motor_channels)
    late_samples = max(2, int(round(n_times * late_fraction)))
    endpoint_samples = max(1, int(round(0.1 * sfreq)))  # last 100 ms

    feature_dict: Dict[str, Union[float, np.ndarray]] = {}
    feature_vals: List[float] = []
    feature_names: List[str] = []

    channel_signals = {}
    channel_slopes = {}
    channel_late_means = {}

    for ch_name in motor_channels:
        if ch_name not in ch_map:
            continue
        idx = ch_map[ch_name]
        sig = mrcp_window[idx]
        channel_signals[ch_name] = sig

        # 1. Amplitude features
        min_amp = float(np.min(sig))  # Bereitschaftspotential negative peak
        min_idx = int(np.argmin(sig))
        min_latency_s = float(min_idx / sfreq)
        mean_amp = float(np.mean(sig))
        late_sig = sig[-late_samples:]
        late_mean = float(np.mean(late_sig))
        endpoint_amp = float(np.mean(sig[-endpoint_samples:]))
        # Negative AUC (integral of negative deflection in Volts * sec)
        auc_neg = float(np.sum(np.minimum(sig, 0.0)) / sfreq)

        channel_late_means[ch_name] = late_mean

        # 2. Dynamic / Slope features
        slope_full = compute_linear_slope(sig, sfreq)
        slope_late = compute_linear_slope(late_sig, sfreq)
        channel_slopes[ch_name] = slope_late

        # 3. Downsampled trajectory (captures negative slope waveform)
        bin_size = n_times / float(trajectory_samples)
        trajectory = np.zeros(trajectory_samples, dtype=np.float32)
        for b in range(trajectory_samples):
            a = int(round(b * bin_size))
            z = int(round((b + 1) * bin_size))
            trajectory[b] = np.mean(sig[a:max(a + 1, z)])

        # Record into dict
        ch_dict = {
            'min_amplitude': min_amp,
            'min_latency_s': min_latency_s,
            'mean_amplitude': mean_amp,
            'late_mean_amplitude': late_mean,
            'endpoint_amplitude': endpoint_amp,
            'auc_negative': auc_neg,
            'slope_full': slope_full,
            'slope_late': slope_late,
            'trajectory': trajectory,
        }
        feature_dict[ch_name] = ch_dict

        # Flatten scalars
        for key in ['min_amplitude', 'min_latency_s', 'mean_amplitude',
                    'late_mean_amplitude', 'endpoint_amplitude', 'auc_negative',
                    'slope_full', 'slope_late']:
            feature_names.append(f'mrcp_{ch_name}_{key}')
            feature_vals.append(ch_dict[key])

        # Flatten trajectory
        for b in range(trajectory_samples):
            feature_names.append(f'mrcp_{ch_name}_traj_{b:02d}')
            feature_vals.append(float(trajectory[b]))

    # Spatial Gradients & Asymmetries across Motor Cortex (C3, Cz, C4)
    if 'C3' in channel_signals and 'C4' in channel_signals:
        diff_c3_c4 = channel_late_means['C3'] - channel_late_means['C4']
        slope_diff_c3_c4 = channel_slopes['C3'] - channel_slopes['C4']
        feature_names.extend(['mrcp_diff_c3_c4_late', 'mrcp_diff_c3_c4_slope'])
        feature_vals.extend([diff_c3_c4, slope_diff_c3_c4])
        feature_dict['diff_c3_c4_late'] = diff_c3_c4
        feature_dict['diff_c3_c4_slope'] = slope_diff_c3_c4

    if 'Cz' in channel_signals:
        if 'C3' in channel_signals:
            diff_c3_cz = channel_late_means['C3'] - channel_late_means['Cz']
            feature_names.append('mrcp_diff_c3_cz_late')
            feature_vals.append(diff_c3_cz)
            feature_dict['diff_c3_cz_late'] = diff_c3_cz
        if 'C4' in channel_signals:
            diff_c4_cz = channel_late_means['C4'] - channel_late_means['Cz']
            feature_names.append('mrcp_diff_c4_cz_late')
            feature_vals.append(diff_c4_cz)
            feature_dict['diff_c4_cz_late'] = diff_c4_cz

    if len(channel_late_means) > 0:
        motor_mean = float(np.mean(list(channel_late_means.values())))
        motor_slope = float(np.mean(list(channel_slopes.values())))
        feature_names.extend(['mrcp_motor_mean_late', 'mrcp_motor_mean_slope'])
        feature_vals.extend([motor_mean, motor_slope])
        feature_dict['motor_mean_late'] = motor_mean
        feature_dict['motor_mean_slope'] = motor_slope

    return np.asarray(feature_vals, dtype=np.float32), feature_names, feature_dict


# =====================================================================
# Feature Extraction: SMR Spectral Features
# =====================================================================

def extract_smr_spectral_features(
    smr_window: np.ndarray,
    sfreq: float = DEFAULT_SFREQ,
    channels: Optional[List[str]] = None,
    motor_channels: Tuple[str, ...] = CORE_MOTOR_CHANNELS,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[np.ndarray, List[str], Dict[str, Union[float, Dict[str, float]]]]:
    """Extract sensorimotor rhythm spectral power features.

    Parameters:
        smr_window: (n_channels, n_times) array in Volts, filtered 4-40 Hz.
        sfreq: sampling frequency (Hz).
        channels: list of channel names in smr_window.
        motor_channels: channels of interest (e.g. C3, Cz, C4).
        bands: dict of band name -> (fmin, fmax). Defaults to theta, mu, beta_low, beta_high, gamma.

    Returns:
        feature_vector: 1D np.ndarray (float32) of log band powers and laterality indices.
        feature_names: list of corresponding feature names.
        feature_dict: dictionary structured by channel and band.
    """
    smr_window = np.asarray(smr_window, dtype=np.float32)
    if smr_window.ndim != 2:
        raise ValueError(f"Expected 2D array (channels x times), got {smr_window.shape}")
    n_ch, n_times = smr_window.shape

    if bands is None:
        bands = SMR_BANDS

    ch_map = resolve_channel_indices(channels, motor_channels)
    taper = np.hanning(n_times)
    taper_energy = np.sum(taper**2)
    scale = 2.0 / (sfreq * max(1e-12, taper_energy))
    freqs = np.fft.rfftfreq(n_times, 1.0 / sfreq)

    feature_dict: Dict[str, Union[float, Dict[str, float]]] = {}
    feature_vals: List[float] = []
    feature_names: List[str] = []

    channel_mu = {}
    channel_beta = {}

    for ch_name in motor_channels:
        if ch_name not in ch_map:
            continue
        idx = ch_map[ch_name]
        sig = smr_window[idx]
        centered = sig - np.mean(sig)
        fft_vals = np.fft.rfft(centered * taper)
        psd = (np.abs(fft_vals)**2) * scale

        ch_band_dict = {}
        total_smr_power = 0.0

        for b_name, (fmin, fmax) in bands.items():
            mask = (freqs >= fmin) & (freqs < fmax)
            power = float(np.mean(psd[mask])) if np.any(mask) else 1e-12
            log_power = float(np.log(max(power, 1e-12)))
            ch_band_dict[f'{b_name}_power'] = power
            ch_band_dict[f'{b_name}_log'] = log_power
            total_smr_power += power

            feature_names.append(f'smr_{ch_name}_{b_name}_log')
            feature_vals.append(log_power)

        # Mu and Beta tracking for ERD / laterality
        mu_p = ch_band_dict.get('mu_power', 1e-12)
        beta_low_p = ch_band_dict.get('beta_low_power', 1e-12)
        beta_high_p = ch_band_dict.get('beta_high_power', 1e-12)
        beta_p = beta_low_p + beta_high_p

        channel_mu[ch_name] = mu_p
        channel_beta[ch_name] = beta_p

        # Relative powers
        rel_mu = float(mu_p / max(1e-12, total_smr_power))
        rel_beta = float(beta_p / max(1e-12, total_smr_power))
        mu_beta_ratio = float(mu_p / max(1e-12, beta_p))

        ch_band_dict['rel_mu'] = rel_mu
        ch_band_dict['rel_beta'] = rel_beta
        ch_band_dict['mu_beta_ratio'] = mu_beta_ratio

        feature_names.extend([
            f'smr_{ch_name}_rel_mu',
            f'smr_{ch_name}_rel_beta',
            f'smr_{ch_name}_mu_beta_ratio',
        ])
        feature_vals.extend([rel_mu, rel_beta, mu_beta_ratio])
        feature_dict[ch_name] = ch_band_dict

    # Contralateral Sensorimotor Laterality (C3 vs C4)
    if 'C3' in channel_mu and 'C4' in channel_mu:
        mu_c3 = channel_mu['C3']
        mu_c4 = channel_mu['C4']
        mu_laterality = float((mu_c3 - mu_c4) / max(1e-12, mu_c3 + mu_c4))

        beta_c3 = channel_beta['C3']
        beta_c4 = channel_beta['C4']
        beta_laterality = float((beta_c3 - beta_c4) / max(1e-12, beta_c3 + beta_c4))

        feature_names.extend(['smr_mu_laterality_c3_c4', 'smr_beta_laterality_c3_c4'])
        feature_vals.extend([mu_laterality, beta_laterality])
        feature_dict['mu_laterality_c3_c4'] = mu_laterality
        feature_dict['beta_laterality_c3_c4'] = beta_laterality

    return np.asarray(feature_vals, dtype=np.float32), feature_names, feature_dict


# =====================================================================
# Unified Dual-Band Feature Extraction
# =====================================================================

def extract_dual_band_features(
    raw_window: np.ndarray,
    dual_filter: Optional[DualBandFilter] = None,
    sfreq: float = DEFAULT_SFREQ,
    channels: Optional[List[str]] = None,
    motor_channels: Tuple[str, ...] = CORE_MOTOR_CHANNELS,
    trajectory_samples: int = 16,
) -> Tuple[np.ndarray, List[str], Dict[str, dict]]:
    """Extract combined MRCP temporal and SMR spectral features directly from raw EEG.

    Parameters:
        raw_window: (n_channels, n_times) in Volts.
        dual_filter: optional DualBandFilter instance.
        sfreq: sampling frequency (Hz).
        channels: list of channel names.
        motor_channels: motor channels for specialized extraction (C3, Cz, C4).
        trajectory_samples: points for MRCP readiness waveform.

    Returns:
        feature_vector: concatenated 1D float32 array.
        feature_names: concatenated list of feature names.
        metadata_dict: {'mrcp': mrcp_dict, 'smr': smr_dict}
    """
    raw_window = np.asarray(raw_window, dtype=np.float32)
    filt = dual_filter or DualBandFilter(sfreq=sfreq)

    # Filter causally with initial DC step transient correction
    mrcp_win, smr_win, _ = filt.filter_continuous(raw_window, initial_dc_correction=True)

    mrcp_feats, mrcp_names, mrcp_dict = extract_mrcp_temporal_features(
        mrcp_win, sfreq=sfreq, channels=channels, motor_channels=motor_channels,
        trajectory_samples=trajectory_samples
    )

    smr_feats, smr_names, smr_dict = extract_smr_spectral_features(
        smr_win, sfreq=sfreq, channels=channels, motor_channels=motor_channels
    )

    combined_vec = np.concatenate([mrcp_feats, smr_feats]).astype(np.float32)
    combined_names = mrcp_names + smr_names
    combined_dict = {'mrcp': mrcp_dict, 'smr': smr_dict}

    return combined_vec, combined_names, combined_dict


# =====================================================================
# Deep Learning Dual-Band Tensor Preparation
# =====================================================================

def slice_continuous_dual_band_windows(
    raw_continuous: np.ndarray,
    window_starts: Union[List[int], np.ndarray],
    window_samples: int,
    dual_filter: Optional[DualBandFilter] = None,
    sfreq: float = DEFAULT_SFREQ,
    mode: str = 'stacked_channels',
) -> np.ndarray:
    """Filter continuous raw EEG once and slice steady-state dual-band windows.

    Avoids per-window 0.3 Hz startup transients by preserving continuous filter
    state across the full recording before slicing windows.

    Parameters:
        raw_continuous: (C, T_total) raw EEG in Volts.
        window_starts: 1D sequence of start sample indices.
        window_samples: window length in samples (W).
        dual_filter: optional DualBandFilter instance.
        sfreq: sampling frequency (Hz).
        mode:
            'stacked_channels': returns (N, 2*C, W) where 0..C-1 is MRCP, C..2C-1 is SMR.
            'dual_branch': returns (N, 2, C, W) where 0 is MRCP, 1 is SMR.
            'tuple': returns (mrcp_windows, smr_windows) each (N, C, W).
    """
    raw_continuous = np.asarray(raw_continuous, dtype=np.float32)
    if raw_continuous.ndim != 2:
        raise ValueError(f"Expected 2D continuous raw array (C, T_total), got {raw_continuous.shape}")
    filt = dual_filter or DualBandFilter(sfreq=sfreq)
    mrcp_cont, smr_cont, _ = filt.filter_continuous(raw_continuous, initial_dc_correction=True)

    n_ch = raw_continuous.shape[0]
    w_len = int(window_samples)
    starts = [int(s) for s in window_starts if 0 <= int(s) and int(s) + w_len <= raw_continuous.shape[1]]

    if not starts:
        if mode == 'stacked_channels':
            return np.empty((0, 2 * n_ch, w_len), dtype=np.float32)
        elif mode == 'dual_branch':
            return np.empty((0, 2, n_ch, w_len), dtype=np.float32)
        elif mode == 'tuple':
            empty = np.empty((0, n_ch, w_len), dtype=np.float32)
            return empty, empty
        raise ValueError(f"Unknown mode '{mode}'.")

    mrcp_wins = np.stack([mrcp_cont[:, s:s + w_len] for s in starts]).astype(np.float32)
    smr_wins = np.stack([smr_cont[:, s:s + w_len] for s in starts]).astype(np.float32)

    if mode == 'stacked_channels':
        return np.concatenate([mrcp_wins, smr_wins], axis=1)
    elif mode == 'dual_branch':
        return np.stack([mrcp_wins, smr_wins], axis=1)
    elif mode == 'tuple':
        return mrcp_wins, smr_wins
    else:
        raise ValueError(f"Unknown mode '{mode}'; choose 'stacked_channels', 'dual_branch', or 'tuple'.")


def prepare_dual_band_tensor(
    raw_windows: Union[np.ndarray, List[np.ndarray]],
    dual_filter: Optional[DualBandFilter] = None,
    sfreq: float = DEFAULT_SFREQ,
    mode: str = 'stacked_channels',
    prefiltered: bool = False,
) -> np.ndarray:
    """Transform EEG windows into dual-band tensors for CNNs/Deep Learning.

    NOTE: For 0.3-3.0 Hz MRCP equivalence between training and streaming, prefer
    `slice_continuous_dual_band_windows` on continuous recordings or pass
    already steady-state filtered windows with `prefiltered=True`.
    """
    if isinstance(raw_windows, list):
        raw_windows = np.stack(raw_windows)
    raw_windows = np.asarray(raw_windows, dtype=np.float32)
    if raw_windows.ndim != 3:
        raise ValueError(f"Expected 3D array (N, C, T), got {raw_windows.shape}")

    if prefiltered:
        if mode == 'stacked_channels':
            return raw_windows
        elif mode == 'dual_branch':
            half = raw_windows.shape[1] // 2
            return np.stack([raw_windows[:, :half, :], raw_windows[:, half:, :]], axis=1)
        else:
            raise ValueError(f"Unknown mode '{mode}'; choose 'stacked_channels' or 'dual_branch'.")

    filt = dual_filter or DualBandFilter(sfreq=sfreq)
    n_trials, n_channels, n_times = raw_windows.shape

    mrcp_batch = np.empty((n_trials, n_channels, n_times), dtype=np.float32)
    smr_batch = np.empty((n_trials, n_channels, n_times), dtype=np.float32)

    for i in range(n_trials):
        m, s, _ = filt.filter_continuous(raw_windows[i], initial_dc_correction=True)
        mrcp_batch[i] = m
        smr_batch[i] = s

    if mode == 'stacked_channels':
        return np.concatenate([mrcp_batch, smr_batch], axis=1)  # (N, 2*C, T)
    elif mode == 'dual_branch':
        return np.stack([mrcp_batch, smr_batch], axis=1)        # (N, 2, C, T)
    else:
        raise ValueError(f"Unknown mode '{mode}'; choose 'stacked_channels' or 'dual_branch'.")
