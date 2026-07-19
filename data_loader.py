"""
data_loader.py
--------------
Loads all EEG Motor Imagery data following the BIDS structure:
  <root>/sub-XX/ses-XX/eeg/sub-XX_ses-XX_task-MI_run-XX_{eeg.edf, events.tsv, channels.tsv}

Usage:
    from data_loader import MIDataLoader
    loader = MIDataLoader(root="C:/Users/giaco/Box/EEGMI")
    epochs, labels, meta = loader.load_all()
"""

import os
import glob
import numpy as np
import mne
import pandas as pd
from pathlib import Path

# Dataset structure
N_SUBJECTS   = 11
N_SESSIONS   = {i: 6 for i in range(1, 12)}
N_SESSIONS[2] = 4          # sub-02 has only 4 sessions
N_RUNS       = 3
SFREQ        = 256          # Hz

# EEG channels only (drop EKG, AUX*)
EEG_CHANNELS = ['F3','F1','Fz','F2','F4',
                 'C3','C1','Cz','C2','C4',
                 'CPz','P3','P1','Pz','P2',
                 'P4','POz','O1','Oz','O2']

# Motor-imagery relevant channels (used for feature focus; full array still fed to net)
MI_CHANNELS = ['C3','C1','Cz','C2','C4']

# Epoch windows (relative to event onset, in seconds)
TMIN, TMAX = 0.5, 4.5      # 4-second windows, 0.5 s after cue to avoid onset artifact

# Frequency band for bandpass
FMIN, FMAX = 8.0, 30.0     # mu (8-13 Hz) + beta (13-30 Hz)


class MIDataLoader:
    """
    Loads, preprocesses, and epochs all Motor Imagery EEG recordings.

    Parameters
    ----------
    root : str
        Root directory of the BIDS dataset (contains sub-XX folders).
    tmin : float
        Epoch start relative to event onset (s).
    tmax : float
        Epoch end relative to event onset (s).
    fmin : float
        High-pass filter cutoff (Hz).
    fmax : float
        Low-pass filter cutoff (Hz).
    verbose : bool
        MNE verbosity.
    """

    def __init__(self, root: str,
                 tmin: float = TMIN, tmax: float = TMAX,
                 fmin: float = FMIN, fmax: float = FMAX,
                 verbose: bool = False):
        self.root    = Path(root)
        self.tmin    = tmin
        self.tmax    = tmax
        self.fmin    = fmin
        self.fmax    = fmax
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_all(self, subjects=None, sessions=None):
        """
        Load and epoch data for all (or selected) subjects/sessions.

        Returns
        -------
        X : np.ndarray, shape (n_epochs, n_channels, n_times)
        y : np.ndarray, shape (n_epochs,)  — 0=rest, 1=action
        meta : pd.DataFrame
            One row per epoch with columns [subject, session, run, trial_type].
        """
        subjects = subjects or list(range(1, N_SUBJECTS + 1))
        all_X, all_y, all_meta = [], [], []

        for sub in subjects:
            n_ses = sessions if sessions else list(range(1, N_SESSIONS[sub] + 1))
            for ses in n_ses:
                for run in range(1, N_RUNS + 1):
                    result = self._load_run(sub, ses, run)
                    if result is None:
                        continue
                    X, y, meta = result
                    all_X.append(X)
                    all_y.append(y)
                    all_meta.append(meta)

        if not all_X:
            raise RuntimeError("No data loaded. Check your root path.")

        X    = np.concatenate(all_X,    axis=0)
        y    = np.concatenate(all_y,    axis=0)
        meta = pd.concat(all_meta, ignore_index=True)

        print(f"\n✓ Loaded {len(X)} epochs | "
              f"action={np.sum(y==1)} | rest={np.sum(y==0)} | "
              f"shape={X.shape}")
        return X, y, meta

    def load_subject(self, subject: int):
        """Convenience: load a single subject."""
        return self.load_all(subjects=[subject])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_run(self, sub: int, ses: int, run: int):
        """Load one run. Returns (X, y, meta) or None if files missing."""
        edf_path    = self._build_path(sub, ses, run, "eeg.edf")
        events_path = self._build_path(sub, ses, run, "events.tsv")

        if not edf_path.exists():
            if self.verbose:
                print(f"  [SKIP] {edf_path.name} not found")
            return None

        try:
            raw = mne.io.read_raw_edf(str(edf_path), preload=True,
                                       verbose=self.verbose)
        except Exception as e:
            print(f"  [ERROR] reading {edf_path.name}: {e}")
            return None

        # ---- Channel selection & filtering --------------------------------
        available = [ch for ch in EEG_CHANNELS if ch in raw.ch_names]
        raw.pick_channels(available, ordered=True)
        raw.filter(self.fmin, self.fmax, method='iir',
                   iir_params=dict(order=4, ftype='butter'),
                   verbose=self.verbose)

        # ---- Events from TSV ----------------------------------------------
        events_df  = pd.read_csv(events_path, sep='\t')
        action_df  = events_df[events_df['trial_type'] == 'action'].copy()
        rest_df    = events_df[events_df['trial_type'] == 'rest'].copy()

        raw_events_action = self._df_to_mne_events(action_df, event_id=1)
        raw_events_rest   = self._df_to_mne_events(rest_df,   event_id=0)

        all_events = np.vstack([raw_events_action, raw_events_rest])
        all_events = all_events[np.argsort(all_events[:, 0])]   # sort by sample

        event_id = {'rest': 0, 'action': 1}

        # ---- Epoching ------------------------------------------------------
        try:
            epochs = mne.Epochs(raw, all_events, event_id=event_id,
                                 tmin=self.tmin, tmax=self.tmax,
                                 baseline=None, preload=True,
                                 verbose=self.verbose)
        except Exception as e:
            print(f"  [ERROR] epoching {edf_path.name}: {e}")
            return None

        if len(epochs) == 0:
            return None

        X = epochs.get_data()               # (n_epochs, n_channels, n_times)
        y = epochs.events[:, 2].astype(np.int64)

        # Clip amplitude outliers (> 100 µV) — simple artefact rejection
        peak = np.max(np.abs(X), axis=(1, 2))
        keep = peak < 100e-6
        if keep.sum() == 0:
            return None
        X, y = X[keep], y[keep]

        # Z-score per channel per epoch
        X = self._normalize(X)

        n_ep = len(y)
        meta = pd.DataFrame({
            'subject':    [sub]  * n_ep,
            'session':    [ses]  * n_ep,
            'run':        [run]  * n_ep,
            'trial_type': ['action' if yi == 1 else 'rest' for yi in y],
        })

        print(f"  sub-{sub:02d} ses-{ses:02d} run-{run:02d} → {n_ep} epochs "
              f"(action={np.sum(y==1)}, rest={np.sum(y==0)})")
        return X, y, meta

    def _build_path(self, sub: int, ses: int, run: int, suffix: str) -> Path:
        fname = f"sub-{sub:02d}_ses-{ses:02d}_task-MI_run-{run:02d}_{suffix}"
        return self.root / f"sub-{sub:02d}" / f"ses-{ses:02d}" / "eeg" / fname

    @staticmethod
    def _df_to_mne_events(df: pd.DataFrame, event_id: int) -> np.ndarray:
        """Convert events TSV rows to MNE events array (n, 3)."""
        samples = df['sample'].astype(int).values
        n = len(samples)
        events = np.zeros((n, 3), dtype=int)
        events[:, 0] = samples
        events[:, 2] = event_id
        return events

    @staticmethod
    def _normalize(X: np.ndarray) -> np.ndarray:
        """Z-score each channel across the time axis, per epoch."""
        mu  = X.mean(axis=2, keepdims=True)
        std = X.std(axis=2,  keepdims=True) + 1e-8
        return (X - mu) / std
