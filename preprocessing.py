"""Preprocessing pipeline EEG: filter, re-reference, resample, windowing."""
from typing import Optional, Tuple, List
import numpy as np
import mne
from sklearn.preprocessing import StandardScaler


def preprocess_raw(
    raw: mne.io.BaseRaw,
    sfreq: float = 256.0,
    lowcut: float = 1.0,
    highcut: float = 45.0,
    notch_freq: Optional[float] = 50.0,
    channels: Optional[List[str]] = None,
    apply_average_ref: bool = True,
    verbose: str = "ERROR",
) -> mne.io.BaseRaw:
    """
    Pipeline preprocessing:
    1. Pick channels (opsional)
    2. Resample
    3. Notch filter
    4. Band-pass filter
    5. Average reference
    """
    raw = raw.copy()

    # 1. Pick channels
    if channels is not None:
        available = [ch for ch in channels if ch in raw.ch_names]
        if not available:
            raise ValueError(f"Tidak ada channel {channels} di data. Tersedia: {raw.ch_names}")
        raw.pick(available)

    # Pastikan hanya tipe EEG
    raw.pick_types(eeg=True, meg=False, eog=False, stim=False, exclude="bads")

    # 2. Resample
    if abs(raw.info["sfreq"] - sfreq) > 1e-6:
        raw.resample(sfreq, verbose=verbose)

    # 3. Notch filter
    if notch_freq is not None and notch_freq < raw.info["sfreq"] / 2:
        raw.notch_filter(freqs=[notch_freq], verbose=verbose)

    # 4. Band-pass filter
    nyq = raw.info["sfreq"] / 2
    highcut = min(highcut, nyq - 1)
    raw.filter(l_freq=lowcut, h_freq=highcut, verbose=verbose)

    # 5. Average reference
    if apply_average_ref:
        raw.set_eeg_reference("average", verbose=verbose)

    return raw


def extract_array(raw: mne.io.BaseRaw) -> np.ndarray:
    """Ambil data numpy: shape (n_channels, n_times)."""
    return raw.get_data()


def zscore_normalize(
    data: np.ndarray,
    per_channel: bool = True,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Z-score normalization.
    data: (n_channels, n_times)
    """
    if per_channel:
        X = data.T  # (n_times, n_channels)
        if scaler is None:
            scaler = StandardScaler()
            Xn = scaler.fit_transform(X)
        else:
            Xn = scaler.transform(X)
        return Xn.T, scaler
    else:
        mu = data.mean()
        sigma = data.std() + 1e-9
        return (data - mu) / sigma, None


def make_windows(
    data: np.ndarray,
    window_size: int,
    stride: int,
    axis: int = -1,
) -> np.ndarray:
    """
    Sliding window.
    data: (n_channels, n_times)
    return: (n_windows, n_channels, window_size)
    """
    n_channels, n_times = data.shape if axis == -1 else (data.shape[0], data.shape[-1])
    if n_times < window_size:
        raise ValueError(f"Data length {n_times} < window {window_size}")

    starts = np.arange(0, n_times - window_size + 1, stride)
    windows = np.stack([data[:, s : s + window_size] for s in starts], axis=0)
    return windows.astype(np.float32)


def build_dataset(
    raw: mne.io.BaseRaw,
    window_sec: float,
    stride_sec: float,
) -> Tuple[np.ndarray, StandardScaler]:
    """
    End-to-end preprocessing + windowing.
    Return: (n_windows, n_channels, window_size), scaler
    """
    data = extract_array(raw)                          # (C, T)
    data_norm, scaler = zscore_normalize(data, per_channel=True)

    win = int(window_sec * raw.info["sfreq"])
    stride = int(stride_sec * raw.info["sfreq"])
    windows = make_windows(data_norm, win, stride)     # (N, C, W)
    return windows, scaler
