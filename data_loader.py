"""Loader untuk data EEG (MNE-based)."""
import os
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import mne


SUPPORTED_FORMATS = {"edf", "fif", "set", "bdf", "vhdr"}


def list_eeg_files(raw_dir: str, fmt: str = "edf") -> List[str]:
    """List semua file EEG di raw_dir dengan format tertentu."""
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Format {fmt} tidak didukung. Pilih dari {SUPPORTED_FORMATS}")
    p = Path(raw_dir)
    files = sorted([str(f) for f in p.glob(f"*.{fmt}")])
    if not files:
        raise FileNotFoundError(f"Tidak ada file *.{fmt} di {raw_dir}")
    return files


def load_raw_eeg(filepath: str) -> mne.io.BaseRaw:
    """Load file EEG mentah dengan MNE."""
    ext = filepath.split(".")[-1].lower()
    if ext == "edf":
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose="ERROR")
    elif ext == "bdf":
        raw = mne.io.read_raw_bdf(filepath, preload=True, verbose="ERROR")
    elif ext == "fif":
        raw = mne.io.read_raw_fif(filepath, preload=True, verbose="ERROR")
    elif ext == "set":
        raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose="ERROR")
    elif ext == "vhdr":
        raw = mne.io.read_raw_brainvision(filepath, preload=True, verbose="ERROR")
    else:
        raise ValueError(f"Format tidak dikenal: {ext}")
    return raw


def load_multiple_files(file_list: List[str]) -> List[mne.io.BaseRaw]:
    """Load banyak file sekaligus."""
    return [load_raw_eeg(f) for f in file_list]


def get_metadata(raw: mne.io.BaseRaw) -> dict:
    """Ambil metadata dasar dari raw EEG."""
    return {
        "sfreq": raw.info["sfreq"],
        "n_channels": len(raw.ch_names),
        "ch_names": raw.ch_names,
        "duration_sec": raw.n_times / raw.info["sfreq"],
        "n_times": raw.n_times,
    }


def make_synthetic_eeg(
    n_channels: int = 8,
    duration_sec: float = 60.0,
    sfreq: float = 256.0,
    seed: int = 42,
) -> mne.io.RawArray:
    """
    Buat data EEG sintetik untuk testing bila tidak ada data riil.
    Mensimulasikan osilasi alpha (10 Hz), beta (20 Hz), theta (6 Hz) + noise.
    """
    rng = np.random.default_rng(seed)
    n_times = int(duration_sec * sfreq)
    t = np.arange(n_times) / sfreq

    data = np.zeros((n_channels, n_times))
    for ch in range(n_channels):
        alpha = 15 * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi))
        beta = 8 * np.sin(2 * np.pi * 20 * t + rng.uniform(0, 2 * np.pi))
        theta = 12 * np.sin(2 * np.pi * 6 * t + rng.uniform(0, 2 * np.pi))
        pink = np.cumsum(rng.normal(0, 1, n_times))
        pink = pink / (np.std(pink) + 1e-9)
        noise = 5 * rng.normal(0, 1, n_times)
        data[ch] = alpha + beta + theta + 3 * pink + noise

    ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data * 1e-6, info, verbose="ERROR")  # V -> uV scale as SI
    return raw
