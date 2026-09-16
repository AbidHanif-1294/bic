"""
Generate dataset .edf sintetik dari simulasi neuron.

Output: data/raw/sub-A01_<n>.edf, data/raw/sub-A02_<n>.edf, dst.
Subject A berbeda dari Subject B karena seed neuron-nya beda
(mensimulasikan variabilitas antar-individu).
"""
import argparse
from pathlib import Path
import numpy as np
import mne

from neuron_simulator import simulate_brain_eeg


def save_as_edf(eeg: np.ndarray, fs: float, out_path: str, ch_names=None):
    """Simpan array (n_channels, n_times) sebagai .edf via MNE."""
    n_channels, n_times = eeg.shape
    if ch_names is None:
        ch_names = [f"EEG{i:03d}" for i in range(n_channels)]

    # MNE pakai satuan Volt. Data kita dalam µV → bagi 1e6.
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(eeg * 1e-6, info, verbose="ERROR")
    raw.export(out_path, fmt="edf", overwrite=True, verbose="ERROR")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/raw")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Durasi per file (detik)")
    parser.add_argument("--n-files", type=int, default=5,
                        help="Jumlah file per subjek")
    parser.add_argument("--n-channels", type=int, default=8)
    parser.add_argument("--fs", type=float, default=256.0)
    parser.add_argument("--subjects", nargs="+", default=["A"],
                        help="Daftar ID subjek. Contoh: A B C")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subject_seeds = {"A": 42, "B": 1337, "C": 2024, "D": 7, "E": 99}

    for subj in args.subjects:
        base_seed = subject_seeds.get(subj, hash(subj) % 10000)
        print(f"\n=== Subjek {subj} (base_seed={base_seed}) ===")

        for i in range(args.n_files):
            seed = base_seed + i
            print(f"\n-- File {i+1}/{args.n_files} (seed={seed}) --")
            eeg = simulate_brain_eeg(
                duration_sec=args.duration,
                n_channels=args.n_channels,
                fs=args.fs,
                seed=seed,
                verbose=True,
            )
            filename = out_dir / f"sub-{subj}{i+1:02d}_eeg.edf"
            save_as_edf(eeg, args.fs, str(filename))
            print(f"[OK] Saved: {filename}")


if __name__ == "__main__":
    main()
