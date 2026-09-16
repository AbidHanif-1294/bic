"""Simulasi aktivitas EEG dari Model A (internal representation dynamics)."""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import load_config, get_device, setup_logger, ensure_dir
from data_loader import list_eeg_files, load_raw_eeg, make_synthetic_eeg
from preprocessing import preprocess_raw, build_dataset
from dynamic_model import DynamicNeuralSystem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--n-steps", type=int, default=200, help="Jumlah step simulasi")
    parser.add_argument("--temperature", type=float, default=0.0, help="Stochasticity")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["evaluation"]["output_dir"]) / cfg["experiment"]["model_name"]
    ensure_dir(str(out_dir))
    logger = setup_logger("sim", str(out_dir / "sim.log"))
    device = get_device(cfg["training"]["device"])

    # --- Data seed ---
    dcfg = cfg["data"]
    if args.synthetic:
        raw = make_synthetic_eeg(
            n_channels=8, duration_sec=120.0, sfreq=dcfg["sfreq"],
            seed=cfg["experiment"]["seed"],
        )
    else:
        files = list_eeg_files(dcfg["raw_dir"], dcfg["file_format"])
        raw = load_raw_eeg(files[0])

    raw = preprocess_raw(
        raw, sfreq=dcfg["sfreq"], lowcut=dcfg["lowcut"],
        highcut=dcfg["highcut"], notch_freq=dcfg.get("notch_freq"),
        channels=dcfg.get("channels"),
    )
    windows, _ = build_dataset(raw, dcfg["window_sec"], dcfg["stride_sec"])
    windows = np.transpose(windows, (0, 2, 1)).astype(np.float32)

    # --- Load model ---
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    model = DynamicNeuralSystem(
        input_dim=ckpt["input_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        latent_dim=cfg["model"]["latent_dim"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"],
        tau=cfg["model"]["tau"],
        dt=cfg["model"]["dt"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Ambil window seed dari test set (bukan training)
    n = len(windows)
    n_train = int(n * cfg["training"]["train_split"])
    n_val = int(n * cfg["training"]["val_split"])
    test_start = n_train + n_val

    seed_window = torch.from_numpy(windows[test_start : test_start + 1]).to(device)
    logger.info(f"Seed window shape: {seed_window.shape}")

    sim = model.simulate(seed_window, n_steps=args.n_steps, temperature=args.temperature)
    sim = sim.cpu().numpy()[0]  # (n_steps, C)
    logger.info(f"Simulated shape: {sim.shape}")

    np.save(out_dir / "simulation.npy", sim)

    # --- Plot ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 5))
    axes[0].plot(seed_window.cpu().numpy()[0, :, 0], label="Seed EEG (asli)")
    axes[0].set_title("Seed window dari Subject A (channel 0)")
    axes[0].legend()

    axes[1].plot(sim[:, 0], color="green", label="Simulasi Model A")
    axes[1].set_title(f"Simulasi aktivitas (Model A) - {args.n_steps} step")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_dir / "simulation.png", dpi=100)
    plt.close()

    logger.info(f"Simulasi selesai. Output: {out_dir}/simulation.npy")


if __name__ == "__main__":
    main()
