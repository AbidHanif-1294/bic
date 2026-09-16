"""Evaluasi Model A terhadap EEG asli."""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as sp_signal
from torch.utils.data import DataLoader, TensorDataset

from utils import load_config, get_device, setup_logger, save_json, ensure_dir
from data_loader import list_eeg_files, load_raw_eeg, make_synthetic_eeg
from preprocessing import preprocess_raw, build_dataset
from dynamic_model import DynamicNeuralSystem, DynamicLoss


def compute_metrics(x_true: np.ndarray, x_pred: np.ndarray) -> dict:
    """x_true, x_pred: (N, T, C)."""
    mse = float(np.mean((x_true - x_pred) ** 2))
    mae = float(np.mean(np.abs(x_true - x_pred)))

    # Correlation per channel, dirata-rata
    corrs = []
    for c in range(x_true.shape[-1]):
        a = x_true[..., c].flatten()
        b = x_pred[..., c].flatten()
        if a.std() > 1e-8 and b.std() > 1e-8:
            corrs.append(float(np.corrcoef(a, b)[0, 1]))
    corr = float(np.mean(corrs)) if corrs else 0.0

    # PSD distance (Welch)
    psd_true = []
    psd_pred = []
    for c in range(x_true.shape[-1]):
        f, p1 = sp_signal.welch(x_true[..., c].flatten(), fs=256, nperseg=256)
        _, p2 = sp_signal.welch(x_pred[..., c].flatten(), fs=256, nperseg=256)
        psd_true.append(p1)
        psd_pred.append(p2)
    psd_true = np.array(psd_true)
    psd_pred = np.array(psd_pred)
    # Normalisasi agar skala sebanding
    psd_true_n = psd_true / (psd_true.sum(axis=-1, keepdims=True) + 1e-12)
    psd_pred_n = psd_pred / (psd_pred.sum(axis=-1, keepdims=True) + 1e-12)
    psd_dist = float(np.mean(np.abs(psd_true_n - psd_pred_n)))

    return {
        "mse": mse,
        "mae": mae,
        "correlation": corr,
        "psd_distance": psd_dist,
    }


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    preds, trues = [], []
    for x, in loader:
        x = x.to(device)
        out = model(x)
        preds.append(out["reconstruction"].cpu().numpy())
        trues.append(x.cpu().numpy())
    return np.concatenate(trues, 0), np.concatenate(preds, 0)


def plot_reconstruction(x_true, x_pred, out_path, ch=0, n_show=500):
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    t = np.arange(n_show)
    axes[0].plot(t, x_true[:n_show, ch], label="EEG asli", lw=1)
    axes[0].set_title("Sinyal EEG asli (windowed)")
    axes[0].legend()

    axes[1].plot(t, x_pred[:n_show, ch], color="orange", label="Rekonstruksi Model A", lw=1)
    axes[1].set_title("Rekonstruksi Model A")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()


def plot_psd(x_true, x_pred, out_path, fs=256):
    fig, ax = plt.subplots(figsize=(10, 4))
    for c in range(min(3, x_true.shape[-1])):
        f, p = sp_signal.welch(x_true[..., c].flatten(), fs=fs, nperseg=256)
        ax.semilogy(f, p, label=f"Asli ch{c}", alpha=0.7)
        f, p = sp_signal.welch(x_pred[..., c].flatten(), fs=fs, nperseg=256)
        ax.semilogy(f, p, label=f"Model A ch{c}", alpha=0.7, linestyle="--")
    ax.set_xlabel("Frekuensi (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("Perbandingan Power Spectral Density")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["evaluation"]["output_dir"]) / cfg["experiment"]["model_name"]
    logger = setup_logger("eval", str(out_dir / "eval.log"))
    device = get_device(cfg["training"]["device"])

    # --- Data ---
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

    # Gunakan test split yang sama dengan training
    n = len(windows)
    n_train = int(n * cfg["training"]["train_split"])
    n_val = int(n * cfg["training"]["val_split"])
    test = windows[n_train + n_val:]

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

    loader = DataLoader(
        TensorDataset(torch.from_numpy(test)),
        batch_size=cfg["training"]["batch_size"], shuffle=False,
    )
    x_true, x_pred = predict_all(model, loader, device)
    logger.info(f"True shape: {x_true.shape}, Pred shape: {x_pred.shape}")

    metrics = compute_metrics(x_true, x_pred)
    logger.info(f"Metrics: {metrics}")

    if cfg["evaluation"]["save_plots"]:
        plot_reconstruction(x_true, x_pred, str(out_dir / "reconstruction.png"))
        plot_psd(x_true, x_pred, str(out_dir / "psd_comparison.png"))
        logger.info("Plots saved.")

    save_json(metrics, str(out_dir / "eval_metrics.json"))
    logger.info(f"Evaluasi selesai. Output: {out_dir}")


if __name__ == "__main__":
    main()
