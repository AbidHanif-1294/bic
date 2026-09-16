"""Training loop untuk Model A."""
import os
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils import load_config, set_seed, get_device, setup_logger, save_json, ensure_dir
from data_loader import list_eeg_files, load_raw_eeg, make_synthetic_eeg, get_metadata
from preprocessing import preprocess_raw, build_dataset
from dynamic_model import DynamicNeuralSystem, DynamicLoss


def prepare_data(cfg: dict, logger, use_synthetic: bool = False):
    """Load + preprocess data Subject A."""
    dcfg = cfg["data"]

    if use_synthetic:
        logger.info("Menggunakan data EEG sintetik (mode testing).")
        raw = make_synthetic_eeg(
            n_channels=8,
            duration_sec=120.0,
            sfreq=dcfg["sfreq"],
            seed=cfg["experiment"]["seed"],
        )
    else:
        files = list_eeg_files(dcfg["raw_dir"], dcfg["file_format"])
        logger.info(f"Ditemukan {len(files)} file EEG.")
        # Untuk simplicity: gunakan file pertama sebagai Subject A
        raw = load_raw_eeg(files[0])
        logger.info(f"Loaded: {files[0]}")
        logger.info(f"Metadata: {get_metadata(raw)}")

    raw = preprocess_raw(
        raw,
        sfreq=dcfg["sfreq"],
        lowcut=dcfg["lowcut"],
        highcut=dcfg["highcut"],
        notch_freq=dcfg.get("notch_freq"),
        channels=dcfg.get("channels"),
    )

    windows, scaler = build_dataset(
        raw,
        window_sec=dcfg["window_sec"],
        stride_sec=dcfg["stride_sec"],
    )
    # windows: (N, C, W) -> transpose ke (N, W, C)
    windows = np.transpose(windows, (0, 2, 1))
    logger.info(f"Dataset shape: {windows.shape}")

    np.save(Path(dcfg["processed_dir"]) / f"subject_{cfg['experiment']['subject_id']}_windows.npy", windows)
    return windows, scaler, raw


def split_data(windows: np.ndarray, cfg: dict):
    """Split time-series: train/val/test (kronologis, bukan shuffle)."""
    tcfg = cfg["training"]
    n = len(windows)
    n_train = int(n * tcfg["train_split"])
    n_val = int(n * tcfg["val_split"])
    # n_test = n - n_train - n_val
    train = windows[:n_train]
    val = windows[n_train : n_train + n_val]
    test = windows[n_train + n_val :]
    return train, val, test


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    total_loss = 0.0
    metrics_acc = {}
    for x, in loader:
        x = x.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss, m = criterion(out, x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        for k, v in m.items():
            metrics_acc[k] = metrics_acc.get(k, 0.0) + v * x.size(0)
    n = len(loader.dataset)
    return total_loss / n, {k: v / n for k, v in metrics_acc.items()}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    metrics_acc = {}
    for x, in loader:
        x = x.to(device)
        out = model(x)
        loss, m = criterion(out, x)
        total_loss += loss.item() * x.size(0)
        for k, v in m.items():
            metrics_acc[k] = metrics_acc.get(k, 0.0) + v * x.size(0)
    n = len(loader.dataset)
    return total_loss / n, {k: v / n for k, v in metrics_acc.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--synthetic", action="store_true", help="Gunakan data sintetik")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])

    out_dir = Path(cfg["evaluation"]["output_dir"]) / cfg["experiment"]["model_name"]
    ensure_dir(str(out_dir))
    logger = setup_logger("train", str(out_dir / "train.log"))
    device = get_device(cfg["training"]["device"])
    logger.info(f"Device: {device}")

    # --- Data ---
    windows, scaler, raw = prepare_data(cfg, logger, use_synthetic=args.synthetic)
    train, val, test = split_data(windows, cfg)
    logger.info(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")

    train_ds = TensorDataset(torch.from_numpy(train))
    val_ds = TensorDataset(torch.from_numpy(val))
    test_ds = TensorDataset(torch.from_numpy(test))

    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

    # --- Model ---
    input_dim = train.shape[-1]
    mcfg = cfg["model"]
    model = DynamicNeuralSystem(
        input_dim=input_dim,
        hidden_dim=mcfg["hidden_dim"],
        latent_dim=mcfg["latent_dim"],
        num_layers=mcfg["num_layers"],
        dropout=mcfg["dropout"],
        tau=mcfg["tau"],
        dt=mcfg["dt"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")
    logger.info(f"Input dim (channels): {input_dim}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = DynamicLoss()

    # --- Training loop ---
    best_val = float("inf")
    patience_counter = 0
    history = {"train": [], "val": []}

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        tr_loss, tr_m = train_one_epoch(
            model, train_loader, optimizer, criterion, device, cfg["training"]["grad_clip"]
        )
        val_loss, val_m = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train"].append({"epoch": epoch, "loss": tr_loss, **tr_m})
        history["val"].append({"epoch": epoch, "loss": val_loss, **val_m})

        logger.info(
            f"Epoch {epoch:03d} | train {tr_loss:.5f} | val {val_loss:.5f} "
            f"| recon {tr_m['recon']:.5f} | smooth {tr_m['smooth']:.5f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "input_dim": input_dim,
                    "scaler_mean": scaler.mean_.tolist(),
                    "scaler_scale": scaler.scale_.tolist(),
                },
                out_dir / "best_model.pt",
            )
            logger.info(f"  ✓ Saved best model (val={val_loss:.5f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg["training"]["early_stopping_patience"]:
                logger.info("Early stopping triggered.")
                break

    # --- Final test ---
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_m = evaluate(model, test_loader, criterion, device)
    logger.info(f"TEST loss: {test_loss:.5f} | metrics: {test_m}")

    save_json(
        {
            "test_loss": test_loss,
            "test_metrics": test_m,
            "history": history,
            "n_params": n_params,
            "input_dim": input_dim,
        },
        str(out_dir / "training_report.json"),
    )
    logger.info(f"Selesai. Model disimpan di {out_dir}/best_model.pt")


if __name__ == "__main__":
    main()
