"""Utility functions untuk logging, seeding, dan IO."""
import os
import random
import logging
import json
import numpy as np
import torch
import yaml
from pathlib import Path


def load_config(path: str) -> dict:
    """Load konfigurasi YAML."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed: int = 42):
    """Set random seed untuk reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(cfg_device: str = "auto") -> torch.device:
    """Pilih device training."""
    if cfg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg_device)


def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """Setup logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def save_json(obj: dict, path: str):
    """Simpan dict ke JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)
