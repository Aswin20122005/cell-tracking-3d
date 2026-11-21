"""Shared helpers: config loading, device selection, seeding."""

from __future__ import annotations

import random

import numpy as np
import yaml


REQUIRED_KEYS = {
    "data": ["root", "synthetic", "train_embryos", "val_embryos"],
    "patch": ["size", "gt_centred_ratio", "uniform_ratio", "sigma"],
    "model": ["in_channels", "base_channels", "depth"],
    "train": ["epochs", "batch_size", "lr", "patches_per_volume", "device", "checkpoint_dir"],
    "detect": ["peak_threshold", "nms_min_distance"],
    "track": ["max_link_distance", "min_track_length"],
}


def load_config(path):
    """Read a YAML config and check the keys the pipeline needs are present.

    A typo such as 'patches_per_vol' otherwise only surfaces as a KeyError deep
    inside training, possibly after minutes of data loading.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    missing = [f"{section}.{key}" for section, keys in REQUIRED_KEYS.items()
               for key in keys if key not in (cfg.get(section) or {})]
    if "seed" not in cfg:
        missing.append("seed")
    if missing:
        raise ValueError(f"{path} is missing config keys: {', '.join(missing)}")
    if len(cfg["patch"]["size"]) != 3:
        raise ValueError(f"{path}: patch.size must be [z, y, x]")
    return cfg


def get_device(requested="cuda"):
    import torch
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda":
        print("[utils] CUDA not available — falling back to CPU")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
