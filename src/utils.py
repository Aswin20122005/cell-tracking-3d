"""Shared helpers: config loading, device selection, seeding."""

from __future__ import annotations

import random

import numpy as np
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


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
