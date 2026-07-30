"""Seed ensembling: average heatmaps from several trained checkpoints.

A small ensemble (e.g. 2-3 seeds) noticeably steadies detections at modest extra
inference cost.
"""

from __future__ import annotations

import numpy as np
import torch

from ..models.unet3d import UNet3D
from ..detection.peaks import predict_volume
from .tta import predict_volume_tta


def load_model(ckpt_path, model_cfg, device):
    model = UNet3D(**model_cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    return model


def predict_volume_ensemble(ckpt_paths, model_cfg, volume, patch_size, device,
                            use_tta=False, overlap=0.25):
    """Average heatmaps across an ensemble of checkpoints (optionally with TTA)."""
    acc = np.zeros_like(volume, dtype=np.float32)
    for ckpt in ckpt_paths:
        model = load_model(ckpt, model_cfg, device)
        if use_tta:
            pred = predict_volume_tta(model, volume, patch_size, device, overlap)
        else:
            pred = predict_volume(model, volume, patch_size, device, overlap)
        acc += pred
    return acc / len(ckpt_paths)
