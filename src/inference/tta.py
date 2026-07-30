"""Test-time augmentation: run the detector over flipped/rotated copies of a
volume and average the heatmaps back in the original orientation. Stabilises
predictions and trims spurious peaks.
"""

from __future__ import annotations

import numpy as np

from ..data.transforms import TTA_TRANSFORMS
from ..detection.peaks import predict_volume


def predict_volume_tta(model, volume, patch_size, device, overlap=0.25):
    """Average heatmaps across the TTA transform set."""
    acc = np.zeros_like(volume, dtype=np.float32)
    for _name, fwd, inv in TTA_TRANSFORMS:
        aug = fwd(volume)
        pred = predict_volume(model, aug, patch_size, device, overlap=overlap)
        acc += inv(pred)
    return acc / len(TTA_TRANSFORMS)
