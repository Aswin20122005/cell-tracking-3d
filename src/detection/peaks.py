"""Turn a predicted heatmap into discrete detections, and run the network over a
full volume via overlapping sliding windows (volumes rarely fit in memory whole).
"""

from __future__ import annotations

import numpy as np
import torch

from .blob_baseline import local_maxima


def peaks_from_heatmap(heatmap, threshold=0.5, min_distance=3):
    """Extract (z, y, x) detections from a predicted heatmap."""
    return local_maxima(heatmap, threshold=threshold, min_distance=min_distance)


@torch.no_grad()
def predict_volume(model, volume, patch_size, device, overlap=0.25):
    """Sliding-window inference; overlapping predictions are max-pooled together."""
    model.eval()
    pz, py, px = patch_size
    step = [max(1, int(s * (1 - overlap))) for s in patch_size]
    acc = np.zeros_like(volume, dtype=np.float32)

    z_starts = _starts(volume.shape[0], pz, step[0])
    y_starts = _starts(volume.shape[1], py, step[1])
    x_starts = _starts(volume.shape[2], px, step[2])

    for z0 in z_starts:
        for y0 in y_starts:
            for x0 in x_starts:
                patch = volume[z0:z0 + pz, y0:y0 + py, x0:x0 + px]
                t = torch.from_numpy(patch)[None, None].to(device)
                pred = model(t)[0, 0].cpu().numpy()
                sub = acc[z0:z0 + pz, y0:y0 + py, x0:x0 + px]
                acc[z0:z0 + pz, y0:y0 + py, x0:x0 + px] = np.maximum(sub, pred)
    return acc


def _starts(dim, patch, step):
    if dim <= patch:
        return [0]
    starts = list(range(0, dim - patch + 1, step))
    if starts[-1] != dim - patch:
        starts.append(dim - patch)
    return starts
