"""Sample training patches and their target heatmaps.

A mix of patch types fixes the always-firing bias described in the README:
- GT-centred patches (a random cell in the middle) teach what a cell looks like;
- uniform patches (anywhere in the volume) include plenty of empty background, so
  the network also learns when NOT to fire.
"""

from __future__ import annotations

import numpy as np


def gaussian_heatmap(shape, centres, sigma=2.0):
    """Target heatmap: an isotropic Gaussian (peak 1.0) at each centre, max-combined."""
    heatmap = np.zeros(shape, dtype=np.float32)
    if len(centres) == 0:
        return heatmap
    radius = int(np.ceil(3 * sigma))
    for c in np.asarray(centres, dtype=float):
        lo = np.maximum(np.floor(c - radius).astype(int), 0)
        hi = np.minimum(np.ceil(c + radius + 1).astype(int), shape)
        if np.any(hi <= lo):
            continue
        grids = np.meshgrid(*[np.arange(l, h) for l, h in zip(lo, hi)], indexing="ij")
        d2 = sum((g - ci) ** 2 for g, ci in zip(grids, c))
        blob = np.exp(-d2 / (2 * sigma ** 2)).astype(np.float32)
        region = tuple(slice(l, h) for l, h in zip(lo, hi))
        heatmap[region] = np.maximum(heatmap[region], blob)
    return heatmap


def _crop(volume, start, size):
    """Crop with zero padding where the patch runs past the volume edge."""
    out = np.zeros(size, dtype=volume.dtype)
    src = tuple(slice(max(s, 0), min(s + n, d)) for s, n, d in zip(start, size, volume.shape))
    dst = tuple(slice(sl.start - s, sl.stop - s) for sl, s in zip(src, start))
    out[dst] = volume[src]
    return out


def sample_patches(frame, patch_size, sigma=2.0, n_patches=32, gt_centred_ratio=0.5,
                   uniform_ratio=0.5, rng=None):
    """Yield n_patches (patch, heatmap) pairs of shape patch_size from one frame.

    gt_centred_ratio and uniform_ratio are relative weights; frames without labelled
    cells only produce uniform patches.
    """
    rng = rng if rng is not None else np.random.default_rng()
    size = np.array(patch_size)
    shape = np.array(frame.volume.shape)
    centres = np.asarray(frame.centres)
    total = gt_centred_ratio + uniform_ratio
    p_gt = gt_centred_ratio / total if total > 0 and len(centres) else 0.0

    # The full-volume heatmap is built once and cropped like the volume.
    full_heatmap = gaussian_heatmap(tuple(shape), centres, sigma)

    for _ in range(n_patches):
        if rng.random() < p_gt:
            centre = centres[rng.integers(len(centres))]
            jitter = rng.integers(-(size // 4), size // 4 + 1)
            start = centre - size // 2 + jitter
        else:
            start = np.array([rng.integers(0, max(d - s, 0) + 1) for d, s in zip(shape, size)])
        yield _crop(frame.volume, start, tuple(size)), _crop(full_heatmap, start, tuple(size))
