"""Load embryo time-lapse volumes and their cell-centre labels.

Expected layout (one folder per embryo):

    data/
    ├── embryo_01/
    │   ├── volumes/     t0000.tif, t0001.tif, ...   (Z, Y, X per frame)
    │   └── labels.csv   t, z, y, x   (one row per cell centre)
    └── ...

With ``synthetic=True`` no files are read: small volumes with Gaussian "cells" that
drift slowly over time are generated instead, so the pipeline and tests run without
the real dataset.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Frame:
    embryo: str
    t: int
    volume: np.ndarray   # (Z, Y, X) float32, normalised to [0, 1]
    centres: np.ndarray  # (N, 3) int (z, y, x)


def normalise(volume, low=1.0, high=99.8):
    """Percentile-based min-max scaling to [0, 1], robust to hot pixels."""
    volume = volume.astype(np.float32)
    lo, hi = np.percentile(volume, [low, high])
    if hi <= lo:
        return np.zeros_like(volume, dtype=np.float32)
    return np.clip((volume - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


class EmbryoDataset:
    def __init__(self, root, embryos, synthetic=False, synth_shape=(16, 96, 96),
                 synth_frames=4, synth_cells=20, seed=0):
        self.root = root
        self.embryos = list(embryos)
        self.synthetic = synthetic
        self.synth_shape = tuple(synth_shape)
        self.synth_frames = synth_frames
        self.synth_cells = synth_cells
        self.seed = seed

    def frames(self):
        """All frames of all embryos, in (embryo, t) order."""
        out = []
        for i, embryo in enumerate(self.embryos):
            if self.synthetic:
                out.extend(self._synthetic_embryo(embryo, self.seed + i))
            else:
                out.extend(self._load_embryo(embryo))
        return out

    # ---- real data -------------------------------------------------------------

    def _load_embryo(self, embryo):
        import tifffile

        folder = os.path.join(self.root, embryo)
        paths = sorted(glob.glob(os.path.join(folder, "volumes", "t*.tif*")))
        if not paths:
            raise FileNotFoundError(f"No volumes found in {folder}/volumes (expected t0000.tif, ...)")
        labels_path = os.path.join(folder, "labels.csv")
        labels = pd.read_csv(labels_path) if os.path.exists(labels_path) else None

        frames = []
        for path in paths:
            t = int(re.search(r"t(\d+)", os.path.basename(path)).group(1))
            volume = normalise(tifffile.imread(path))
            if labels is not None:
                centres = labels.loc[labels["t"] == t, ["z", "y", "x"]].to_numpy(dtype=int)
            else:
                centres = np.empty((0, 3), dtype=int)
            frames.append(Frame(embryo, t, volume, centres))
        return frames

    # ---- synthetic data ----------------------------------------------------------

    def _synthetic_embryo(self, embryo, seed):
        rng = np.random.default_rng(seed)
        shape = np.array(self.synth_shape)
        margin = np.array([2, 6, 6])
        centres = rng.uniform(margin, shape - margin, size=(self.synth_cells, 3))
        frames = []
        for t in range(self.synth_frames):
            if t:
                # small random drift, kept inside the volume
                centres = np.clip(centres + rng.normal(0, 0.8, centres.shape), margin, shape - margin)
            volume = render_cells(self.synth_shape, centres, sigma=1.5)
            volume += rng.normal(0, 0.03, volume.shape).astype(np.float32)
            frames.append(Frame(embryo, t, normalise(volume), np.round(centres).astype(int)))
        return frames


def render_cells(shape, centres, sigma=1.5):
    """Volume with an isotropic Gaussian blob at every centre (peak value 1)."""
    zz, yy, xx = np.indices(shape, dtype=np.float32)
    volume = np.zeros(shape, dtype=np.float32)
    for z, y, x in centres:
        d2 = (zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2
        volume = np.maximum(volume, np.exp(-d2 / (2 * sigma ** 2)))
    return volume
