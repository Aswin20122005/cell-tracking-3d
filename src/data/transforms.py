"""Training augmentation and test-time augmentation (TTA) transforms.

Only flips and 90° rotations in the Y-X plane are used: they keep voxel values
exact and never move a cell out of the patch. Z is not rotated into Y/X because
microscopy volumes are anisotropic (fewer, thicker Z slices).
"""

from __future__ import annotations

import numpy as np


def _c(a):
    # torch.from_numpy can't take arrays with negative strides (which flips produce)
    return np.ascontiguousarray(a)


def random_augment(patch, heatmap, rng):
    """Apply the same random flips / Y-X rotation to a patch and its heatmap."""
    for axis in range(3):
        if rng.random() < 0.5:
            patch, heatmap = np.flip(patch, axis), np.flip(heatmap, axis)
    if patch.shape[1] == patch.shape[2]:  # rotation keeps the shape only for square Y-X
        k = int(rng.integers(4))
        patch, heatmap = np.rot90(patch, k, axes=(1, 2)), np.rot90(heatmap, k, axes=(1, 2))
    return _c(patch), _c(heatmap)


# (name, forward, inverse): inverse maps a prediction back to the original orientation.
TTA_TRANSFORMS = [
    ("identity", lambda v: _c(v), lambda v: _c(v)),
    ("flip_z", lambda v: _c(np.flip(v, 0)), lambda v: _c(np.flip(v, 0))),
    ("flip_y", lambda v: _c(np.flip(v, 1)), lambda v: _c(np.flip(v, 1))),
    ("flip_x", lambda v: _c(np.flip(v, 2)), lambda v: _c(np.flip(v, 2))),
    ("rot90_yx", lambda v: _c(np.rot90(v, 1, axes=(1, 2))), lambda v: _c(np.rot90(v, -1, axes=(1, 2)))),
    ("rot180_yx", lambda v: _c(np.rot90(v, 2, axes=(1, 2))), lambda v: _c(np.rot90(v, -2, axes=(1, 2)))),
]
