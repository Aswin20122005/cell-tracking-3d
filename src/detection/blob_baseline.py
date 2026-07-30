"""Classical Difference-of-Gaussians blob detector — the no-training baseline.

Fast reference point and a sanity check that cells are actually detectable in the
raw signal before we invest in training a network.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def detect_blobs(volume, min_sigma=1.0, max_sigma=3.0, threshold=0.1, min_distance=3):
    """Return an (N, 3) array of (z, y, x) blob centres via DoG + local maxima.

    A simple two-scale DoG highlights blob-like structures; local maxima above a
    threshold, with non-max suppression, become detections.
    """
    g_small = ndi.gaussian_filter(volume, min_sigma)
    g_large = ndi.gaussian_filter(volume, max_sigma)
    dog = g_small - g_large

    return local_maxima(dog, threshold=threshold, min_distance=min_distance)


def local_maxima(response, threshold, min_distance):
    """Non-max suppression: peaks above `threshold`, separated by `min_distance`."""
    size = 2 * min_distance + 1
    maxf = ndi.maximum_filter(response, size=size, mode="constant")
    peaks = (response == maxf) & (response >= threshold)
    coords = np.argwhere(peaks)
    return coords.astype(int)
