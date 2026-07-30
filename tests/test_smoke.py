"""End-to-end smoke test on synthetic data — no real dataset or GPU required.

Run: pytest -q   (or just `python -m tests.test_smoke`)
"""

from __future__ import annotations

import numpy as np

from src.data.dataset import EmbryoDataset
from src.data.patches import sample_patches, gaussian_heatmap
from src.detection.blob_baseline import detect_blobs
from src.tracking.linker import link_frames
from src.validation.metrics import match_detections, score_frames


def _synthetic_frames():
    ds = EmbryoDataset(root="data", embryos=["embryo_01"], synthetic=True,
                       synth_shape=(12, 64, 64))
    return ds.frames()


def test_dataset_synthetic():
    frames = _synthetic_frames()
    assert len(frames) > 0
    f = frames[0]
    assert f.volume.ndim == 3
    assert f.centres.shape[1] == 3


def test_patch_sampling_shapes():
    frame = _synthetic_frames()[0]
    rng = np.random.default_rng(0)
    patches = list(sample_patches(frame, (8, 32, 32), sigma=2.0, n_patches=5, rng=rng))
    assert len(patches) == 5
    for patch, hm in patches:
        assert patch.shape == (8, 32, 32)
        assert hm.shape == (8, 32, 32)
        assert 0.0 <= hm.max() <= 1.0


def test_dog_baseline_detects_something():
    frame = _synthetic_frames()[0]
    coords = detect_blobs(frame.volume, threshold=0.05, min_distance=2)
    assert coords.ndim == 2 and coords.shape[1] == 3


def test_tracking_links_frames():
    frames = _synthetic_frames()
    dets = [f.centres for f in frames]
    tracks = link_frames(dets, max_distance=15.0, min_track_length=2)
    assert len(tracks) > 0
    assert all("coords" in t for t in tracks)


def test_metrics_perfect_match():
    gt = np.array([[1, 10, 10], [2, 20, 20]])
    tp, fp, fn = match_detections(gt, gt, tol=1.0)
    assert (tp, fp, fn) == (2, 0, 0)


if __name__ == "__main__":
    test_dataset_synthetic()
    test_patch_sampling_shapes()
    test_dog_baseline_detects_something()
    test_tracking_links_frames()
    test_metrics_perfect_match()
    print("all smoke tests passed")
