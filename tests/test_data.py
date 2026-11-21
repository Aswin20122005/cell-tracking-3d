import os

import numpy as np
import pandas as pd
import pytest

from src.data.dataset import EmbryoDataset, normalise
from src.data.patches import gaussian_heatmap, sample_patches
from src.data.transforms import TTA_TRANSFORMS, random_augment


def test_normalise_maps_to_unit_range_and_handles_constant_volumes():
    v = normalise(np.arange(1000, dtype=float).reshape(10, 10, 10))
    assert v.min() == 0.0 and v.max() == 1.0 and v.dtype == np.float32
    assert normalise(np.ones((4, 4, 4))).max() == 0.0


def test_heatmap_peaks_at_centres():
    hm = gaussian_heatmap((10, 20, 20), np.array([[5, 10, 10]]), sigma=2.0)
    assert hm[5, 10, 10] == pytest.approx(1.0)
    assert np.unravel_index(hm.argmax(), hm.shape) == (5, 10, 10)
    assert hm[0, 0, 0] < 1e-3


def test_heatmap_handles_centres_near_the_edge_and_no_centres():
    hm = gaussian_heatmap((10, 20, 20), np.array([[0, 0, 0]]), sigma=2.0)
    assert hm[0, 0, 0] == pytest.approx(1.0)
    assert gaussian_heatmap((4, 4, 4), np.empty((0, 3))).max() == 0


def _frame():
    # sparse: 3 cells in a 12x48x48 volume, so random patches are often empty
    return EmbryoDataset("data", ["e1"], synthetic=True, synth_shape=(12, 48, 48), synth_cells=3).frames()[0]


def test_uniform_only_sampling_includes_empty_patches():
    rng = np.random.default_rng(0)
    patches = list(sample_patches(_frame(), (8, 16, 16), n_patches=40, gt_centred_ratio=0,
                                  uniform_ratio=1, rng=rng))
    assert any(hm.max() < 0.5 for _, hm in patches)


def test_gt_centred_sampling_always_contains_a_cell():
    rng = np.random.default_rng(0)
    patches = list(sample_patches(_frame(), (8, 16, 16), n_patches=20, gt_centred_ratio=1,
                                  uniform_ratio=0, rng=rng))
    assert all(hm.max() > 0.5 for _, hm in patches)


def test_augment_keeps_patch_and_heatmap_aligned():
    rng = np.random.default_rng(1)
    patch = np.zeros((8, 16, 16), np.float32)
    patch[2, 3, 4] = 1
    p, h = random_augment(patch, patch.copy(), rng)
    assert np.array_equal(p, h) and p.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("name, fwd, inv", TTA_TRANSFORMS)
def test_tta_inverse_restores_orientation(name, fwd, inv):
    v = np.random.default_rng(2).random((6, 10, 10)).astype(np.float32)
    assert np.array_equal(inv(fwd(v)), v)


def test_real_layout_is_loaded(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    emb = tmp_path / "embryo_01" / "volumes"
    os.makedirs(emb)
    for t in range(2):
        tifffile.imwrite(emb / f"t{t:04d}.tif", np.random.default_rng(t).random((4, 8, 8)).astype(np.float32))
    pd.DataFrame({"t": [0, 1, 1], "z": [1, 2, 3], "y": [2, 3, 4], "x": [3, 4, 5]}).to_csv(
        tmp_path / "embryo_01" / "labels.csv", index=False)

    frames = EmbryoDataset(str(tmp_path), ["embryo_01"]).frames()
    assert [f.t for f in frames] == [0, 1]
    assert len(frames[0].centres) == 1 and len(frames[1].centres) == 2
    assert frames[0].volume.shape == (4, 8, 8)


def test_missing_volumes_give_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No volumes"):
        EmbryoDataset(str(tmp_path), ["nope"]).frames()
