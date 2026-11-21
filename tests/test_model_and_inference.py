import numpy as np
import pytest
import torch

from src.detection.peaks import _starts, peaks_from_heatmap, predict_volume
from src.inference.tta import predict_volume_tta
from src.models.unet3d import UNet3D
from src.validation.metrics import match_detections, prf, score_frames


@pytest.mark.parametrize("shape", [(16, 32, 32), (12, 30, 34)])
def test_unet_output_matches_input_shape_and_range(shape):
    model = UNet3D(in_channels=1, base_channels=4, depth=2).eval()
    with torch.no_grad():
        out = model(torch.rand(1, 1, *shape))
    assert out.shape == (1, 1, *shape)
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_sliding_window_covers_every_voxel():
    assert _starts(100, 32, 24) == [0, 24, 48, 68]
    assert _starts(20, 32, 24) == [0]


def test_predict_volume_and_tta_return_full_size_heatmaps():
    model = UNet3D(base_channels=4, depth=2)
    volume = np.random.default_rng(0).random((10, 40, 40)).astype(np.float32)
    hm = predict_volume(model, volume, (8, 16, 16), torch.device("cpu"))
    tta = predict_volume_tta(model, volume, (8, 16, 16), torch.device("cpu"))
    assert hm.shape == tta.shape == volume.shape


def test_peaks_respect_threshold_and_min_distance():
    hm = np.zeros((10, 20, 20), np.float32)
    hm[5, 5, 5], hm[5, 6, 5], hm[5, 15, 15] = 0.9, 0.8, 0.4
    assert peaks_from_heatmap(hm, threshold=0.5, min_distance=2).tolist() == [[5, 5, 5]]


def test_matching_counts_fp_and_fn():
    gt = np.array([[0, 0, 0], [10, 10, 10]])
    pred = np.array([[0, 1, 0], [20, 20, 20]])
    assert match_detections(pred, gt, tol=3) == (1, 1, 1)
    assert match_detections(np.empty((0, 3)), gt) == (0, 0, 2)


def test_score_frames_aggregates():
    res = score_frames([np.array([[0, 0, 0]])] * 2, [np.array([[0, 0, 0]])] * 2)
    assert res["f1"] == 1.0 and res["tp"] == 2
    assert prf(0, 0, 0)["f1"] == 0.0
