"""Detection metrics: match predictions to ground truth within a distance tol
and report precision / recall / F1. Adapt to the exact competition metric.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def match_detections(pred, gt, tol=3.0, voxel_size=(1.0, 1.0, 1.0)):
    """One-to-one match pred<->gt within `tol` (in voxel_size units). Returns (tp, fp, fn)."""
    scale = np.asarray(voxel_size, dtype=float)
    pred = np.asarray(pred, dtype=float).reshape(-1, 3) * scale
    gt = np.asarray(gt, dtype=float).reshape(-1, 3) * scale
    if len(pred) == 0:
        return 0, 0, len(gt)
    if len(gt) == 0:
        return 0, len(pred), 0

    cost = np.linalg.norm(pred[:, None, :] - gt[None, :, :], axis=2)
    cost[cost > tol] = 1e6
    rows, cols = linear_sum_assignment(cost)
    tp = sum(1 for r, c in zip(rows, cols) if cost[r, c] < 1e6)
    fp = len(pred) - tp
    fn = len(gt) - tp
    return tp, fp, fn


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def score_frames(preds_per_frame, gts_per_frame, tol=3.0, voxel_size=(1.0, 1.0, 1.0)):
    """Aggregate P/R/F1 across all frames."""
    TP = FP = FN = 0
    for pred, gt in zip(preds_per_frame, gts_per_frame):
        tp, fp, fn = match_detections(pred, gt, tol=tol, voxel_size=voxel_size)
        TP += tp; FP += fp; FN += fn
    return prf(TP, FP, FN)
