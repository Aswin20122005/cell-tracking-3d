"""Score a predictions CSV against ground-truth labels.

Usage:
    python -m src.evaluate --pred outputs/pred.csv --gt data/embryo_05/labels.csv
"""

from __future__ import annotations

import argparse

import pandas as pd

from .validation.metrics import score_frames


def run(pred_path, gt_path, tol):
    pred = pd.read_csv(pred_path)
    gt = pd.read_csv(gt_path)

    timepoints = sorted(set(pred["t"]).union(set(gt["t"])))
    preds_per_frame, gts_per_frame = [], []
    for t in timepoints:
        preds_per_frame.append(pred[pred["t"] == t][["z", "y", "x"]].to_numpy())
        gts_per_frame.append(gt[gt["t"] == t][["z", "y", "x"]].to_numpy())

    res = score_frames(preds_per_frame, gts_per_frame, tol=tol)
    print("[evaluate] " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                    for k, v in res.items()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--tol", type=float, default=3.0)
    args = ap.parse_args()
    run(args.pred, args.gt, args.tol)


if __name__ == "__main__":
    main()
