"""Score a predictions CSV against ground-truth labels.

Usage:
    python -m src.evaluate --pred outputs/pred.csv --gt data/embryo_05/labels.csv
"""

from __future__ import annotations

import argparse

import pandas as pd

from .validation.metrics import score_frames


def run(pred_path, gt_path, tol, embryo=None):
    pred = pd.read_csv(pred_path)
    gt = pd.read_csv(gt_path)

    # labels.csv belongs to one embryo, but a predictions file can hold several.
    # Matching on t alone would mix detections from different embryos together.
    if embryo is not None:
        pred = pred[pred["embryo"] == embryo]
    elif "embryo" in pred.columns and pred["embryo"].nunique() > 1:
        raise SystemExit(
            f"{pred_path} contains embryos {sorted(pred['embryo'].unique())}; "
            "pass --embryo to choose the one that matches the ground truth"
        )

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
    ap.add_argument("--embryo", help="embryo in the predictions file that --gt belongs to")
    args = ap.parse_args()
    run(args.pred, args.gt, args.tol, args.embryo)


if __name__ == "__main__":
    main()
