"""Leave-one-embryo-out (LOEO) cross-validation.

Holds out a whole embryo per fold instead of splitting frames randomly — frames
from the same embryo are highly correlated, so a random split leaks information
and inflates scores. LOEO is what surfaced the GT-centred patch-sampling bias.

Usage:
    python -m src.validation.loeo --config config/default.yaml
"""

from __future__ import annotations

import argparse
import copy

import numpy as np

from ..utils import load_config, get_device, set_seed
from ..data.dataset import EmbryoDataset
from ..detection.peaks import peaks_from_heatmap, predict_volume
from .metrics import score_frames


def loeo(cfg):
    all_embryos = list(cfg["data"]["train_embryos"]) + list(cfg["data"]["val_embryos"])
    fold_scores = []

    for held_out in all_embryos:
        train_emb = [e for e in all_embryos if e != held_out]
        print(f"\n=== Fold: hold out {held_out} | train on {train_emb} ===")

        fold_cfg = copy.deepcopy(cfg)
        fold_cfg["data"]["train_embryos"] = train_emb
        fold_cfg["data"]["val_embryos"] = [held_out]

        # Import here to avoid a heavy import when only inspecting the harness.
        from ..train import train as train_model

        ckpt = train_model(fold_cfg)
        score = _evaluate_fold(fold_cfg, held_out, ckpt)
        print(f"[loeo] {held_out}: f1={score['f1']:.4f}")
        fold_scores.append(score["f1"])

    mean = float(np.mean(fold_scores)) if fold_scores else 0.0
    std = float(np.std(fold_scores)) if fold_scores else 0.0
    print(f"\n[loeo] mean F1 = {mean:.4f} ± {std:.4f} across {len(fold_scores)} folds")
    return mean, std


def _evaluate_fold(cfg, held_out, ckpt):
    from ..inference.ensemble import load_model
    device = get_device(cfg["train"]["device"])
    patch_size = tuple(cfg["patch"]["size"])

    ds = EmbryoDataset(cfg["data"]["root"], [held_out],
                       synthetic=cfg["data"]["synthetic"])
    frames = ds.frames()
    model = load_model(ckpt, cfg["model"], device)

    preds, gts = [], []
    for frame in frames:
        hm = predict_volume(model, frame.volume, patch_size, device)
        coords = peaks_from_heatmap(
            hm, threshold=cfg["detect"]["peak_threshold"],
            min_distance=cfg["detect"]["nms_min_distance"],
        )
        preds.append(coords)
        gts.append(frame.centres)
    return score_frames(preds, gts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    loeo(cfg)


if __name__ == "__main__":
    main()
