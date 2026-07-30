"""Run detection (+ optional tracking) over a set of embryo frames and write a
predictions CSV.

Usage:
    python -m src.predict --config config/default.yaml --method dog
    python -m src.predict --config config/default.yaml --method unet --tta --ensemble
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from .utils import load_config, get_device
from .data.dataset import EmbryoDataset
from .detection.blob_baseline import detect_blobs
from .detection.peaks import peaks_from_heatmap, predict_volume
from .inference.tta import predict_volume_tta
from .inference.ensemble import load_model, predict_volume_ensemble
from .tracking.linker import link_frames


def run(cfg, method, use_tta, use_ensemble, do_track, out_path):
    device = get_device(cfg["train"]["device"])
    patch_size = tuple(cfg["patch"]["size"])

    ds = EmbryoDataset(
        cfg["data"]["root"], cfg["data"]["val_embryos"],
        synthetic=cfg["data"]["synthetic"],
    )
    frames = ds.frames()

    single_model = None
    if method == "unet" and not use_ensemble:
        ckpt = os.path.join(cfg["train"]["checkpoint_dir"], f"detector_s{cfg['seed']}.pt")
        single_model = load_model(ckpt, cfg["model"], device)

    rows = []
    detections_per_frame = []
    for frame in frames:
        if method == "dog":
            coords = detect_blobs(
                frame.volume, threshold=0.1,
                min_distance=cfg["detect"]["nms_min_distance"],
            )
        else:
            if use_ensemble:
                ckpts = cfg["inference"]["ensemble_checkpoints"] or [
                    os.path.join(cfg["train"]["checkpoint_dir"], f"detector_s{cfg['seed']}.pt")
                ]
                hm = predict_volume_ensemble(
                    ckpts, cfg["model"], frame.volume, patch_size, device,
                    use_tta=use_tta,
                )
            elif use_tta:
                hm = predict_volume_tta(single_model, frame.volume, patch_size, device)
            else:
                hm = predict_volume(single_model, frame.volume, patch_size, device)
            coords = peaks_from_heatmap(
                hm, threshold=cfg["detect"]["peak_threshold"],
                min_distance=cfg["detect"]["nms_min_distance"],
            )

        detections_per_frame.append(coords)
        for (z, y, x) in coords:
            rows.append({"embryo": frame.embryo, "t": frame.t, "z": z, "y": y, "x": x})

    if do_track:
        tracks = link_frames(
            detections_per_frame,
            max_distance=cfg["track"]["max_link_distance"],
            min_track_length=cfg["track"]["min_track_length"],
        )
        print(f"[predict] {len(tracks)} tracks after linking + filtering")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[predict] wrote {len(rows)} detections -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--method", choices=["dog", "unet"], default="dog")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--ensemble", action="store_true")
    ap.add_argument("--track", action="store_true")
    ap.add_argument("--out", default="outputs/pred.csv")
    args = ap.parse_args()
    run(load_config(args.config), args.method, args.tta, args.ensemble, args.track, args.out)


if __name__ == "__main__":
    main()
