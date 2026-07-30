"""Train the 3D U-Net heatmap detector.

Usage:
    python -m src.train --config config/default.yaml
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .utils import load_config, get_device, set_seed
from .data.dataset import EmbryoDataset
from .data.patches import sample_patches
from .data.transforms import random_augment
from .models.unet3d import UNet3D


class PatchDataset(Dataset):
    """Materialises a pool of (patch, heatmap) pairs sampled from the frames."""

    def __init__(self, frames, cfg, rng):
        self.items = []
        pcfg = cfg["patch"]
        n = cfg["train"]["patches_per_volume"]
        for frame in frames:
            for patch, hm in sample_patches(
                frame,
                patch_size=tuple(pcfg["size"]),
                sigma=pcfg["sigma"],
                n_patches=n,
                gt_centred_ratio=pcfg["gt_centred_ratio"],
                uniform_ratio=pcfg["uniform_ratio"],
                rng=rng,
            ):
                patch, hm = random_augment(patch, hm, rng)
                self.items.append((patch.copy(), hm.copy()))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        patch, hm = self.items[i]
        return torch.from_numpy(patch)[None], torch.from_numpy(hm)[None]


def train(cfg):
    set_seed(cfg["seed"])
    device = get_device(cfg["train"]["device"])
    rng = np.random.default_rng(cfg["seed"])

    ds = EmbryoDataset(
        cfg["data"]["root"], cfg["data"]["train_embryos"],
        synthetic=cfg["data"]["synthetic"],
    )
    frames = ds.frames()
    print(f"[train] {len(frames)} frames from {len(cfg['data']['train_embryos'])} embryos")

    patch_ds = PatchDataset(frames, cfg, rng)
    print(f"[train] {len(patch_ds)} training patches")
    loader = DataLoader(patch_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)

    model = UNet3D(**cfg["model"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    loss_fn = nn.BCELoss()

    os.makedirs(cfg["train"]["checkpoint_dir"], exist_ok=True)
    ckpt_path = os.path.join(
        cfg["train"]["checkpoint_dir"], f"detector_s{cfg['seed']}.pt"
    )

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        running = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        avg = running / max(1, len(patch_ds))
        print(f"[train] epoch {epoch + 1}/{cfg['train']['epochs']}  loss={avg:.4f}")

    torch.save({"model": model.state_dict(), "cfg": cfg["model"]}, ckpt_path)
    print(f"[train] saved checkpoint -> {ckpt_path}")
    return ckpt_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
