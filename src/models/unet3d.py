"""A compact 3D U-Net that predicts a per-voxel cell-centre heatmap.

Kept intentionally small (configurable base width / depth) to train under a
tight compute budget. Output is a single-channel sigmoid heatmap.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, depth=3):
        super().__init__()
        self.depth = depth
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        ch = in_channels
        widths = [base_channels * (2 ** i) for i in range(depth)]

        # encoder
        for w in widths:
            self.downs.append(DoubleConv(ch, w))
            self.pools.append(nn.MaxPool3d(2))
            ch = w

        # bottleneck
        self.bottleneck = DoubleConv(ch, ch * 2)
        ch = ch * 2

        # decoder
        for w in reversed(widths):
            self.ups.append(nn.ConvTranspose3d(ch, w, 2, stride=2))
            self.up_convs.append(DoubleConv(w * 2, w))
            ch = w

        self.head = nn.Conv3d(ch, 1, 1)

    def forward(self, x):
        skips = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, up_conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = up(x)
            x = _match_and_cat(x, skip)
            x = up_conv(x)

        return torch.sigmoid(self.head(x))


def _match_and_cat(x, skip):
    """Centre-crop/pad `x` to `skip`'s spatial size, then concatenate on channels."""
    if x.shape[2:] != skip.shape[2:]:
        diffs = [s - xs for s, xs in zip(skip.shape[2:], x.shape[2:])]
        pad = []
        for d in reversed(diffs):
            pad.extend([d // 2, d - d // 2])
        x = nn.functional.pad(x, pad)
    return torch.cat([skip, x], dim=1)
