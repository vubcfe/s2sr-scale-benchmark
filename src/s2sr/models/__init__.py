"""Model registry: build any benchmark method from its configs/methods.yaml entry."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..bands import TARGET_BANDS
from ..degrade import band_sigmas, upsample
from .atprk import ATPRK
from .common import LowFrequencyConstraint, count_params
from .dsen2 import DSen2
from .edsr_rcan import EDSR, RCAN


class Bicubic(nn.Module):
    trainable = False

    def forward(self, lr, guide):
        return upsample(lr)


class ScrambledGuide(nn.Module):
    """Negative control for RQ3: the guide keeps its channels and statistics but loses its spatial
    correspondence with the 20 m bands (rotated by 180 degrees), at training and at test time."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, lr, guide):
        return self.model(lr, torch.flip(guide, dims=(-2, -1)))


class DSen2Global(nn.Module):
    """Original DSen2 weights (Lanaras et al. 2018; github.com/lanha/DSen2, models/s2_032_lr_1e-04.hdf5),
    trained on Sentinel-2 L1C from globally distributed tiles. Inputs are divided by 2000 (DN), i.e.
    reflectance * 5, and the 20 m bands are bilinearly upsampled (skimage resize, as in the original code)."""
    trainable = False
    SCALE = 10000.0 / 2000.0

    def __init__(self, weights: str):
        super().__init__()
        self.net = DSen2(nf=128, n_blocks=6)
        load_keras_dsen2(self.net, Path(weights))

    def forward(self, lr, guide):
        s = self.SCALE
        up = F.interpolate(lr * s, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.net.head(torch.cat([guide * s, up], dim=1))
        return (up + self.net.tail(self.net.body(x))) / s


def load_keras_dsen2(net: DSen2, path: Path) -> None:
    """Copy the 14 Keras Conv2D layers (head, 6 x 2 residual convs, tail) into `net` in graph order."""
    import h5py

    with h5py.File(path, "r") as f:
        g = f["model_weights"]
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in g.attrs["layer_names"]]
        names = [n for n in names if n.startswith("conv2d")]
        convs = [m for m in net.modules() if isinstance(m, nn.Conv2d)]
        if len(names) != len(convs):
            raise ValueError(f"{len(names)} Keras convs vs {len(convs)} torch convs")
        for name, conv in zip(names, convs):
            k = g[name][name]["kernel:0"][()]        # (kh, kw, cin, cout)
            b = g[name][name]["bias:0"][()]
            if tuple(k.shape[2:]) != (conv.in_channels, conv.out_channels):
                raise ValueError(f"{name}: shape {k.shape} does not match {conv}")
            conv.weight.data.copy_(torch.from_numpy(k.transpose(3, 2, 0, 1).copy()))
            conv.bias.data.copy_(torch.from_numpy(b))


_ARCH = {"dsen2": DSen2, "edsr": EDSR, "rcan": RCAN}
ROOT = Path(__file__).resolve().parents[3]


def build_model(spec: dict, mtf: dict | None = None) -> nn.Module:
    """spec: {"arch": ..., "params": {...}, "lfc": bool, "scramble_guide": bool, "weights": path}."""
    arch = spec["arch"]
    if arch == "bicubic":
        return Bicubic()
    if arch == "atprk":
        if mtf is None:
            raise ValueError("ATPRK needs the MTF table")
        return ATPRK(band_sigmas(mtf, TARGET_BANDS))
    if arch == "dsen2_global":
        return DSen2Global(str(ROOT / spec["weights"]))
    model = _ARCH[arch](**spec.get("params", {}))
    if spec.get("lfc"):
        if mtf is None:
            raise ValueError("LFC needs the MTF table")
        model = LowFrequencyConstraint(model, band_sigmas(mtf, TARGET_BANDS))
    if spec.get("scramble_guide"):
        model = ScrambledGuide(model)
    model.trainable = True
    return model


__all__ = ["build_model", "count_params", "Bicubic", "ATPRK", "DSen2", "DSen2Global", "EDSR", "RCAN"]
