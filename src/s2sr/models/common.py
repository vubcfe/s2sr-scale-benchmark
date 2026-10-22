"""Building blocks shared by the SR models.

All models implement forward(lr, guide) -> sr where
  lr    : (N, 6, h, w)    bands to super-resolve
  guide : (N, 4, 2h, 2w)  guide bands (ignored by unguided models)
  sr    : (N, 6, 2h, 2w)
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from ..degrade import degrade, upsample


def conv3(cin, cout):
    return nn.Conv2d(cin, cout, 3, padding=1)


class ResBlock(nn.Module):
    """EDSR/DSen2 residual block: conv-ReLU-conv with residual scaling, no batch norm."""

    def __init__(self, nf: int, scale: float = 0.1, attention: nn.Module | None = None):
        super().__init__()
        self.body = nn.Sequential(conv3(nf, nf), nn.ReLU(inplace=True), conv3(nf, nf))
        self.att = attention
        self.scale = scale

    def forward(self, x):
        r = self.body(x)
        if self.att is not None:
            r = self.att(r)
        return x + self.scale * r


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation channel attention as in RCAN."""

    def __init__(self, nf: int, reduction: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(nf, max(nf // reduction, 1), 1), nn.ReLU(inplace=True),
            nn.Conv2d(max(nf // reduction, 1), nf, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.net(x)


class LowFrequencyConstraint(nn.Module):
    """Wrap a model so that degrading its output reproduces the low-resolution input.

    y = y_raw + U(lr - D(y_raw)), following the hard constraint idea of SEN2SR
    (Aybar et al., RSE 2026). D is the MTF-matched degradation, U bicubic upsampling.
    D(U(.)) is close to, but not exactly, the identity, so D(y) ~ lr up to that residual.
    """

    def __init__(self, model: nn.Module, sigmas: Sequence[float]):
        super().__init__()
        self.model = model
        self.sigmas = list(sigmas)

    def forward(self, lr, guide):
        y = self.model(lr, guide)
        return y + upsample(lr - degrade(y, self.sigmas))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
