"""EDSR (Lim et al., 2017) and RCAN (Zhang et al., 2018) adapted to Sentinel-2 guided SR.

Adaptations, identical for both: the pixel-shuffle upsampler is removed because the input is
already bicubic-upsampled to the output grid; a global skip from that bicubic input is added;
`guided=False` uses only the 6 upsampled bands, `guided=True` concatenates the 4 guide bands.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..bands import N_GUIDE, N_TARGET
from ..degrade import upsample
from .common import ChannelAttention, ResBlock, conv3


class _Adapted(nn.Module):
    def __init__(self, nf: int, body: nn.Module, guided: bool):
        super().__init__()
        self.guided = guided
        self.head = conv3(N_TARGET + (N_GUIDE if guided else 0), nf)
        self.body = body
        self.body_tail = conv3(nf, nf)
        self.tail = conv3(nf, N_TARGET)

    def forward(self, lr, guide):
        up = upsample(lr)
        x = torch.cat([up, guide], dim=1) if self.guided else up
        h = self.head(x)
        h = h + self.body_tail(self.body(h))
        return up + self.tail(h)


class EDSR(_Adapted):
    def __init__(self, nf: int = 80, n_blocks: int = 16, guided: bool = True):
        body = nn.Sequential(*[ResBlock(nf, 0.1) for _ in range(n_blocks)])
        super().__init__(nf, body, guided)


class _RCAB(ResBlock):
    def __init__(self, nf, reduction=16):
        super().__init__(nf, scale=1.0, attention=ChannelAttention(nf, reduction))


class _ResidualGroup(nn.Module):
    def __init__(self, nf, n_rcab, reduction):
        super().__init__()
        self.body = nn.Sequential(*[_RCAB(nf, reduction) for _ in range(n_rcab)], conv3(nf, nf))

    def forward(self, x):
        return x + self.body(x)


class RCAN(_Adapted):
    def __init__(self, nf: int = 64, n_groups: int = 5, n_rcab: int = 5, reduction: int = 16,
                 guided: bool = True):
        body = nn.Sequential(*[_ResidualGroup(nf, n_rcab, reduction) for _ in range(n_groups)])
        super().__init__(nf, body, guided)
