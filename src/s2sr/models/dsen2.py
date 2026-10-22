"""DSen2 (Lanaras et al., 2018) and a re-implementation of DSen2+CA+HPF (Nguyen-Vi et al., 2025).

DSen2: concatenate the guide bands with the bicubic-upsampled 20 m bands, one conv + ReLU,
a stack of residual blocks (scaling 0.1), one conv to the output bands, plus a global skip
from the upsampled input. The paper uses 128 features and 6 blocks.

DSen2+CA+HPF: re-implemented from the paper's description (Nguyen-Vi et al., Front. Remote Sens. 2025,
doi:10.3389/frsen.2025.1644460; checked against the full text on 19/10/2026, no code is published):
channel attention (global average pooling, bottleneck MLP, sigmoid) inside every residual block after the
second convolution and before the residual scaling (their Fig. 4), and a high-frequency enhancement defined
as "the residual between the input image and a smoothed version of itself", integrated early: the Gaussian
high-pass (sigma = 1 px) of the whole input stack (guide and upsampled 20 m bands) is concatenated to the input.
Not specified in the paper and taken from DSen2: 128 features, 6 blocks, residual scaling 0.1; CA reduction 16.
The common training recipe of the benchmark is used instead of theirs (MSE, SGD, batch 128).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..bands import N_GUIDE, N_TARGET
from ..degrade import upsample
from .common import ChannelAttention, ResBlock, conv3


class DSen2(nn.Module):
    def __init__(self, nf: int = 128, n_blocks: int = 6, channel_attention: bool = False,
                 high_pass: bool = False):
        super().__init__()
        self.high_pass = high_pass
        cin = (N_GUIDE + N_TARGET) * (2 if high_pass else 1)
        self.head = nn.Sequential(conv3(cin, nf), nn.ReLU(inplace=True))
        self.body = nn.Sequential(*[
            ResBlock(nf, 0.1, ChannelAttention(nf) if channel_attention else None)
            for _ in range(n_blocks)])
        self.tail = conv3(nf, N_TARGET)

    @staticmethod
    def _high_pass(x, sigma: float = 1.0):
        """x minus its Gaussian-smoothed version (separable 7-tap kernel, reflected borders)."""
        r = 3
        k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=x.dtype, device=x.device) / sigma) ** 2)
        k = k / k.sum()
        c = x.shape[1]
        s = F.pad(x, (r, r, r, r), mode="reflect")
        s = F.conv2d(s, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
        s = F.conv2d(s, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
        return x - s

    def forward(self, lr, guide):
        up = upsample(lr)
        x = torch.cat([guide, up], dim=1)
        feats = [x] + ([self._high_pass(x)] if self.high_pass else [])
        x = self.head(torch.cat(feats, dim=1))
        return up + self.tail(self.body(x))
