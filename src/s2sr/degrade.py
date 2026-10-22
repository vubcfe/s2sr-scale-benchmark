"""MTF-matched degradation used by Wald pairs, the consistency protocol and the LFC layer.

A Gaussian PSF with standard deviation sigma (in fine-grid pixels) has
MTF(f) = exp(-2 pi^2 sigma^2 f^2). Setting MTF at the coarse-grid Nyquist frequency
f_N = 1 / (2 r) equal to the nominal value m gives sigma = (r / pi) * sqrt(-2 ln m).

For an even factor r the centre of a coarse pixel falls between fine pixels, so the
kernel is sampled on a half-pixel grid and applied with stride r. This keeps coarse
and fine grids aligned (no half-pixel shift), matching how Sentinel-2 10 m and 20 m
grids nest.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F


def sigma_from_mtf(mtf_nyquist: float, ratio: int = 2) -> float:
    if not 0.0 < mtf_nyquist < 1.0:
        raise ValueError(f"MTF at Nyquist must be in (0, 1), got {mtf_nyquist}")
    return ratio / math.pi * math.sqrt(-2.0 * math.log(mtf_nyquist))


def gaussian_taps(sigma: float, ratio: int = 2) -> torch.Tensor:
    """1-D kernel sampled so that its centre lies on the centre of a coarse pixel."""
    half = max(ratio, math.ceil(3.0 * sigma + ratio / 2))
    # positions relative to the coarse-pixel centre; offset (ratio-1)/2 handles even ratios
    pos = torch.arange(-half, half + ratio - 1 + 1, dtype=torch.float64) - (ratio - 1) / 2.0
    taps = torch.exp(-0.5 * (pos / sigma) ** 2)
    return (taps / taps.sum()).float()


def degrade(x: torch.Tensor, sigmas: Sequence[float], ratio: int = 2) -> torch.Tensor:
    """Blur each band with its own Gaussian and decimate by `ratio`.

    x: (N, C, H, W) with H, W divisible by `ratio`. Returns (N, C, H / ratio, W / ratio).
    Differentiable, so it can be used inside the LFC layer.
    """
    n, c, h, w = x.shape
    if len(sigmas) != c:
        raise ValueError(f"need one sigma per band: {len(sigmas)} sigmas for {c} bands")
    if h % ratio or w % ratio:
        raise ValueError(f"spatial size {(h, w)} not divisible by ratio {ratio}")
    out = []
    for b, s in enumerate(sigmas):
        k = gaussian_taps(s, ratio).to(x.device, x.dtype)
        klen = k.numel()
        # first tap index aligned with the first fine pixel of the first coarse pixel
        pad_lo = (klen - ratio) // 2
        pad_hi = klen - ratio - pad_lo
        xb = x[:, b:b + 1]
        xb = F.pad(xb, (pad_lo, pad_hi, pad_lo, pad_hi), mode="reflect")
        xb = F.conv2d(xb, k.view(1, 1, 1, -1), stride=(1, ratio))
        xb = F.conv2d(xb, k.view(1, 1, -1, 1), stride=(ratio, 1))
        out.append(xb)
    return torch.cat(out, dim=1)


def upsample(x: torch.Tensor, ratio: int = 2) -> torch.Tensor:
    """Bicubic upsampling with half-pixel alignment (align_corners=False)."""
    return F.interpolate(x, scale_factor=ratio, mode="bicubic", align_corners=False)


def band_sigmas(mtf: dict, bands: Sequence[str], ratio: int = 2) -> list[float]:
    """Look up per-band sigma from an {band: mtf_at_nyquist} mapping."""
    missing = [b for b in bands if b not in mtf]
    if missing:
        raise KeyError(f"no MTF value for bands {missing}")
    return [sigma_from_mtf(float(mtf[b]), ratio) for b in bands]
