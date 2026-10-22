"""Per-sample quality metrics. Every function takes (N, C, H, W) reflectance tensors and returns
one value per sample (shape (N,) or (N, C)), so paired statistics can be computed later.

Conventions (stated in the paper):
  PSNR  : peak = 1.0 reflectance, computed over all bands jointly.
  SSIM  : Gaussian window 11, sigma 1.5, data range 1.0, averaged over bands.
  SAM   : mean spectral angle in degrees over pixels with non-zero spectra.
  ERGAS : 100 * (h / l) * sqrt(mean_b (RMSE_b / mean_b)^2), h / l = 10 / 20 = 0.5.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .bands import target_index


def crop(x: torch.Tensor, border: int) -> torch.Tensor:
    return x[..., border:-border, border:-border] if border > 0 else x


def rmse_per_band(pred, ref, mask=None):
    se = (pred - ref) ** 2
    if mask is None:
        return se.mean(dim=(-2, -1)).sqrt()
    m = mask.unsqueeze(1).float()
    return ((se * m).sum(dim=(-2, -1)) / m.sum(dim=(-2, -1)).clamp_min(1)).sqrt()


def psnr(pred, ref, peak: float = 1.0, mask=None):
    se = (pred - ref) ** 2
    if mask is None:
        mse = se.mean(dim=(1, 2, 3))
    else:
        m = mask.unsqueeze(1).float().expand_as(se)
        mse = (se * m).sum(dim=(1, 2, 3)) / m.sum(dim=(1, 2, 3)).clamp_min(1)
    return 10.0 * torch.log10(peak ** 2 / mse.clamp_min(1e-12))


def _gauss_window(size=11, sigma=1.5, device=None, dtype=None):
    x = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(pred, ref, data_range: float = 1.0, mask=None):
    """Mean SSIM; with `mask` (N, H, W) only windows lying entirely on valid pixels are averaged."""
    n, c, h, w = pred.shape
    win = _gauss_window(device=pred.device, dtype=pred.dtype).expand(c, 1, 11, 11)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu_x = F.conv2d(pred, win, groups=c)
    mu_y = F.conv2d(ref, win, groups=c)
    sxx = F.conv2d(pred * pred, win, groups=c) - mu_x ** 2
    syy = F.conv2d(ref * ref, win, groups=c) - mu_y ** 2
    sxy = F.conv2d(pred * ref, win, groups=c) - mu_x * mu_y
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    if mask is None:
        return s.mean(dim=(1, 2, 3))
    inv = (~mask.bool()).float().unsqueeze(1)
    ok = (F.max_pool2d(inv, 11, stride=1) == 0).float()          # window fully valid (valid conv size)
    cnt = ok.sum(dim=(1, 2, 3)) * c
    out = (s * ok).sum(dim=(1, 2, 3)) / cnt.clamp_min(1)
    return torch.where(cnt > 0, out, torch.full_like(out, float("nan")))


def sam(pred, ref, mask=None, eps: float = 1e-8):
    dot = (pred * ref).sum(dim=1)
    norm = pred.norm(dim=1) * ref.norm(dim=1)
    valid = norm > eps
    if mask is not None:
        valid = valid & mask.bool()
    ang = torch.acos((dot / norm.clamp_min(eps)).clamp(-1.0, 1.0)) * (180.0 / math.pi)
    ang = torch.where(valid, ang, torch.zeros_like(ang))
    return ang.sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp_min(1)


def ergas(pred, ref, ratio: float = 0.5, mask=None):
    r = rmse_per_band(pred, ref, mask)
    if mask is None:
        mu = ref.mean(dim=(-2, -1)).clamp_min(1e-6)
    else:
        m = mask.unsqueeze(1).float()
        mu = ((ref * m).sum(dim=(-2, -1)) / m.sum(dim=(-2, -1)).clamp_min(1)).clamp_min(1e-6)
    return 100.0 * ratio * ((r / mu) ** 2).mean(dim=1).sqrt()


def nd_index(x, a: str, b: str, eps: float = 1e-6):
    xa, xb = x[:, target_index(a)], x[:, target_index(b)]
    return (xa - xb) / (xa + xb + eps)


def index_mae(pred, ref, mask, a: str, b: str):
    """MAE of a normalised-difference index on masked (e.g. vegetated) pixels; NaN if empty."""
    d = (nd_index(pred, a, b) - nd_index(ref, a, b)).abs()
    m = mask.float()
    cnt = m.sum(dim=(1, 2))
    out = (d * m).sum(dim=(1, 2)) / cnt.clamp_min(1)
    return torch.where(cnt > 0, out, torch.full_like(out, float("nan")))


def full_reference(pred, ref, veg_mask=None, border: int = 4, valid=None) -> dict:
    """All P1 metrics for 6-band outputs. Returns {name: tensor (N,) or (N, C)}.
    `valid` (N, H, W) masks every metric (SSIM over windows lying entirely on valid pixels)."""
    pred, ref = crop(pred, border), crop(ref, border)
    v = crop(valid, border).bool() if valid is not None else None
    out = dict(psnr=psnr(pred, ref, mask=v), ssim=ssim(pred, ref, mask=v), sam=sam(pred, ref, v),
               ergas=ergas(pred, ref, mask=v), rmse_band=rmse_per_band(pred, ref, v))
    if veg_mask is not None:
        veg = crop(veg_mask, border)
        if v is not None:
            veg = veg.bool() & v
        out["mae_ndre"] = index_mae(pred, ref, veg, "B8A", "B05")
        out["mae_ndmi"] = index_mae(pred, ref, veg, "B8A", "B11")
    return out
