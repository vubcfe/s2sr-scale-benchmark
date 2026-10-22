"""ATPRK (Wang et al., IEEE TGRS 2016): area-to-point regression kriging for the Sentinel-2 20 m bands.

Our Python implementation of the synthesized-band scheme, per sample and band b:
  1. Regression. The 20 m band is regressed on the four 10 m bands degraded to 20 m with the PSF of band b,
     x_b = a_0 + sum_k a_k D_b(g_k) + r_b (ordinary least squares over valid 20 m pixels). The regression part
     at 10 m is p_b = a_0 + sum_k a_k g_k; by linearity D_b(p_b) equals the fitted 20 m values.
  2. Area-to-point kriging (ATPK) of the residual r_b. The point covariance is exponential,
     C(d) = c exp(-d / L) with d in 10 m pixels. Coarse pixels integrate fine pixels with the same separable
     Gaussian PSF used for the Wald degradation (s2sr.degrade), so area-to-area and point-to-area covariances
     are exact sums over the PSF weights. L is fitted by deconvolution: the regularised (areal) semivariogram
     implied by L is matched to the empirical semivariogram of r_b at lags 1..N_LAG coarse pixels (the scale c
     has a closed-form least-squares value for each L and does not affect the kriging weights).
     Ordinary kriging with a (2R+1) x (2R+1) window of coarse pixels gives one weight kernel per sub-pixel
     position; with a stationary covariance these kernels are the same for every pixel, so ATPK reduces to four
     correlations and an interleave.
  3. Prediction y_b = p_b + ATPK(r_b).
Because ATPK honours the coarse data under the PSF, D_b(y_b) is close to x_b (coherence), up to the window
truncation and the reflected borders.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from scipy.optimize import minimize_scalar

from ..degrade import degrade, gaussian_taps

RATIO = 2
WINDOW_R = 2      # kriging window (2R+1)^2 coarse pixels
N_LAG = 5         # coarse lags used to fit the variogram
L_BOUNDS = (0.3, 300.0)
L_STEP = 0.02     # log step of the L grid


@lru_cache(maxsize=64)
def _taps(sigma: float) -> np.ndarray:
    return gaussian_taps(sigma, RATIO).double().numpy()


def _footprint(sigma: float):
    """1-D PSF weights and the fine offset of the first tap relative to 2*i (coarse pixel i)."""
    t = _taps(sigma)
    pad_lo = (len(t) - RATIO) // 2
    return t, -pad_lo


def _cov(points_a: np.ndarray, points_b: np.ndarray, L: float) -> np.ndarray:
    d = np.sqrt(((points_a[:, None, :] - points_b[None, :, :]) ** 2).sum(-1))
    return np.exp(-d / L)


def _coarse_support(sigma: float, ci: int, cj: int):
    """Fine-pixel coordinates and weights of the footprint of coarse pixel (ci, cj)."""
    t, off = _footprint(sigma)
    k = np.arange(len(t))
    yy, xx = np.meshgrid(RATIO * ci + off + k, RATIO * cj + off + k, indexing="ij")
    w = np.outer(t, t).ravel()
    return np.stack([yy.ravel(), xx.ravel()], 1).astype(np.float64), w


@lru_cache(maxsize=4096)
def _areal_cov_lag(sigma: float, L: float, lag: int) -> float:
    pa, wa = _coarse_support(sigma, 0, 0)
    pb, wb = _coarse_support(sigma, 0, lag)
    return float(wa @ _cov(pa, pb, L) @ wb)


def _model_semivariogram(sigma: float, L: float) -> np.ndarray:
    c0 = _areal_cov_lag(sigma, L, 0)
    return np.array([c0 - _areal_cov_lag(sigma, L, h) for h in range(1, N_LAG + 1)])


def empirical_semivariogram(r: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Isotropic (row and column directions) semivariogram of r at coarse lags 1..N_LAG and pair counts."""
    g, n = np.zeros(N_LAG), np.zeros(N_LAG)
    for h in range(1, N_LAG + 1):
        for a, b, va, vb in ((r[:, h:], r[:, :-h], valid[:, h:], valid[:, :-h]),
                             (r[h:, :], r[:-h, :], valid[h:, :], valid[:-h, :])):
            m = va & vb
            g[h - 1] += 0.5 * ((a - b)[m] ** 2).sum()
            n[h - 1] += m.sum()
    return g / np.maximum(n, 1), n


def fit_range(sigma: float, gamma: np.ndarray, counts: np.ndarray) -> float:
    """Point-covariance range L (fine pixels) whose regularised semivariogram best matches `gamma`."""
    w = counts / max(counts.sum(), 1)

    def loss(logL):
        model = _model_semivariogram(sigma, _quantise(math.exp(logL)))
        c = (w * model * gamma).sum() / max((w * model * model).sum(), 1e-30)
        return float((w * (c * model - gamma) ** 2).sum())

    res = minimize_scalar(loss, bounds=(math.log(L_BOUNDS[0]), math.log(L_BOUNDS[1])), method="bounded",
                          options=dict(xatol=1e-2))
    return _quantise(math.exp(res.x))


def _quantise(L: float) -> float:
    """Snap L to a logarithmic grid (2 % steps) so that kriging kernels are cached across samples."""
    return float(np.round(math.exp(round(math.log(L) / L_STEP) * L_STEP), 4))


@lru_cache(maxsize=8192)
def _areal_cov_2d(sigma: float, L: float, di: int, dj: int) -> float:
    """Covariance between two coarse pixels (di, dj) coarse pixels apart."""
    pa, wa = _coarse_support(sigma, 0, 0)
    pb, wb = _coarse_support(sigma, di, dj)
    return float(wa @ _cov(pa, pb, L) @ wb)


@lru_cache(maxsize=2048)
def kriging_kernels(sigma: float, L: float) -> np.ndarray:
    """(2, 2, 2R+1, 2R+1) ordinary-kriging weights for the four sub-pixel positions of a coarse pixel."""
    offs = [(i, j) for i in range(-WINDOW_R, WINDOW_R + 1) for j in range(-WINDOW_R, WINDOW_R + 1)]
    n = len(offs)
    Caa = np.array([[_areal_cov_2d(sigma, L, a[0] - b[0], a[1] - b[1]) for b in offs] for a in offs])
    A = np.zeros((n + 1, n + 1))
    A[:n, :n], A[:n, n], A[n, :n] = Caa, 1.0, 1.0
    sup = [_coarse_support(sigma, i, j) for i, j in offs]
    out = np.zeros((RATIO, RATIO, 2 * WINDOW_R + 1, 2 * WINDOW_R + 1))
    for sy in range(RATIO):
        for sx in range(RATIO):
            x = np.array([[sy, sx]], dtype=np.float64)
            rhs = np.append([(_cov(x, p, L)[0] @ w) for p, w in sup], 1.0)
            lam = np.linalg.solve(A, rhs)[:n]
            out[sy, sx] = lam.reshape(2 * WINDOW_R + 1, 2 * WINDOW_R + 1)
    return out


def atpk(r: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    """Downscale a coarse residual field (h, w) to (2h, 2w)."""
    gamma, counts = empirical_semivariogram(r, valid)
    h, w = r.shape
    if counts.min() < 10 or gamma.max() <= 0:
        return np.repeat(np.repeat(np.where(valid, r, 0.0), RATIO, 0), RATIO, 1)
    kern = kriging_kernels(sigma, fit_range(sigma, gamma, counts))
    if valid.all():
        rf = r
    else:   # invalid residuals: nearest valid value (they are excluded from the metrics anyway)
        idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
        rf = r[idx[0], idx[1]]
    out = np.zeros((RATIO * h, RATIO * w))
    for sy in range(RATIO):
        for sx in range(RATIO):
            out[sy::RATIO, sx::RATIO] = ndimage.correlate(rf, kern[sy, sx], mode="reflect")
    return out


def atprk_sample(lr: np.ndarray, guide: np.ndarray, sigmas: Sequence[float],
                 valid20: np.ndarray | None = None) -> np.ndarray:
    """lr (6, h, w), guide (4, 2h, 2w) -> (6, 2h, 2w)."""
    c, h, w = lr.shape
    valid20 = np.ones((h, w), bool) if valid20 is None else valid20
    g = torch.from_numpy(np.ascontiguousarray(guide, dtype=np.float32))[None]
    out = np.empty((c, RATIO * h, RATIO * w))
    for b in range(c):
        dg = degrade(g, [sigmas[b]] * g.shape[1])[0].double().numpy()           # (4, h, w)
        X = np.concatenate([np.ones((1, h, w)), dg]).reshape(1 + dg.shape[0], -1).T
        y = lr[b].astype(np.float64).ravel()
        m = valid20.ravel()
        if m.sum() < X.shape[1] * 10:   # too few valid pixels for the regression: use all
            m = np.ones_like(m)
        coef, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
        p10 = coef[0] + np.tensordot(coef[1:], guide.astype(np.float64), axes=1)
        r = lr[b] - (X @ coef).reshape(h, w)
        out[b] = p10 + atpk(r, valid20, sigmas[b])
    return out


class ATPRK(nn.Module):
    trainable = False

    def __init__(self, sigmas: Sequence[float]):
        super().__init__()
        self.sigmas = [float(s) for s in sigmas]

    @torch.no_grad()
    def forward(self, lr, guide, valid=None):
        """valid: optional (N, h, w) bool mask on the grid of `lr`; invalid pixels are left out of the
        regression and the variogram and are set to zero in the kriged residual."""
        dev = lr.device
        lr_np, g_np = lr.double().cpu().numpy(), guide.double().cpu().numpy()
        v = None if valid is None else valid.bool().cpu().numpy()
        out = np.stack([atprk_sample(lr_np[i], g_np[i], self.sigmas, None if v is None else v[i])
                        for i in range(lr_np.shape[0])])
        return torch.from_numpy(out).float().to(dev)
