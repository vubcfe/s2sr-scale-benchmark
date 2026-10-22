"""Contrast-invariant structural metrics used as primary endpoints of the P1 benchmark (P1 paper, Sec. 3.5).

High-pass: HP(z) = z - G_sigma * z with sigma = 20 m (1 px on the 20 m grid, 2 px on the 10 m grid); the
Gaussian smoothing is normalised by the valid mask. Statistics use only pixels whose whole 4-sigma footprint is
valid and that are at least `border` pixels from the patch edge.

  hp_corr    : Pearson correlation of HP(pred) and HP(ref) per band; invariant to a per-band gain and offset
               of either image (primary).
  edge_error : Sobel gradient-magnitude maxima (3x3 non-maximum suppression) above the q-quantile of the valid
               pixels form edge maps; the error is the mean of the two directed mean distances between the
               maps, each distance capped at `max_dist` pixels (primary; depends on positions, not contrast).
  hp_gain    : sd(HP(pred)) / sd(HP(ref)) per band (secondary; depends on the reference contrast).
Failure scores: a prediction without high-pass variance (while the reference has some) gets hp_corr = 0, and one
without detected edges gets the maximal edge error, so that flat outputs are penalised rather than dropped. NaN is
returned only when the reference itself has no variance or no edges, or too few pixels are valid.

All functions take (C, H, W) numpy arrays of one sample and a (H, W) bool mask and return one value per band
(NaN when too few pixels remain).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

HP_SIGMA_M = 20.0
MIN_PIXELS = 64


def sigma_px(grid_m: float) -> float:
    return HP_SIGMA_M / grid_m


def _smooth(img, sigma, valid):
    num = ndimage.gaussian_filter(np.where(valid, img, 0.0), sigma)
    den = ndimage.gaussian_filter(valid.astype(np.float64), sigma)
    return np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)


def stat_mask(valid: np.ndarray, sigma: float, border: int) -> np.ndarray:
    m = ndimage.binary_erosion(valid, structure=np.ones((3, 3), bool), iterations=int(np.ceil(4 * sigma)),
                               border_value=0)
    if border:
        m[:border], m[-border:], m[:, :border], m[:, -border:] = False, False, False, False
    return m


def highpass(z: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return np.stack([z[k] - _smooth(z[k], sigma, valid) for k in range(z.shape[0])])


def _edges(band: np.ndarray, m: np.ndarray, q: float) -> np.ndarray:
    gy, gx = ndimage.sobel(band, 0), ndimage.sobel(band, 1)
    mag = np.hypot(gx, gy)
    thr = np.quantile(mag[m], q)
    return (mag >= ndimage.maximum_filter(mag, size=3)) & (mag > thr) & m


def structural(pred: np.ndarray, ref: np.ndarray, valid: np.ndarray, grid_m: float, border: int = 4,
               q: float = 0.9, max_dist: float = 3.0) -> dict:
    """{hp_corr, edge, hp_gain}: arrays of shape (C,) for one sample."""
    s = sigma_px(grid_m)
    m = stat_mask(np.asarray(valid, bool), s, border)
    c = pred.shape[0]
    nan = np.full(c, np.nan)
    if m.sum() < MIN_PIXELS:
        return dict(hp_corr=nan, edge=nan.copy(), hp_gain=nan.copy())
    bad = ~np.isfinite(pred[:, m]).all(axis=1)          # a method that outputs NaN/Inf gets failure scores
    if bad.any():
        pred = np.where(np.isfinite(pred), pred, 0.0)
    hp = highpass(pred, valid, s)[:, m]
    hr = highpass(ref, valid, s)[:, m]
    hp0 = hp - hp.mean(axis=1, keepdims=True)
    hr0 = hr - hr.mean(axis=1, keepdims=True)
    sp, sr = np.sqrt((hp0 ** 2).sum(1)), np.sqrt((hr0 ** 2).sum(1))
    corr = np.where(sp * sr > 1e-20, (hp0 * hr0).sum(1) / np.maximum(sp * sr, 1e-20), np.nan)
    corr = np.where((sp <= 1e-10) & (sr > 1e-10), 0.0, corr)
    sd_r = hr.std(1)
    gain = np.where(sd_r > 1e-20, hp.std(1) / np.maximum(sd_r, 1e-20), np.nan)
    edge = np.empty(c)
    for k in range(c):
        ep, er = _edges(np.asarray(pred[k], np.float64), m, q), _edges(np.asarray(ref[k], np.float64), m, q)
        if not er.any():
            edge[k] = np.nan
            continue
        if not ep.any():
            edge[k] = max_dist
            continue
        d_to_r = ndimage.distance_transform_edt(~er)
        d_to_p = ndimage.distance_transform_edt(~ep)
        edge[k] = 0.5 * (np.minimum(d_to_r[ep], max_dist).mean() + np.minimum(d_to_p[er], max_dist).mean())
    corr[bad], edge[bad], gain[bad] = 0.0, max_dist, np.nan
    return dict(hp_corr=corr, edge=edge, hp_gain=gain)


def nd(x: np.ndarray, a: int, b: int, eps: float = 1e-6) -> np.ndarray:
    return (x[a] - x[b]) / (x[a] + x[b] + eps)
