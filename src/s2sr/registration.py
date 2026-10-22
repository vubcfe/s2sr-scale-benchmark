"""Sub-pixel registration of cross-sensor references to the observed Sentinel-2 bands.

Convention used throughout: a shift d = (dy, dx) means `mov(y, x) = ref(y - dy, x - dx)`, i.e. the
content of `mov` sits d pixels further down/right than in `ref`. `fourier_shift(img, d)` applies
such a shift, so `fourier_shift(mov, -d)` aligns `mov` back onto `ref`.

Phase correlation follows Guizar-Sicairos et al. (Opt. Lett. 2008): a coarse FFT peak refined by
a matrix-multiply DFT on an upsampled neighbourhood. Several bands are combined by summing their
normalised cross-power spectra, which is more robust than averaging per-band shifts.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from scipy.signal.windows import tukey


def _prepare(img: np.ndarray, valid: np.ndarray | None) -> np.ndarray:
    """Fill invalid pixels with the band mean, remove the mean and apply a Tukey window.

    The window suppresses the spurious zero-shift peak caused by non-periodic tile edges; a Tukey
    window (flat centre) biases sub-pixel estimates less than a Hann window.
    """
    img = np.asarray(img, dtype=np.float64)
    if img.ndim == 2:
        img = img[None]
    out = np.empty_like(img)
    h, w = img.shape[-2:]
    win = np.outer(tukey(h, 0.3), tukey(w, 0.3))
    for c in range(img.shape[0]):
        b = img[c]
        mu = b[valid].mean() if valid is not None and valid.any() else b.mean()
        b = np.where(valid, b, mu) if valid is not None else b
        out[c] = (b - mu) * win
    return out


def _upsampled_dft(data: np.ndarray, size: int, up: int, offsets) -> np.ndarray:
    """Inverse DFT of `data` on a size x size grid, `up` times finer, starting at `offsets`."""
    for n, off in zip(data.shape[::-1], offsets[::-1]):
        kernel = (np.arange(size) - off)[:, None] * np.fft.fftfreq(n, up)
        data = np.tensordot(np.exp(-2j * np.pi * kernel), data, axes=(1, -1))
    return data


def phase_corr(ref: np.ndarray, mov: np.ndarray, valid: np.ndarray | None = None,
               upsample: int = 100, fmax: float = 0.35) -> tuple[float, float, float]:
    """Shift of `mov` relative to `ref` (see module convention) and the correlation peak in [0, 1].

    ref, mov: (H, W) or (C, H, W) on the same grid. valid: (H, W) bool mask shared by both.
    Only frequencies with |f| <= fmax (cycles per pixel) are used. Near Nyquist, Sentinel-2-like
    images are aliased (MTF ~0.2-0.3 at Nyquist), so those frequencies carry phase that does not
    follow the shift; with full whitening they biased estimates towards zero by about 5% on 10 m
    and 20% on 20 m grids (test_phase_corr_unbiased_on_smooth_coarse_tiles). fmax is deliberately in
    cycles per pixel of the grid being registered, not in metres: aliasing is tied to each grid's own
    sampling (its Nyquist frequency), so the same fraction of Nyquist is excluded on 10 m and 20 m grids.
    Returns (nan, nan, 0.0) when no frequency carries energy (flat or fully invalid input).
    The peak is the normalised phase-correlation maximum; values below ~0.1 indicate an unreliable
    estimate (low texture, change between acquisitions, heavy nodata).
    """
    a, b = _prepare(ref, valid), _prepare(mov, valid)
    fa, fb = np.fft.fft2(a), np.fft.fft2(b)
    cross = fa * np.conj(fb)
    # whiten each band, but drop frequencies with negligible energy in that band: normalising them
    # would give unit weight to pure noise phase (flat bands, filled nodata)
    mag = np.abs(cross)
    h, w = cross.shape[-2:]
    radius = np.hypot(np.fft.fftfreq(h)[:, None], np.fft.fftfreq(w)[None, :])
    keep = (mag > 1e-6 * mag.max(axis=(-2, -1), keepdims=True)) & (radius <= fmax)
    cross = np.where(keep, cross / np.maximum(mag, 1e-30), 0.0).sum(axis=0) / a.shape[0]
    n_keep = int(keep.any(axis=0).sum())
    if n_keep == 0:  # flat or fully invalid input: no estimate
        return float("nan"), float("nan"), 0.0
    cc = np.fft.ifft2(cross)
    py, px = np.unravel_index(np.argmax(np.abs(cc)), cc.shape)
    shift = np.array([py if py <= h // 2 else py - h, px if px <= w // 2 else px - w], dtype=np.float64)
    if upsample > 1:
        shift = np.round(shift * upsample) / upsample
        size = math.ceil(upsample * 1.5)
        centre = np.fix(size / 2.0)
        offsets = centre - shift * upsample
        up = np.conj(_upsampled_dft(np.conj(cross), size, upsample, offsets))
        my, mx = np.unravel_index(np.argmax(np.abs(up)), up.shape)
        shift = shift + (np.array([my, mx]) - centre) / upsample
        peak = float(np.abs(up[my, mx])) / n_keep
    else:
        peak = float(np.abs(cc[py, px])) * h * w / n_keep
    # `cross` = F(ref) conj(F(mov)) peaks at -d; return d in the module convention
    return float(-shift[0]), float(-shift[1]), peak


def subpixel_shift(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Shift content by (dy, dx) pixels with a cubic spline and mirrored borders.

    Unlike `fourier_shift`, nothing wraps around from the opposite edge, so only a border of about
    ceil(|d|) + 2 pixels is affected. Use this for any comparison of pixel values after alignment.
    """
    img = np.asarray(img, dtype=np.float64)
    shift = (0.0,) * (img.ndim - 2) + (dy, dx)
    return ndimage.shift(img, shift, order=3, mode="mirror", prefilter=True)


def fourier_shift(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Shift content by (dy, dx) pixels with periodic boundary (exact for periodic, band-limited
    images; used to build synthetic test data). For real tiles a fractional shift rings across the
    whole tile from the opposite edge, so use `subpixel_shift` there."""
    h, w = img.shape[-2:]
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    ramp = np.exp(-2j * np.pi * (fy * dy + fx * dx))
    return np.real(np.fft.ifft2(np.fft.fft2(img, axes=(-2, -1)) * ramp, axes=(-2, -1)))


def block_shifts(ref: np.ndarray, mov: np.ndarray, valid: np.ndarray, block: int = 64,
                 min_valid: float = 0.95, min_std: float = 0.005, upsample: int = 100,
                 fmax: float = 0.35) -> list[dict]:
    """Phase correlation on non-overlapping blocks; low-texture or incomplete blocks are skipped.

    Returns one dict per usable block: centre (cy, cx), shift (dy, dx), peak.
    """
    ref = ref[None] if ref.ndim == 2 else ref
    mov = mov[None] if mov.ndim == 2 else mov
    h, w = ref.shape[-2:]
    out = []
    for y0 in range(0, h - block + 1, block):
        for x0 in range(0, w - block + 1, block):
            sl = (slice(y0, y0 + block), slice(x0, x0 + block))
            v = valid[sl]
            if v.mean() < min_valid:
                continue
            r, m = ref[(slice(None),) + sl], mov[(slice(None),) + sl]
            # skip only if every band is flat: visible bands over forest are often very flat
            # while NIR still carries enough texture for a reliable estimate
            if min(r[:, v].std(axis=1).max(), m[:, v].std(axis=1).max()) < min_std:
                continue
            dy, dx, pk = phase_corr(r, m, v, upsample, fmax)
            if not np.isfinite(dy):
                continue
            out.append(dict(cy=y0 + block / 2, cx=x0 + block / 2, dy=dy, dx=dx, peak=pk))
    return out


def shift_field_residual(blocks: list[dict], min_peak: float = 0.1) -> dict:
    """How far block shifts depart from (i) one translation and (ii) an affine field.

    Robust translation = median block shift. The affine fit is ordinary least squares of each
    component on (1, y, x). Residuals are RMS over blocks, in the same pixel unit as the shifts.
    """
    bl = [b for b in blocks if b["peak"] >= min_peak]
    res = dict(n_blocks=len(bl), rms_translation=np.nan, rms_affine=np.nan,
               med_dy=np.nan, med_dx=np.nan)
    if len(bl) < 3:
        return res
    d = np.array([[b["dy"], b["dx"]] for b in bl])
    med = np.median(d, axis=0)
    res.update(med_dy=float(med[0]), med_dx=float(med[1]),
               rms_translation=float(np.sqrt(((d - med) ** 2).sum(1).mean())))
    if len(bl) >= 6:
        design = np.array([[1.0, b["cy"], b["cx"]] for b in bl])
        coef, *_ = np.linalg.lstsq(design, d, rcond=None)
        res["rms_affine"] = float(np.sqrt(((d - design @ coef) ** 2).sum(1).mean()))
    return res

