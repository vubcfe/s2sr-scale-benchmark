import numpy as np
import pytest

from s2sr.registration import block_shifts, fourier_shift, phase_corr, shift_field_residual


def _texture(h=128, w=128, c=1, seed=0):
    """Scene-like texture (1/f^2 spectrum), cut from a larger periodic image so tile edges are not
    periodic. Returns a function mapping a shift (dy, dx) to the cropped, shifted tile."""
    rng = np.random.default_rng(seed)
    n = 4 * max(h, w)
    fy, fx = np.fft.fftfreq(n)[:, None], np.fft.fftfreq(n)[None, :]
    big = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((c, n, n)))
                               / (1 + (fy ** 2 + fx ** 2) / 0.05 ** 2)))
    return lambda dy=0.0, dx=0.0: fourier_shift(big, dy, dx)[:, n // 3:n // 3 + h, n // 3:n // 3 + w]


@pytest.mark.parametrize("d", [(0.0, 0.0), (0.3, -0.7), (-1.25, 0.5), (2.6, 1.1)])
def test_phase_corr_recovers_subpixel_shift_and_sign(d):
    tex = _texture(c=3)
    ref, mov = tex(), tex(*d)
    dy, dx, peak = phase_corr(ref, mov, upsample=50)
    assert abs(dy - d[0]) < 0.05 and abs(dx - d[1]) < 0.05
    assert peak > 0.3


def test_phase_corr_unbiased_on_smooth_coarse_tiles():
    """Aliased near-Nyquist frequencies bias smooth (MTF-blurred, decimated) tiles towards zero."""
    import torch
    from s2sr.degrade import degrade
    rng = np.random.default_rng(4)
    n = 1024
    fy, fx = np.fft.fftfreq(n)[:, None], np.fft.fftfreq(n)[None, :]
    big = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((2, n, n))) / (1 + (fy ** 2 + fx ** 2) / 0.01 ** 2)))
    d = (1.6, -1.2)  # in 2.5 m px, i.e. (0.2, -0.15) px of a 20 m grid
    to20 = lambda z: degrade(torch.from_numpy(z[:, :512, :512]).float()[None], [4.0, 4.0], 8)[0].numpy()
    ref, mov = to20(big), to20(fourier_shift(big, *d))
    dy0, dx0, _ = phase_corr(ref, mov, fmax=0.75)           # all frequencies, incl. aliased ones
    dy, dx, _ = phase_corr(ref, mov)
    assert abs(dy0 - 0.2) > 0.02                             # the bias is real ...
    assert abs(dy - 0.2) < 0.01 and abs(dx + 0.15) < 0.01    # ... and the frequency limit removes it


def test_fourier_shift_convention():
    img = np.zeros((32, 32))
    img[10, 12] = 1.0
    out = fourier_shift(img, 3, -2)
    assert np.unravel_index(np.argmax(out), out.shape) == (13, 10)


def test_phase_corr_with_nodata_mask():
    tex = _texture(c=2, seed=1)
    ref, mov = tex(), tex(0.5, -0.4)
    valid = np.ones(ref.shape[-2:], bool)
    valid[:30, :40] = False
    mov[:, ~valid] = 0.0
    dy, dx, _ = phase_corr(ref, mov, valid, upsample=50)
    assert abs(dy - 0.5) < 0.1 and abs(dx + 0.4) < 0.1


def test_block_shifts_detect_translation_vs_affine_field():
    tex = _texture(256, 256, c=2, seed=2)
    ref, mov = tex(), tex(0.6, -0.3)
    valid = np.ones((256, 256), bool)
    bl = block_shifts(ref, mov, valid, block=64, min_std=0.0)
    res = shift_field_residual(bl, min_peak=0.0)
    assert res["n_blocks"] == 16
    assert abs(res["med_dy"] - 0.6) < 0.05 and abs(res["med_dx"] + 0.3) < 0.05
    assert res["rms_translation"] < 0.05


def test_block_shifts_keep_blocks_when_only_some_bands_are_flat():
    tex = _texture(128, 128, c=2, seed=3)
    ref, mov = tex(), tex(0.4, 0.2)
    ref[0] *= 1e-3  # one nearly flat band, as visible bands over dense forest
    mov[0] *= 1e-3
    ref[1] *= 0.05 / ref[1].std()
    mov[1] *= 0.05 / mov[1].std()
    bl = block_shifts(ref, mov, np.ones((128, 128), bool), block=64, min_std=0.005)
    assert len(bl) == 4


def test_phase_corr_returns_nan_for_flat_or_invalid_input():
    flat = np.ones((2, 64, 64))
    dy, dx, pk = phase_corr(flat, flat)
    assert np.isnan(dy) and np.isnan(dx) and pk == 0.0
    tex = _texture(64, 64, c=2)()
    dy, dx, pk = phase_corr(tex, tex, np.zeros((64, 64), bool))
    assert np.isnan(dy) and pk == 0.0

