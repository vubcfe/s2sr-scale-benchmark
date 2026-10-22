import math

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from s2sr.bands import N_GUIDE, N_TARGET, TARGET_BANDS
from s2sr.data.blocks import make_blocks, min_gap_between_splits
from s2sr.data.dataset import NativeDataset, WaldDataset
from s2sr.degrade import degrade, gaussian_taps, sigma_from_mtf, upsample
from s2sr.evaluate import eval_p1, eval_p2
from s2sr.metrics import ergas, psnr, sam, ssim
from s2sr.models import build_model, count_params
from s2sr.models.common import LowFrequencyConstraint
from s2sr.stats import holm, pairwise_wilcoxon_holm, rank_agreement

MTF = {b: 0.3 for b in ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12"]}


# ---------------------------------------------------------------- degradation
def test_sigma_matches_mtf_at_nyquist():
    s = sigma_from_mtf(0.3, 2)
    f_n = 1 / (2 * 2)
    assert math.isclose(math.exp(-2 * math.pi ** 2 * s ** 2 * f_n ** 2), 0.3, rel_tol=1e-9)
    assert 0.98 < s < 1.0


def test_taps_symmetric_and_normalised():
    k = gaussian_taps(1.0, 2)
    assert torch.allclose(k, k.flip(0))
    assert math.isclose(float(k.sum()), 1.0, rel_tol=1e-6)


def test_degrade_shape_and_constant_preservation():
    x = torch.full((2, 3, 64, 64), 0.25)
    y = degrade(x, [1.0, 1.2, 0.8])
    assert y.shape == (2, 3, 32, 32)
    assert torch.allclose(y, torch.full_like(y, 0.25), atol=1e-6)


def test_degrade_has_no_half_pixel_shift():
    # a linear ramp sampled at fine-pixel centres must map to the coarse-pixel centres
    xs = torch.arange(64, dtype=torch.float32)
    x = xs.view(1, 1, 1, -1).expand(1, 1, 64, 64).contiguous()
    y = degrade(x, [1.0])[0, 0, 16, 4:28]          # away from reflect-padding borders
    expected = (2 * torch.arange(32, dtype=torch.float32) + 0.5)[4:28]
    assert torch.allclose(y, expected, atol=1e-4)


def test_lfc_reproduces_input_after_degradation():
    torch.manual_seed(0)
    model = build_model({"arch": "edsr", "params": {"nf": 16, "n_blocks": 2, "guided": True}}, MTF)
    sig = [sigma_from_mtf(0.3)] * N_TARGET
    lfc = LowFrequencyConstraint(model, sig)
    lr = torch.rand(1, N_TARGET, 32, 32)
    guide = torch.rand(1, N_GUIDE, 64, 64)
    with torch.no_grad():
        raw_err = (degrade(model(lr, guide), sig) - lr).abs()[..., 4:-4, 4:-4].mean()
        lfc_err = (degrade(lfc(lr, guide), sig) - lr).abs()[..., 4:-4, 4:-4].mean()
    assert lfc_err < 0.5 * raw_err


# ---------------------------------------------------------------- splits
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_blocks_keep_buffer_between_splits(seed):
    blocks = make_blocks((0, 0, 51200, 40960), 5120, 1000,
                         {"train": 0.7, "val": 0.15, "test": 0.15}, seed)
    assert len(blocks) == 80
    assert min_gap_between_splits(blocks) >= 1000 - 1e-6
    counts = pd.Series([b.split for b in blocks]).value_counts()
    assert counts["train"] == 56 and counts["val"] == 12 and counts["test"] == 12


def test_heldout_region_is_test_only():
    blocks = make_blocks((0, 0, 20480, 20480), 5120, 1000, {"train": 1.0}, 0, heldout=True)
    assert {b.split for b in blocks} == {"test_ood"}


# ---------------------------------------------------------------- models
@pytest.mark.parametrize("name", ["dsen2", "dsen2_ca_hpf", "lfc_guided", "edsr_u", "edsr_g", "rcan_u", "rcan_g"])
def test_models_forward_shape(name):
    import yaml
    from pathlib import Path

    spec = yaml.safe_load((Path(__file__).parents[1] / "configs/methods.yaml").read_text())["methods"][name]
    m = build_model(spec, MTF)
    lr, guide = torch.rand(1, N_TARGET, 24, 24), torch.rand(1, N_GUIDE, 48, 48)
    with torch.no_grad():
        assert m(lr, guide).shape == (1, N_TARGET, 48, 48)


def test_learned_models_parameter_matched():
    import yaml
    from pathlib import Path

    specs = yaml.safe_load((Path(__file__).parents[1] / "configs/methods.yaml").read_text())["methods"]
    n = {k: count_params(build_model(v, MTF)) for k, v in specs.items()
         if v["arch"] not in ("bicubic", "atprk", "dsen2_global")}
    ref = n["dsen2"]
    for k, v in n.items():
        assert abs(v - ref) / ref <= 0.10, f"{k}: {v} vs dsen2 {ref}"


def test_unguided_model_ignores_guide():
    m = build_model({"arch": "edsr", "params": {"nf": 16, "n_blocks": 2, "guided": False}})
    lr = torch.rand(1, N_TARGET, 16, 16)
    with torch.no_grad():
        a = m(lr, torch.rand(1, N_GUIDE, 32, 32))
        b = m(lr, torch.rand(1, N_GUIDE, 32, 32))
    assert torch.equal(a, b)


# ---------------------------------------------------------------- metrics
def test_metrics_perfect_and_degraded():
    ref = torch.rand(2, N_TARGET, 32, 32) * 0.4 + 0.05
    assert torch.all(sam(ref, ref) < 1e-2)
    assert torch.allclose(ssim(ref, ref), torch.ones(2), atol=1e-4)
    assert torch.all(ergas(ref, ref) < 1e-4)
    noisy = ref + 0.01 * torch.randn_like(ref)
    assert torch.allclose(psnr(noisy, ref), torch.full((2,), 40.0), atol=0.5)
    assert torch.all(sam(noisy, ref) > 0)


def test_sam_ignores_scale():
    ref = torch.rand(1, N_TARGET, 8, 8) + 0.1
    assert sam(2 * ref, ref).item() < 1e-2


# ---------------------------------------------------------------- data + evaluation end to end
def _fake_shard(path, n=6, p=32, seed=0):
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        base = rng.uniform(0.05, 0.4, (n, 1, 2 * p, 2 * p))
        f["g10"] = (np.repeat(base, N_GUIDE, 1) * 10000).astype(np.uint16)
        x20 = base.reshape(n, 1, p, 2, p, 2).mean((3, 5))
        f["x20"] = (np.repeat(x20, N_TARGET, 1) * 10000).astype(np.uint16)
        f["scl"] = np.full((n, p, p), 4, np.uint8)
        f.create_dataset("meta", data=['{"region": "r"}'] * n, dtype=h5py.string_dtype())


def test_end_to_end_bicubic_eval(tmp_path):
    _fake_shard(tmp_path / "test.h5")
    wald = WaldDataset(tmp_path / "test.h5", MTF)
    s = wald[0]
    assert s["lr"].shape == (N_TARGET, 16, 16) and s["guide"].shape == (N_GUIDE, 32, 32)
    assert s["target"].shape == (N_TARGET, 32, 32)
    model = build_model({"arch": "bicubic"})
    p1 = eval_p1(model, wald, "bicubic", 0, "test", torch.device("cpu"), batch_size=4)
    p2 = eval_p2(model, NativeDataset(tmp_path / "test.h5"), "bicubic", 0, "test", torch.device("cpu"), MTF)
    assert len(p1) == 6 and len(p2) == 6
    assert {"psnr", "ssim", "sam", "ergas", "mae_ndre", "mae_ndmi"} <= set(p1.columns)
    assert all(f"rmse_band_{b}" in p2.columns for b in TARGET_BANDS)
    assert p1["psnr"].notna().all()


# ---------------------------------------------------------------- statistics
def test_holm_is_monotone_and_bounded():
    adj = holm([0.01, 0.04, 0.03, 0.5])
    assert adj == pytest.approx([0.04, 0.09, 0.09, 0.5])


def _fake_results(order, n=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for rank, m in enumerate(order):
        for i in range(n):
            rows.append(dict(method=m, seed=0, idx=i, rmse=0.01 * (rank + 1) + 0.001 * rng.standard_normal()))
    return pd.DataFrame(rows)


def test_rank_agreement_detects_same_and_reversed_order():
    order = ["a", "b", "c", "d", "e"]
    same = rank_agreement(_fake_results(order), _fake_results(order, seed=1), "rmse", n_boot=200)
    rev = rank_agreement(_fake_results(order), _fake_results(order[::-1], seed=1), "rmse", n_boot=200)
    assert same["tau"] == pytest.approx(1.0) and rev["tau"] == pytest.approx(-1.0)
    assert same["ci_low"] > 0.5


def test_pairwise_wilcoxon_finds_clear_difference():
    out = pairwise_wilcoxon_holm(_fake_results(["a", "b", "c"]), "rmse")
    assert (out["p_holm"] < 0.05).all()


# ---------------------------------------------------------------- real-reference loaders
def test_recover_native_20m_exact_and_rejects_interpolated():
    from s2sr.data.sen2neon import recover_native_20m

    native = np.random.default_rng(0).integers(0, 10000, (6, 16, 16))
    replicated = native.repeat(2, axis=1).repeat(2, axis=2)
    assert np.array_equal(recover_native_20m(replicated), native)
    smoothed = replicated.copy()
    smoothed[:, 1::2, :] += 1  # breaks the 2x2 replication
    with pytest.raises(ValueError):
        recover_native_20m(smoothed)


def test_sen2venus_band_order_check(tmp_path):
    from s2sr.data.sen2venus import G10, S20, check_band_order

    prefix = tmp_path / "SITE_00XXX_2020-01-01"
    g10 = torch.full((2, 4, 8, 8), 50, dtype=torch.int16)      # B4 ~ 0.005
    torch.save(g10, f"{prefix}{G10}")
    torch.save(torch.full((2, 4, 4, 4), 500, dtype=torch.int16), f"{prefix}{S20}")  # B5 ~ 0.05
    check_band_order(prefix)
    torch.save(torch.full((2, 4, 4, 4), 10, dtype=torch.int16), f"{prefix}{S20}")
    with pytest.raises(ValueError):
        check_band_order(prefix)
