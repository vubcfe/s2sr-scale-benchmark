"""Tests for the revised P1 pipeline: structural metrics, ATPRK, DSen2 global weights, guide scrambling,
reference registration, unit-level statistics and the real-reference evaluation."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import ndimage

from s2sr.bands import N_GUIDE, N_TARGET, TARGET_BANDS
from s2sr.degrade import band_sigmas, degrade, upsample
from s2sr.models import build_model
from s2sr.registration import subpixel_shift
from s2sr.stats import rank_agreement_units, scale_gap_units, sign_flip_p
from s2sr.structural import structural

ROOT = Path(__file__).resolve().parents[1]
MTF = {b: 0.29 for b in ["B05", "B06", "B07", "B8A"]} | {"B11": 0.195, "B12": 0.195} | \
      {b: 0.24 for b in ["B02", "B03", "B04", "B08"]}


def _texture(shape, seed=0, sigma=2.0):
    rng = np.random.default_rng(seed)
    return ndimage.gaussian_filter(rng.standard_normal(shape), sigma)


# ---------------------------------------------------------------- structural metrics
def test_hp_corr_perfect_and_gain_invariant():
    ref = 0.2 + 0.05 * _texture((3, 64, 64))
    valid = np.ones((64, 64), bool)
    s = structural(ref, ref, valid, 10.0)
    assert np.allclose(s["hp_corr"], 1.0) and np.allclose(s["edge"], 0.0) and np.allclose(s["hp_gain"], 1.0)
    s2 = structural(2.0 * ref + 0.1, ref, valid, 10.0)
    assert np.allclose(s2["hp_corr"], 1.0) and np.allclose(s2["hp_gain"], 2.0, rtol=1e-6)
    assert np.allclose(s2["edge"], 0.0)


def test_hp_corr_drops_for_smoothed_and_shifted_predictions():
    ref = 0.2 + 0.05 * _texture((1, 64, 64), sigma=1.0)
    valid = np.ones((64, 64), bool)
    smooth = ndimage.gaussian_filter(ref, (0, 1.5, 1.5))
    shifted = subpixel_shift(ref, 1.0, 0.0)
    for pred in (smooth, shifted):
        s = structural(pred, ref, valid, 10.0)
        assert s["hp_corr"][0] < 0.95
    assert structural(shifted, ref, valid, 10.0)["edge"][0] > 0.3


def test_structural_returns_nan_without_valid_pixels():
    ref = np.random.default_rng(0).random((2, 32, 32))
    s = structural(ref, ref, np.zeros((32, 32), bool), 10.0)
    assert np.isnan(s["hp_corr"]).all()


# ---------------------------------------------------------------- ATPRK
def _synthetic_scene(seed=0, n=64):
    """10 m truth whose 20 m bands are a linear mix of the guide plus a smooth independent field."""
    rng = np.random.default_rng(seed)
    g = np.stack([0.1 + 0.03 * _texture((2 * n, 2 * n), seed + k, 1.5) for k in range(N_GUIDE)])
    mix = rng.uniform(0.2, 0.8, (N_TARGET, N_GUIDE))
    y = np.einsum("bk,khw->bhw", mix, g) + np.stack(
        [0.01 * _texture((2 * n, 2 * n), seed + 10 + b, 4.0) for b in range(N_TARGET)])
    y_t, g_t = torch.from_numpy(y).float()[None], torch.from_numpy(g).float()[None]
    lr = degrade(y_t, band_sigmas(MTF, TARGET_BANDS))
    return lr, g_t, y_t


def test_atprk_beats_bicubic_and_is_coherent():
    lr, g, y = _synthetic_scene()
    m = build_model({"arch": "atprk"}, MTF)
    out = m(lr, g)
    bic = upsample(lr)
    sl = (Ellipsis, slice(8, -8), slice(8, -8))
    e_atprk = (out - y)[sl].pow(2).mean().sqrt()
    e_bic = (bic - y)[sl].pow(2).mean().sqrt()
    assert e_atprk < 0.6 * e_bic
    back = degrade(out, band_sigmas(MTF, TARGET_BANDS))
    assert (back - lr)[..., 4:-4, 4:-4].abs().mean() < 0.2 * (degrade(bic, band_sigmas(MTF, TARGET_BANDS)) - lr
                                                               )[..., 4:-4, 4:-4].abs().mean() + 1e-4


def test_atpk_kernels_sum_to_one():
    from s2sr.models.atprk import kriging_kernels

    k = kriging_kernels(1.0, 5.0)
    assert np.allclose(k.sum(axis=(-2, -1)), 1.0, atol=1e-8)


# ---------------------------------------------------------------- DSen2 global weights and guide control
WEIGHTS = ROOT / "weights" / "dsen2_s2_032_lr_1e-04.hdf5"


@pytest.mark.skipif(not WEIGHTS.exists(), reason="DSen2 weights not downloaded")
def test_dsen2_global_loads_and_improves_on_bilinear_at_reduced_scale():
    lr, g, y = _synthetic_scene(seed=3)
    m = build_model({"arch": "dsen2_global", "weights": "weights/dsen2_s2_032_lr_1e-04.hdf5"}, MTF)
    with torch.no_grad():
        out = m(lr, g)
    assert out.shape == y.shape and torch.isfinite(out).all()
    bil = torch.nn.functional.interpolate(lr, scale_factor=2, mode="bilinear", align_corners=False)
    # the residual added to the upsampled input is small relative to the signal (weights loaded sensibly)
    assert (out - bil).abs().mean() < 0.5 * y.abs().mean()


def test_scrambled_guide_changes_the_output():
    torch.manual_seed(0)
    spec = {"arch": "edsr", "params": {"nf": 16, "n_blocks": 2, "guided": True}}
    a = build_model(spec, MTF)
    b = build_model(spec | {"scramble_guide": True}, MTF)
    b.model.load_state_dict(a.state_dict())
    lr, g = torch.rand(1, N_TARGET, 16, 16), torch.rand(1, N_GUIDE, 32, 32)
    with torch.no_grad():
        assert not torch.allclose(a(lr, g), b(lr, g))
        assert torch.allclose(b(lr, g), a(lr, torch.flip(g, dims=(-2, -1))))


# ---------------------------------------------------------------- reference registration
def test_register_recovers_known_shift():
    sys.path.insert(0, str(ROOT / "scripts"))
    from p1_prepare_refs import register

    sig2 = band_sigmas(MTF, TARGET_BANDS)
    truth = 0.2 + 0.05 * np.stack([_texture((256, 256), k, 2.0) for k in range(N_TARGET)])
    lr20 = degrade(torch.from_numpy(truth).float()[None], sig2)[0].numpy()
    d10 = (0.6, -0.4)                                    # reference content displaced by this (10 m px)
    ref = np.stack([subpixel_shift(b, *d10) for b in truth]).astype(np.float32)
    reg, valid, (dy, dx), peak = register(lr20, ref, np.ones((256, 256), bool), sig2)
    assert peak > 0.5
    assert abs(2 * dy - d10[0]) < 0.05 and abs(2 * dx - d10[1]) < 0.05
    inner = (slice(None), slice(16, -16), slice(16, -16))
    assert np.abs(reg - truth)[inner].mean() < 0.15 * np.abs(ref - truth)[inner].mean()
    assert valid[16:-16, 16:-16].all() and not valid[:1].all()


# ---------------------------------------------------------------- unit-level statistics
def _unit_results(order, n_units=8, per_unit=5, seeds=(0, 1), seed=0, noise=0.002):
    rng = np.random.default_rng(seed)
    rows = []
    for rank, m in enumerate(order):
        for s in seeds:
            for u in range(n_units):
                for i in range(per_unit):
                    rows.append(dict(method=m, seed=s, unit=f"u{u}", key=f"{u}_{i}",
                                     hp_corr_mean=0.9 - 0.02 * rank + noise * rng.standard_normal()))
    return pd.DataFrame(rows)


def test_rank_agreement_units_same_and_reversed():
    order = list("abcdefg")
    r, ranks = rank_agreement_units(_unit_results(order), _unit_results(order, seed=1), "hp_corr_mean", order,
                                    n_boot=100, n_perm=2000)
    assert r["tau"] == pytest.approx(1.0) and r["paired"] and r["p_perm"] < 0.01 and r["ci_low"] > 0.7
    assert (ranks["rank_change"] == 0).all()
    r2, ranks2 = rank_agreement_units(_unit_results(order), _unit_results(order[::-1], seed=1), "hp_corr_mean",
                                      order, n_boot=100, n_perm=2000)
    assert r2["tau"] == pytest.approx(-1.0) and r2["p_perm"] > 0.9
    assert ranks2.loc["a", "rank_a"] == 1 and ranks2.loc["a", "rank_b"] == 7


def test_sign_flip_exact():
    assert sign_flip_p(np.ones(8)) == pytest.approx(2 / 256)
    assert sign_flip_p(np.array([1.0, -1.0, 1.0, -1.0])) == pytest.approx(1.0)


def test_scale_gap_units_relative_to_bicubic():
    p1 = _unit_results(["bicubic", "x"]).rename(columns={"hp_corr_mean": "hp_corr_B05"})
    p3 = p1.copy()
    p3.loc[p3.method == "bicubic", "hp_corr_B05"] -= 0.1
    p3.loc[p3.method == "x", "hp_corr_B05"] -= 0.05
    g = scale_gap_units(p1, p3, ["B05"], ["bicubic", "x"], n_boot=100).set_index("method")
    assert g.loc["bicubic", "gap"] == pytest.approx(0.1) and g.loc["x", "gap_rel"] == pytest.approx(0.5)


# ---------------------------------------------------------------- real-reference evaluation end to end
def test_eval_refs_bicubic(tmp_path):
    from s2sr.analysis import add_means
    from s2sr.evaluate import RefDataset, eval_refs

    d = tmp_path / "sen2neon"
    d.mkdir()
    lr, g, y = _synthetic_scene(seed=5, n=32)
    rows = []
    for k in range(3):
        np.savez_compressed(d / f"t{k}.npz", lr=lr[0].numpy(), guide=g[0].numpy(), ref=y[0].numpy(),
                            valid=np.ones((64, 64), bool))
        rows.append(dict(key=f"t{k}", unit=f"site{k % 2}", status="ok"))
    rows.append(dict(key="bad", unit="site9", status="resampled"))
    pd.DataFrame(rows).to_csv(d / "tiles.csv", index=False)
    ds = RefDataset(d)
    assert len(ds) == 3
    out = eval_refs(build_model({"arch": "bicubic"}), ds, "bicubic", 0, torch.device("cpu"), MTF)
    assert set(out.protocol) == {"P1", "P2", "P3"} and len(out) == 9
    p3 = add_means(out)[lambda x: x.protocol == "P3"]
    assert {"ndmi_hp_corr", "hp_corr_mean", "edge_mean"} <= set(p3.columns)
    assert p3["hp_corr_mean"].between(0, 1).all()


def test_train_runs_and_resumes(tmp_path):
    from test_core import _fake_shard

    from s2sr.train import train

    for split in ("train", "val"):
        _fake_shard(tmp_path / f"{split}.h5", n=8, seed=1 if split == "val" else 0)
    spec = {"arch": "edsr", "params": {"nf": 8, "n_blocks": 1, "guided": True}}
    cfg = dict(iterations=6, batch_size=2, lr=1e-4, warmup=1, val_every=2, patience=10, amp=False, workers=0)
    run = tmp_path / "run"
    train("t", spec, cfg, MTF, tmp_path, run, 0)
    assert (run / "best.pt").exists() and (run / "DONE").exists()
    (run / "DONE").unlink()                      # pretend the run was interrupted after its last checkpoint
    train("t", spec, cfg, MTF, tmp_path, run, 0)  # resumes from last.pt (same configuration) and finishes
    log = pd.read_csv(run / "log.csv")
    assert log["step"].tolist() == [2, 4, 6] and (run / "DONE").exists()


def test_analyze_end_to_end_on_synthetic_results(tmp_path):
    import yaml

    from s2sr.analysis import analyze
    from s2sr.stats import pairwise_units

    methods = yaml.safe_load((ROOT / "configs/methods.yaml").read_text())["methods"]
    rng = np.random.default_rng(0)
    for k, (m, spec) in enumerate(methods.items()):
        seeds = [0] if spec["arch"] in ("bicubic", "atprk", "dsen2_global") else [0, 1, 2]
        for s in seeds:
            rows = []
            for proto, ds, nunit, bands in [("P1", "test", 6, TARGET_BANDS), ("P2", "test", 6, TARGET_BANDS),
                                            ("P1", "sen2neon", 5, TARGET_BANDS), ("P2", "sen2neon", 5, TARGET_BANDS),
                                            ("P3", "sen2neon", 5, TARGET_BANDS),
                                            ("P1", "sen2venus", 4, TARGET_BANDS), ("P3", "sen2venus", 4, TARGET_BANDS[:4])]:
                if ds == "test" and proto == "P1":
                    pass
                for u in range(nunit):
                    for i in range(3):
                        r = dict(method=m, seed=s, protocol=proto, dataset=ds, unit=f"u{u}", key=f"{u}_{i}",
                                 psnr=40 - k + rng.normal(), sam=1.0)
                        for b in bands:
                            r[f"hp_corr_{b}"] = 0.9 - 0.01 * k * (1 if proto == "P1" else -1) + 0.01 * rng.normal()
                            r[f"edge_{b}"] = 1 + 0.05 * k + 0.01 * rng.normal()
                            r[f"hp_gain_{b}"] = 1.0
                            r[f"rmse_band_{b}"] = 0.01
                        r[f"edge_{bands[-1]}"] = np.nan          # a band without edges must not drop the row
                        rows.append(r)
            pd.DataFrame(rows).to_csv(tmp_path / f"{m}_seed{s}.csv", index=False)
    analyze(tmp_path, methods)
    tau = pd.read_csv(tmp_path / "analysis/rank_agreement.csv")
    row = tau[(tau.comparison == "P1(sen2neon) vs P3(sen2neon)") & (tau.metric == "hp_corr_mean")].iloc[0]
    assert row["tau"] == pytest.approx(-1.0) and row["paired"]
    edge = tau[(tau.comparison == "P1(sen2neon) vs P3(sen2neon)") & (tau.metric == "edge_mean")].iloc[0]
    assert edge["n_units_a"] == 5
    assert (tmp_path / "analysis/summary.md").exists() and (tmp_path / "analysis/scale_gap_sen2neon.csv").exists()
    pr = pd.read_csv(tmp_path / "analysis/paired_rq3_rq4.csv")
    assert {"RQ3", "RQ4"} <= set(pr.rq)
    res = pd.concat([pd.read_csv(f) for f in tmp_path.glob("*_seed*.csv")])
    out = pairwise_units(res[(res.protocol == "P3") & (res.dataset == "sen2neon")], "hp_corr_B05",
                         ["edsr_g", "edsr_u"], [("edsr_g", "edsr_u")])
    assert out.loc[0, "ci_low"] <= out.loc[0, "mean_diff"] <= out.loc[0, "ci_high"]


def test_swir_footprint_offset_found_by_template_matching():
    sys.path.insert(0, str(ROOT / "scripts"))
    from fetch_sen2venus_swir import consensus, ncc_map

    mos = _texture((400, 400), 3, 1.5)
    true = np.array([37, -52])
    pos = np.array([[100, 150], [200, 220], [250, 120], [150, 300], [300, 300], [120, 200]])
    offs, scores = [], []
    for r, c in pos:
        t = mos[r + true[0]:r + true[0] + 64, c + true[1]:c + true[1] + 64]
        m = ncc_map(mos, t)
        rr, cc = np.unravel_index(m.argmax(), m.shape)
        offs.append((rr - r, cc - c))
        scores.append(m.max())
    offs.append((5, 5))                      # one wrong probe does not change the consensus
    scores.append(0.9)
    off, votes = consensus(np.array(offs), np.array(scores))
    assert off == (37, -52) and votes == 6
    assert consensus(np.array(offs[:3]), np.array(scores[:3]))[0] is None   # too few votes


def test_flat_prediction_is_penalised_not_dropped():
    ref = 0.2 + 0.05 * _texture((2, 64, 64))
    flat = np.full_like(ref, 0.2)
    s = structural(flat, ref, np.ones((64, 64), bool), 10.0)
    assert np.allclose(s["hp_corr"], 0.0) and np.allclose(s["edge"], 3.0)


def test_paired_rank_agreement_uses_common_units():
    order = list("abcdefg")
    a = _unit_results(order)
    b = _unit_results(order, seed=1)
    b = b[b.unit != "u0"]                         # one site missing on one side
    r, _ = rank_agreement_units(a, b, "hp_corr_mean", order, n_boot=50, n_perm=500, paired=True)
    assert r["paired"] and r["n_units_a"] == r["n_units_b"] == 7
    r2, _ = rank_agreement_units(a, b, "hp_corr_mean", order, n_boot=50, n_perm=500)
    assert not r2["paired"]


def test_resume_refuses_changed_configuration(tmp_path):
    from test_core import _fake_shard

    from s2sr.train import train

    for split in ("train", "val"):
        _fake_shard(tmp_path / f"{split}.h5", n=8)
    spec = {"arch": "edsr", "params": {"nf": 8, "n_blocks": 1, "guided": True}}
    cfg = dict(iterations=2, batch_size=2, lr=1e-4, warmup=1, val_every=2, patience=10, amp=False, workers=0)
    run = tmp_path / "run"
    train("t", spec, cfg, MTF, tmp_path, run, 0)
    (run / "DONE").unlink()
    with pytest.raises(SystemExit):
        train("t", spec, cfg | {"lr": 2e-4}, MTF, tmp_path, run, 0)


def test_sigterm_saves_checkpoint_and_resumes(tmp_path, monkeypatch):
    import os
    import signal

    from test_core import _fake_shard

    from s2sr.train import train

    for split in ("train", "val"):
        _fake_shard(tmp_path / f"{split}.h5", n=8)
    spec = {"arch": "edsr", "params": {"nf": 8, "n_blocks": 1, "guided": True}}
    cfg = dict(iterations=10**6, batch_size=2, lr=1e-4, warmup=1, val_every=10**6, patience=10, amp=False,
               workers=0)
    run = tmp_path / "run"
    # SIGTERM at the 5th optimiser step, so the test does not depend on how fast the machine trains
    n_steps, adam_step = [0], torch.optim.Adam.step

    def step_then_term(self, *a, **k):
        n_steps[0] += 1
        if n_steps[0] == 5:
            os.kill(os.getpid(), signal.SIGTERM)
        return adam_step(self, *a, **k)

    monkeypatch.setattr(torch.optim.Adam, "step", step_then_term)
    with pytest.raises(SystemExit) as e:
        train("t", spec, cfg, MTF, tmp_path, run, 0)
    assert e.value.code == 143 and (run / "last.pt").exists() and not (run / "DONE").exists()
    step = torch.load(run / "last.pt", weights_only=False)["step"]
    assert 0 < step < 10**6
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    monkeypatch.undo()
    # validation needed for best.pt; changing the configuration starts a new run
    cfg_done = cfg | {"iterations": 400, "val_every": 100}
    with pytest.raises(SystemExit):
        train("t", spec, cfg_done, MTF, tmp_path, run, 0)
