"""Evaluation protocols. Each function returns a long-format DataFrame with one row per sample.

P1  reduced scale : restore 20 m from 40 m (Wald pairs), compare with the original 20 m.
P2  consistency   : super-resolve native 20 m to 10 m, degrade back with the MTF filter, compare with the
                    original 20 m. Necessary, not sufficient.
P3  real reference: compare the 10 m output with a registered real reference aggregated to 10 m
                    (SEN2VENuS: B5-B8A; SEN2NEON: all six bands). The reference radiometry is not changed.

On the real-reference tiles P1 and P2 are computed as well (dataset "<name>"), so that reduced-scale and
real-resolution scores can be compared on the same tiles (site-matched; removes the domain confound).
Every row carries `unit` (block or site), the resampling unit of the statistics.

Primary metrics (P1 paper, Sec. 3.5): hp_corr_<band> and edge_<band> (s2sr.structural); everything else is
secondary.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .bands import GUIDE_BANDS, RED_EDGE_BANDS, TARGET_BANDS, target_index
from .degrade import band_sigmas, degrade
from .metrics import crop, full_reference, psnr, rmse_per_band, sam
from .structural import highpass, nd, sigma_px, stat_mask, structural

BORDER = 4


def _put(row: dict, name: str, values, bands) -> None:
    for b, v in zip(bands, np.atleast_1d(values)):
        row[f"{name}_{b}"] = float(v)


def _struct_row(row: dict, pred: np.ndarray, ref: np.ndarray, valid: np.ndarray, grid_m: float, bands) -> None:
    s = structural(pred, ref, valid, grid_m, border=BORDER)
    for k in ("hp_corr", "edge", "hp_gain"):
        _put(row, k, s[k], bands)


def _index_hp_corr(pred: np.ndarray, ref: np.ndarray, valid: np.ndarray, grid_m: float, a: str, b: str) -> float:
    """Correlation of the high-pass components of a normalised-difference index (gain-sensitive only through
    the ratio of the two band gains)."""
    s = sigma_px(grid_m)
    m = stat_mask(valid, s, BORDER)
    if m.sum() < 64:
        return float("nan")
    ia, ib = target_index(a), target_index(b)
    hp = highpass(nd(pred, ia, ib)[None], valid, s)[0][m]
    hr = highpass(nd(ref, ia, ib)[None], valid, s)[0][m]
    if not np.isfinite(hp).all():
        return 0.0
    hp, hr = hp - hp.mean(), hr - hr.mean()
    sp, sr = np.sqrt((hp ** 2).sum()), np.sqrt((hr ** 2).sum())
    if sr <= 1e-10:
        return float("nan")
    return float((hp * hr).sum() / (sp * sr)) if sp > 1e-10 else 0.0


def coarse_valid(valid: torch.Tensor) -> torch.Tensor:
    """(N, H, W) -> (N, H/2, W/2): a coarse pixel is valid if its 2 x 2 fine pixels are."""
    n, h, w = valid.shape
    return valid.reshape(n, h // 2, 2, w // 2, 2).all(dim=4).all(dim=2)


def run_model(model, lr, guide, valid_lr=None):
    """Forward pass; methods that can exclude invalid pixels (ATPRK) receive the validity on the lr grid."""
    from .models.atprk import ATPRK

    if isinstance(model, ATPRK) and valid_lr is not None:
        return model(lr, guide, valid_lr)
    return model(lr, guide)


def _base(method, seed, protocol, dataset, unit, key) -> dict:
    return dict(method=method, seed=seed, protocol=protocol, dataset=dataset, unit=str(unit), key=str(key))


# ---------------------------------------------------------------- Vietnamese test blocks
@torch.no_grad()
def eval_p1(model, dataset, method, seed, name, device, batch_size=16) -> pd.DataFrame:
    model.eval()
    rows = []
    for b in DataLoader(dataset, batch_size=batch_size):
        sr = run_model(model, b["lr"].to(device), b["guide"].to(device), coarse_valid(b["valid"])).float().cpu()
        m = full_reference(sr, b["target"], b["veg"], BORDER, valid=b["valid"])
        for k in range(sr.shape[0]):
            r = _base(method, seed, "P1", name, b["unit"][k], int(b["idx"][k]))
            for n, v in m.items():
                if v.ndim == 2:
                    _put(r, n, v[k].numpy(), TARGET_BANDS)
                else:
                    r[n] = float(v[k])
            _struct_row(r, sr[k].numpy(), b["target"][k].numpy(), b["valid"][k].numpy(), 20.0, TARGET_BANDS)
            rows.append(r)
    return pd.DataFrame(rows)


@torch.no_grad()
def eval_p2(model, dataset, method, seed, name, device, mtf, batch_size=8) -> pd.DataFrame:
    model.eval()
    sig = band_sigmas(mtf, TARGET_BANDS)
    rows = []
    for b in DataLoader(dataset, batch_size=batch_size):
        sr = run_model(model, b["lr"].to(device), b["guide"].to(device), b["valid"]).float()
        back = degrade(sr, sig).cpu()
        lr = b["lr"]
        bc, lc, vc = crop(back, BORDER), crop(lr, BORDER), crop(b["valid"], BORDER)
        ps, rm, sa = psnr(bc, lc, mask=vc), rmse_per_band(bc, lc, vc), sam(bc, lc, vc)
        for k in range(sr.shape[0]):
            r = _base(method, seed, "P2", name, b["unit"][k], int(b["idx"][k]))
            r.update(psnr=float(ps[k]), sam=float(sa[k]))
            _put(r, "rmse_band", rm[k].numpy(), TARGET_BANDS)
            _struct_row(r, back[k].numpy(), lr[k].numpy(), b["valid"][k].numpy(), 20.0, TARGET_BANDS)
            rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- real-reference tiles
class RefDataset(Dataset):
    """Tiles written by scripts/p1_prepare_refs.py (status 'ok' only)."""

    def __init__(self, root):
        self.root = Path(root)
        t = pd.read_csv(self.root / "tiles.csv")
        self.tiles = t[t["status"] == "ok"].reset_index(drop=True)
        self.name = self.root.name

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i):
        row = self.tiles.iloc[i]
        z = np.load(self.root / f"{row['key']}.npz")
        f32 = lambda k: torch.from_numpy(np.asarray(z[k], np.float32))   # noqa: E731
        out = dict(lr=f32("lr"), guide=f32("guide"), ref=f32("ref"), valid=torch.from_numpy(z["valid"].astype(bool)),
                   unit=str(row["unit"]), key=str(row["key"]), ref_unreg=None, valid_unreg=None)
        if "ref_unreg" in z:
            out.update(ref_unreg=f32("ref_unreg"), valid_unreg=torch.from_numpy(z["valid_unreg"].astype(bool)))
        for k in ("lr", "guide", "ref"):     # reflectance in [0, 1] (small negative values occur in L2A)
            v = out[k][..., out["valid"]] if k == "ref" else out[k]
            if not torch.isfinite(v).all() or v.min() < -0.5 or v.max() > 2.0:
                raise ValueError(f"{row['key']}: {k} outside the reflectance range [{float(v.min())}, {float(v.max())}]")
        return out


@torch.no_grad()
def eval_refs(model, dataset: RefDataset, method, seed, device, mtf) -> pd.DataFrame:
    """P3 against the registered reference, plus P2 and site-matched P1 on the same tiles."""
    model.eval()
    sig_t, sig_g = band_sigmas(mtf, TARGET_BANDS), band_sigmas(mtf, GUIDE_BANDS)
    ref_bands = TARGET_BANDS if dataset.name == "sen2neon" else RED_EDGE_BANDS
    nb = len(ref_bands)
    rows = []
    for i in range(len(dataset)):
        b = dataset[i]
        lr, guide = b["lr"][None].to(device), b["guide"][None].to(device)
        valid = b["valid"].numpy()
        valid20 = valid.reshape(valid.shape[0] // 2, 2, valid.shape[1] // 2, 2).all(axis=(1, 3))
        v20t = torch.from_numpy(valid20)[None]
        sr = run_model(model, lr, guide, v20t).float()
        # P3
        sr_np = sr[0].cpu().numpy()
        ref = b["ref"].numpy()
        r = _base(method, seed, "P3", dataset.name, b["unit"], b["key"])
        _struct_row(r, sr_np[:nb], ref, valid, 10.0, ref_bands)
        _amplitude_row(r, sr[0, :nb].cpu(), b["ref"], b["valid"], ref_bands)
        r["ndre_hp_corr"] = _index_hp_corr(sr_np, _pad_bands(ref, nb), valid, 10.0, "B8A", "B05")
        if nb == len(TARGET_BANDS):
            r["ndmi_hp_corr"] = _index_hp_corr(sr_np, ref, valid, 10.0, "B8A", "B11")
        rows.append(r)
        if b.get("ref_unreg") is not None:     # sensitivity: reference before registration
            r = _base(method, seed, "P3u", dataset.name, b["unit"], b["key"])
            vu = b["valid_unreg"].numpy()
            _struct_row(r, sr_np[:nb], b["ref_unreg"].numpy(), vu, 10.0, ref_bands)
            _amplitude_row(r, sr[0, :nb].cpu(), b["ref_unreg"], b["valid_unreg"], ref_bands)
            r["ndre_hp_corr"] = _index_hp_corr(sr_np, _pad_bands(b["ref_unreg"].numpy(), nb), vu, 10.0, "B8A", "B05")
            if nb == len(TARGET_BANDS):
                r["ndmi_hp_corr"] = _index_hp_corr(sr_np, b["ref_unreg"].numpy(), vu, 10.0, "B8A", "B11")
            rows.append(r)
        # P2 on the same tile
        back = degrade(sr, sig_t)[0].cpu()
        r = _base(method, seed, "P2", dataset.name, b["unit"], b["key"])
        _struct_row(r, back.numpy(), b["lr"].numpy(), valid20, 20.0, TARGET_BANDS)
        _amplitude_row(r, back, b["lr"], v20t[0])
        rows.append(r)
        # P1 on the same tile: 40 m -> 20 m against the observed 20 m bands
        sr20 = run_model(model, degrade(lr, sig_t), degrade(guide, sig_g), coarse_valid(v20t)).float()[0].cpu()
        r = _base(method, seed, "P1", dataset.name, b["unit"], b["key"])
        _struct_row(r, sr20.numpy(), b["lr"].numpy(), valid20, 20.0, TARGET_BANDS)
        _amplitude_row(r, sr20, b["lr"], v20t[0])
        rows.append(r)
    return pd.DataFrame(rows)


def _amplitude_row(row: dict, pred: torch.Tensor, ref: torch.Tensor, valid: torch.Tensor, bands=TARGET_BANDS):
    """Secondary amplitude-sensitive metrics of one sample: PSNR, SAM, per-band RMSE over valid pixels."""
    p, q, v = crop(pred[None], BORDER), crop(ref[None], BORDER), crop(valid[None].bool(), BORDER)
    row["psnr"] = float(psnr(p, q, mask=v)[0])
    row["sam"] = float(sam(p, q, v)[0])
    _put(row, "rmse_band", rmse_per_band(p, q, v)[0].numpy(), bands)


def _pad_bands(ref: np.ndarray, nb: int) -> np.ndarray:
    """Red-edge-only references padded to six bands (SWIR channels unused) so that band indices match."""
    if nb == len(TARGET_BANDS):
        return ref
    out = np.zeros((len(TARGET_BANDS),) + ref.shape[1:], ref.dtype)
    out[:nb] = ref
    return out
