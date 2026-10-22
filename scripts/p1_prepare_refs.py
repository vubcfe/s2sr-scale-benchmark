"""P1: prepare the real references (protocol P3) once, so that every method is scored on identical data.

For every SEN2NEON tile / SEN2VENuS patch:
  1. quality control  : SEN2NEON 20 m bands must be 2x2 replicas (<= 1 % non-constant blocks), >= 50 % valid
                        pixels, |dt| <= 7 days; SEN2VENuS pairs are same-day and fully valid.
  2. aggregation      : reference to 10 m with the MTF-matched Gaussian (ratio 4 from 2.5 m, ratio 2 from 5 m).
  3. registration     : phase correlation restricted to |f| <= 0.35 cycles/px (s2sr.registration.phase_corr)
                        between the observed Sentinel-2 20 m bands and the reference degraded to 20 m; the
                        translation is removed from the 10 m reference with a cubic spline (subpixel_shift).
                        SEN2NEON: one shift per tile (excluded if the peak is below 0.1).
                        SEN2VENuS: patches are only 64 px at 20 m, so the shift is the median over the patches
                        of one acquisition with peak >= 0.1; patches with a peak below 0.1 are excluded, and so
                        are patches whose fetched Sentinel-2 B8A (Sen2Cor, fetched with B11/B12) correlates
                        below 0.8 with the provided B8A (SWIR inputs misaligned or from another product).
  4. local residual   : SEN2NEON tiles whose 640 m block shifts after registration depart from one translation
                        by more than 0.5 px (RMS, 10 m pixels) are excluded; SEN2VENuS patches whose own shift
                        departs from the acquisition shift by more than 0.5 px (10 m) are excluded.
The reference radiometry is not changed. Output per dataset: <out>/<dataset>/<key>.npz (lr, guide, ref, valid)
and <out>/<dataset>/tiles.csv with one row per tile/patch, its unit (site), shifts, peaks and exclusion reason.

  python scripts/p1_prepare_refs.py sen2neon --root data/sen2neon --out data/p1_refs --workers 16
  python scripts/p1_prepare_refs.py sen2venus --root data/sen2venus --out data/p1_refs
"""
from __future__ import annotations

import argparse
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s2sr.bands import GUIDE_BANDS, RED_EDGE_BANDS, TARGET_BANDS  # noqa: E402
from s2sr.degrade import band_sigmas, degrade  # noqa: E402
from s2sr.registration import block_shifts, phase_corr, shift_field_residual, subpixel_shift  # noqa: E402

MAX_DT = 7.0
MIN_VALID = 0.5
MAX_NONCONST = 0.01
MIN_PEAK = 0.1
MAX_RESID_10M = 0.5
RF_MARGIN_10M = 8
MIN_SWIR_B8A_CORR = 0.8  # SEN2VENuS: Sen2Cor B8A (fetched with the SWIR) vs the provided MAJA B8A, per patch        # input nodata is filled; pixels within this distance of it are excluded
BLOCK_20M = 32           # 640 m blocks
SQUARE = np.ones((3, 3), bool)   # erosions remove every pixel within the Chebyshev distance (diagonals too)
UPSAMPLE = 100


def mtf() -> dict:
    return yaml.safe_load((ROOT / "configs" / "mtf.yaml").read_text())["mtf_nyquist"]


def _deg(x: np.ndarray, sig, ratio) -> np.ndarray:
    with torch.no_grad():
        return degrade(torch.from_numpy(np.ascontiguousarray(x, np.float32))[None], sig, ratio)[0].numpy()


def shift_valid(valid: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Validity after shifting by (dy, dx): shifted mask, eroded by the spline support."""
    v = ndimage.shift(valid.astype(np.float32), (dy, dx), order=1, mode="constant", cval=0.0) > 0.999
    return ndimage.binary_erosion(v, structure=SQUARE, iterations=int(math.ceil(max(abs(dy), abs(dx)))) + 2,
                                  border_value=0)


def fill_nodata(x: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Replace invalid pixels of a (C, H, W) stack by their nearest valid pixel."""
    if ok.all():
        return x
    idx = ndimage.distance_transform_edt(~ok, return_distances=False, return_indices=True)
    return x[:, idx[0], idx[1]]


def register(lr20: np.ndarray, ref10: np.ndarray, valid10: np.ndarray, sig20: list[float],
             shift20: tuple[float, float] | None = None):
    """Estimate (unless given) and remove the translation of ref10 relative to the Sentinel-2 20 m bands.

    Returns the registered reference, its validity, the shift (dy, dx) in 20 m pixels and the peak.
    """
    valid20 = valid10.reshape(valid10.shape[0] // 2, 2, valid10.shape[1] // 2, 2).all(axis=(1, 3))
    peak = np.nan
    if shift20 is None:
        # degrade a nodata-filled reference and use only 20 m pixels whose kernel support is valid
        filled = fill_nodata(ref10, valid10) if valid10.any() else ref10
        reach = math.ceil(3 * max(sig20) / 2) + 1
        v_est = ndimage.binary_erosion(valid20, structure=SQUARE, iterations=reach, border_value=1)
        dy, dx, peak = phase_corr(lr20, _deg(filled, sig20, 2), v_est if v_est.any() else valid20, UPSAMPLE)
    else:
        dy, dx = shift20
    if not np.isfinite(dy):
        return None, None, (np.nan, np.nan), 0.0
    ref_reg = np.stack([subpixel_shift(b, -2 * dy, -2 * dx) for b in ref10]).astype(np.float32)
    return ref_reg, shift_valid(valid10, -2 * dy, -2 * dx), (dy, dx), peak


# ---------------------------------------------------------------- SEN2NEON
NEON_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
G_IDX = [NEON_BANDS.index(b) for b in GUIDE_BANDS]
T_IDX = [NEON_BANDS.index(b) for b in TARGET_BANDS]


def _neon_tile(args):
    import rasterio

    from s2sr.data.sen2neon import HR_NODATA, LR_NODATA, SCALE

    root, out, row, m = args
    rec = dict(key=row["id"], unit=row["id"].split("_")[1], dt_days=abs(float(row["temporal_difference_days"])))
    try:
        if rec["dt_days"] > MAX_DT:
            return dict(rec, status="dt")
        with rasterio.open(root / row["lr"]) as d:
            lr_all = d.read()
        with rasterio.open(root / row["hr_2_5m_path"]) as d:
            hr_all = d.read()
        t = lr_all[T_IDX]
        blk = t.reshape(t.shape[0], t.shape[1] // 2, 2, t.shape[2] // 2, 2)
        ok_blk = np.all(blk != LR_NODATA, axis=(0, 2, 4))
        nonconst = (blk.max(axis=(2, 4)) != blk.min(axis=(2, 4))).any(axis=0)
        rec["nonconst_frac"] = float(nonconst[ok_blk].mean()) if ok_blk.any() else 1.0
        if rec["nonconst_frac"] > MAX_NONCONST:
            return dict(rec, status="resampled")
        sig_t4, sig_t2 = band_sigmas(m, TARGET_BANDS, 4), band_sigmas(m, TARGET_BANDS, 2)
        lr_ok = np.all(lr_all[G_IDX + T_IDX] != LR_NODATA, axis=0)     # only the bands the methods use
        hr_ok = np.all(hr_all[T_IDX] != HR_NODATA, axis=0)
        # reference validity at 10 m: every 2.5 m pixel within the aggregation kernel reach must be valid
        # (square erosion; outside the tile counts as invalid)
        h4, w4 = hr_ok.shape
        ok_c = hr_ok.reshape(h4 // 4, 4, w4 // 4, 4).all(axis=(1, 3))
        reach = math.ceil(3 * max(sig_t4) / 4) + 1
        valid10 = lr_ok & ndimage.binary_erosion(ok_c, structure=SQUARE, iterations=reach, border_value=0)
        rec["valid_frac"] = float(valid10.mean())
        if rec["valid_frac"] < MIN_VALID:
            return dict(rec, status="invalid")
        # nodata in the inputs would leak through the receptive field of the networks and into the ATPRK
        # regression: fill it and exclude its neighbourhood from the statistics
        guide = fill_nodata(lr_all[G_IDX].astype(np.float32) / SCALE, lr_ok)
        ok20 = lr_ok.reshape(lr_ok.shape[0] // 2, 2, lr_ok.shape[1] // 2, 2).all(axis=(1, 3))
        lr20 = fill_nodata(t[:, ::2, ::2].astype(np.float32) / SCALE, ok20)
        valid10 &= ndimage.binary_erosion(lr_ok, structure=SQUARE, iterations=RF_MARGIN_10M, border_value=1)
        ref10 = _deg(hr_all[T_IDX].astype(np.float32) / SCALE, sig_t4, 4)
        ref_reg, v_reg, (dy, dx), peak = register(lr20, ref10, valid10, sig_t2)
        rec.update(dy20=dy, dx20=dx, shift10=2 * math.hypot(dy, dx) if np.isfinite(dy) else np.nan, peak=peak)
        if not peak >= MIN_PEAK:
            return dict(rec, status="peak")
        h, w = v_reg.shape
        v20 = v_reg.reshape(h // 2, 2, w // 2, 2).all(axis=(1, 3))
        res = shift_field_residual(block_shifts(lr20, _deg(ref_reg, sig_t2, 2), v20, block=BLOCK_20M,
                                                upsample=UPSAMPLE), MIN_PEAK)
        rec.update(n_blocks=res["n_blocks"], resid10=2 * res["rms_translation"])
        if res["n_blocks"] >= 3 and rec["resid10"] > MAX_RESID_10M:
            return dict(rec, status="residual")
        rec["valid_frac_reg"] = float(v_reg.mean())
        if rec["valid_frac_reg"] < MIN_VALID:      # recheck after the no-data margin and the shift
            return dict(rec, status="invalid")
        np.savez_compressed(out / f"{rec['key']}.npz", lr=lr20, guide=guide, ref=ref_reg, valid=v_reg,
                            ref_unreg=ref10.astype(np.float32), valid_unreg=valid10)
        return dict(rec, status="ok")
    except Exception as e:  # noqa: BLE001  record and continue; reported in tiles.csv
        return dict(rec, status=f"error: {e}")


def prepare_sen2neon(root: Path, out: Path, workers: int):
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(root / "metadata.csv")
    m = mtf()
    jobs = [(root, out, r, m) for r in meta.to_dict("records")]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_neon_tile, jobs, chunksize=4))
    finish(pd.DataFrame(rows), out)


def finish(df: pd.DataFrame, out: Path, min_ok: int = 20) -> None:
    df.to_csv(out / "tiles.csv", index=False)
    print(df["status"].value_counts().to_string())
    ok = df[df["status"] == "ok"]
    print(f"ok: {len(ok)} tiles from {ok['unit'].nunique() if len(ok) else 0} sites")
    if len(ok) < min_ok:
        raise SystemExit(f"{out}: only {len(ok)} reference tiles passed QC (< {min_ok})")


# ---------------------------------------------------------------- SEN2VENuS
def prepare_sen2venus(root: Path, out: Path):
    from s2sr.data.sen2venus import G10, R05, S20, SW20, _load, check_band_order, list_acquisitions

    out.mkdir(parents=True, exist_ok=True)
    m = mtf()
    sig5 = band_sigmas(m, RED_EDGE_BANDS, 2)
    sig20 = band_sigmas(m, RED_EDGE_BANDS, 2)
    rows = []
    for p in list_acquisitions(root):
        site = p.name.split("_")[0]
        base = dict(unit=site, acquisition=p.name, dt_days=0.0)
        if not Path(f"{p}{SW20}").exists():
            rows.append(dict(base, key=p.name, status="no_swir"))
            continue
        check_band_order(p)
        g10, re20, sw20, ref05 = (_load(Path(f"{p}{s}")).numpy() for s in (G10, S20, SW20, R05))
        n = re20.shape[0]
        ref10 = np.stack([_deg(ref05[k], sig5, 2) for k in range(n)])
        valid10 = np.ones(ref10.shape[-2:], bool)
        valid20 = np.ones(re20.shape[-2:], bool)
        est = [phase_corr(re20[k], _deg(ref10[k], sig20, 2), valid20, UPSAMPLE) for k in range(n)]
        good = np.array([e[2] >= MIN_PEAK and np.isfinite(e[0]) for e in est])
        if not good.any():
            rows += [dict(base, key=f"{p.name}_{k:04d}", peak=est[k][2], status="peak") for k in range(n)]
            continue
        d = np.array([[e[0], e[1]] for e in est])
        acq = np.median(d[good], axis=0)
        chk_path = Path(f"{p}_swir_check.csv")
        swir_corr = (pd.read_csv(chk_path).set_index("patch")["b8a_corr"] if chk_path.exists()
                     else pd.Series(np.nan, index=range(n)))
        for k in range(n):
            key = f"{p.name}_{k:04d}"
            rec = dict(base, key=key, dy20=acq[0], dx20=acq[1], shift10=2 * float(np.hypot(*acq)),
                       peak=est[k][2], n_reliable=int(good.sum()))
            rec["resid10"] = 2 * float(np.hypot(*(d[k] - acq))) if good[k] else np.nan
            rec["swir_b8a_corr"] = float(swir_corr.get(k, np.nan))
            if not good[k]:                      # no reliable estimate of its own: cannot verify alignment
                rows.append(dict(rec, status="peak"))
                continue
            if rec["resid10"] > MAX_RESID_10M:
                rows.append(dict(rec, status="residual"))
                continue
            if not rec["swir_b8a_corr"] >= MIN_SWIR_B8A_CORR:
                rows.append(dict(rec, status="swir_qc"))
                continue
            ref_reg, v_reg, _, _ = register(re20[k], ref10[k], valid10, sig20, shift20=tuple(acq))
            if v_reg.mean() < MIN_VALID:
                rows.append(dict(rec, valid_frac_reg=float(v_reg.mean()), status="invalid"))
                continue
            lr = np.concatenate([re20[k], sw20[k]]).astype(np.float32)
            np.savez_compressed(out / f"{key}.npz", lr=lr, guide=g10[k].astype(np.float32), ref=ref_reg,
                                valid=v_reg, ref_unreg=ref10[k].astype(np.float32), valid_unreg=valid10)
            rows.append(dict(rec, valid_frac_reg=float(v_reg.mean()), status="ok"))
        print(f"{p.name}: {n} patches, shift20 = ({acq[0]:+.3f}, {acq[1]:+.3f}), reliable {good.sum()}")
    finish(pd.DataFrame(rows), out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["sen2neon", "sen2venus"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="data/p1_refs")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out) / a.dataset
    if a.dataset == "sen2neon":
        prepare_sen2neon(Path(a.root), out, a.workers)
    else:
        prepare_sen2venus(Path(a.root), out)


if __name__ == "__main__":
    main()
