"""Fetch Sentinel-2 B11 and B12 for every SEN2VENuS patch, located on the provided B8A.

SEN2VENuS provides B5, B6, B7, B8A at 20 m but not the SWIR bands our models take as input. The patch
footprints in <prefix>_patches.gpkg are not always where the provided Sentinel-2 patches are: on some sites
(e.g. LERIDA-1) every footprint is displaced by the same large offset (about 10 km), and elsewhere by a fraction
of a pixel (checked on 19/10/2026 by template matching). The SWIR bands are therefore located on the image data,
not on the footprints. For each acquisition <SITE>_<TILE>_<DATE>:
  1. every Sentinel-2 L2A item of that date on Planetary Computer is read (B8A) on the UTM grid of the gpkg over
     the footprints plus a MARGIN_PX margin (nearest neighbour on the same MGRS tile, bilinear otherwise);
  2. the PROBES most textured provided B8A patches are located on each mosaic by normalised cross-correlation;
     the integer offset of the footprints is the most frequent probe offset (+-1 px), accepted when at least
     MIN_VOTES probes with NCC >= MIN_NCC agree; the item with the most votes is used first, the others fill
     patches it does not cover;
  3. the sub-pixel residual is the median phase-correlation shift between provided and fetched B8A over the
     patches (peak >= 0.3) and is removed from the fetched bands with a cubic spline;
  4. B11 and B12 are cut at the corrected positions. Written next to the SEN2VENuS files:
       <prefix>_20m_b11b12.pt   (N, 2, 64, 64) int16, reflectance * 10000 (zeros where not covered)
       <prefix>_swir_check.csv  per patch: item, offsets, B8A bias / RMSE / correlation after alignment
     Acquisitions without a consensus offset are skipped (no files).
SEN2VENuS is MAJA L2A, Planetary Computer is Sen2Cor L2A; the B8A correlation after alignment is the QC used by
scripts/p1_prepare_refs.py.

  .venv/bin/python scripts/fetch_sen2venus_swir.py --root data/sen2venus [--sites LERIDA-1] [--overwrite]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import fftconvolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s2sr.data.sen2venus import S20, SW20, list_acquisitions  # noqa: E402
from s2sr.data.stac_download import STAC_URL, to_reflectance  # noqa: E402
from s2sr.registration import phase_corr, subpixel_shift  # noqa: E402

PATCH_20M = 64
MARGIN_PX = 640          # search radius for the footprint offset (12.8 km)
PROBES = 30
MIN_NCC = 0.5
MIN_VOTES = 5
PAD = 6                  # extra pixels cut around a patch for the sub-pixel shift


def ncc_map(big: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Normalised cross-correlation of template t at every valid position of big."""
    t = t - t.mean()
    ones = np.ones_like(t)
    s = fftconvolve(big, ones, mode="valid")
    s2 = fftconvolve(big ** 2, ones, mode="valid")
    num = fftconvolve(big, t[::-1, ::-1], mode="valid")
    den = np.sqrt(np.maximum(s2 - s ** 2 / t.size, 1e-12)) * np.sqrt((t ** 2).sum())
    return num / den


def consensus(offsets: np.ndarray, scores: np.ndarray) -> tuple[tuple[int, int] | None, int]:
    """Most frequent integer offset among reliable probes (neighbours within +-1 px vote together)."""
    good = offsets[scores >= MIN_NCC]
    if len(good) == 0:
        return None, 0
    votes = [int((np.abs(good - o).max(axis=1) <= 1).sum()) for o in good]
    best = int(np.argmax(votes))
    if votes[best] < MIN_VOTES:
        return None, votes[best]
    close = good[np.abs(good - good[best]).max(axis=1) <= 1]
    return (int(np.round(np.median(close[:, 0]))), int(np.round(np.median(close[:, 1])))), votes[best]


def main():
    import geopandas as gpd
    import odc.stac
    import planetary_computer
    import pystac_client
    from odc.geo.geobox import GeoBox
    from rasterio.transform import from_origin

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--sites", default="", help="comma-separated site names (default: all)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    sites = set(filter(None, a.sites.split(",")))
    cat = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)

    for prefix in list_acquisitions(a.root):
        parts = prefix.name.split("_")   # <SITE>[_extra]_<TILE>_<DATE>, e.g. ANJI_C_V_50RQV_2018-04-09
        site, tile, day = parts[0], parts[-2], parts[-1]
        if sites and site not in sites:
            continue
        out_pt = Path(f"{prefix}{SW20}")
        if out_pt.exists() and not a.overwrite:
            continue
        gpkg = Path(f"{prefix}_patches.gpkg")
        if not gpkg.exists():   # a few acquisitions of the release have no footprint file
            print(f"{prefix.name}: no patch footprints; skipped")
            continue
        provided = torch.load(Path(f"{prefix}{S20}"), weights_only=True).float().numpy() / 10000
        gdf = gpd.read_file(gpkg)
        n = len(gdf)
        if n != provided.shape[0]:
            raise ValueError(f"{prefix.name}: {n} footprints but {provided.shape[0]} patches")
        found = list(cat.search(collections=["sentinel-2-l2a"], bbox=list(gdf.to_crs(4326).total_bounds),
                                datetime=day).items())
        if not found:
            print(f"{prefix.name}: no Sentinel-2 L2A item on {day}; skipped")
            continue
        x0, y0, x1, y1 = gdf.total_bounds
        H = int(round((y1 - y0) / 20)) + 2 * MARGIN_PX
        W = int(round((x1 - x0) / 20)) + 2 * MARGIN_PX
        gb = GeoBox((H, W), from_origin(x0 - 20 * MARGIN_PX, y1 + 20 * MARGIN_PX, 20, 20), gdf.crs)
        # position (row, col) of every footprint's upper-left corner in the mosaic
        pos = np.array([[int(round((y1 - g.bounds[3]) / 20)) + MARGIN_PX,
                         int(round((g.bounds[0] - x0) / 20)) + MARGIN_PX] for g in gdf.geometry])
        b8 = provided[:, 3]
        probes = np.argsort(-b8.reshape(n, -1).std(1))[:PROBES]
        cands = []
        for item in found:
            item = planetary_computer.sign(item)
            same = f"_T{tile}_" in item.id
            base = item.properties.get("s2:processing_baseline", "00.00")
            ds = odc.stac.load([item], bands=["B8A"], geobox=gb, resampling="nearest" if same else "bilinear")
            mos = np.nan_to_num(to_reflectance(ds["B8A"].isel(time=0).values, base).astype(np.float64))
            offs, scores = [], []
            for k in probes:
                m = ncc_map(mos, b8[k])
                r, c = np.unravel_index(np.nanargmax(m), m.shape)
                offs.append((r - pos[k, 0], c - pos[k, 1]))
                scores.append(float(np.nanmax(m)))
            off, votes = consensus(np.array(offs), np.array(scores))
            cands.append(dict(item=item, same=same, base=base, off=off, votes=votes,
                              ncc=float(np.median(scores))))
        cands = sorted([c for c in cands if c["off"] is not None], key=lambda c: (-c["votes"], -c["ncc"]))
        if not cands:
            print(f"{prefix.name}: no consensus footprint offset ({len(found)} items); skipped")
            continue
        # cut every patch (with PAD) from the best item that covers it
        cut = np.zeros((n, 3, PATCH_20M + 2 * PAD, PATCH_20M + 2 * PAD))
        src = np.full(n, -1)
        for ci, c in enumerate(cands):
            todo = np.where(src < 0)[0]
            if not len(todo):
                break
            ds = odc.stac.load([c["item"]], bands=["B8A", "B11", "B12"], geobox=gb,
                               resampling="nearest" if c["same"] else "bilinear")
            mos = np.stack([np.nan_to_num(to_reflectance(ds[b].isel(time=0).values, c["base"]).astype(np.float64))
                            for b in ("B8A", "B11", "B12")])
            for k in todo:
                r, cc = pos[k] + np.array(c["off"]) - PAD
                if r < 0 or cc < 0 or r + PATCH_20M + 2 * PAD > H or cc + PATCH_20M + 2 * PAD > W:
                    continue
                win = mos[:, r:r + PATCH_20M + 2 * PAD, cc:cc + PATCH_20M + 2 * PAD]
                if (win[0] == 0).mean() > 0.01:
                    continue
                cut[k], src[k] = win, ci
        # sub-pixel residual: median over well-textured patches
        inner = (slice(PAD, PAD + PATCH_20M), slice(PAD, PAD + PATCH_20M))
        est = [phase_corr(b8[k], cut[k, 0][inner]) for k in np.where(src >= 0)[0]]
        good = [e for e in est if e[2] >= 0.3 and np.isfinite(e[0])]
        dy, dx = (float(np.median([e[0] for e in good])), float(np.median([e[1] for e in good]))) if good \
            else (0.0, 0.0)
        out = np.zeros((n, 2, PATCH_20M, PATCH_20M), np.int16)
        checks = []
        for k in range(n):
            rec = dict(patch=k, item=cands[src[k]]["item"].id if src[k] >= 0 else "",
                       tile_match=bool(cands[src[k]]["same"]) if src[k] >= 0 else False,
                       off_row=cands[src[k]]["off"][0] if src[k] >= 0 else np.nan,
                       off_col=cands[src[k]]["off"][1] if src[k] >= 0 else np.nan,
                       votes=cands[src[k]]["votes"] if src[k] >= 0 else 0, sub_dy=dy, sub_dx=dx)
            if src[k] < 0:
                checks.append(dict(rec, b8a_bias=np.nan, b8a_rmse=np.nan, b8a_corr=np.nan))
                continue
            al = np.stack([subpixel_shift(cut[k, b], -dy, -dx)[inner] for b in range(3)])
            out[k] = np.round(al[1:] * 10000).clip(-32768, 32767).astype(np.int16)
            ref = b8[k]
            checks.append(dict(rec, b8a_bias=float((al[0] - ref).mean()),
                               b8a_rmse=float(np.sqrt(((al[0] - ref) ** 2).mean())),
                               b8a_corr=float(np.corrcoef(al[0].ravel(), ref.ravel())[0, 1])))
        torch.save(torch.from_numpy(out), out_pt)
        c = pd.DataFrame(checks)
        c.to_csv(Path(f"{prefix}_swir_check.csv"), index=False)
        best = cands[0]
        print(f"{prefix.name}: {n} patches, {int((src >= 0).sum())} covered | offset {best['off']} "
              f"({best['votes']}/{len(probes)} votes) + sub-pixel ({dy:+.2f}, {dx:+.2f}) | "
              f"B8A corr median {c.b8a_corr.median():.3f}")


if __name__ == "__main__":
    main()
