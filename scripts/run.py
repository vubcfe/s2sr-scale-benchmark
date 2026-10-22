"""Command-line entry point for the P1 benchmark pipeline.

  python scripts/run.py download            # Sentinel-2 L2A scenes per region (Planetary Computer)
  python scripts/run.py blocks              # block grids + split assignment
  python scripts/run.py patches             # HDF5 shards per split
  python scripts/run.py count-params        # check parameter matching of learned methods
  python scripts/run.py train --method edsr_g --seed 0
  python scripts/run.py eval  --method edsr_g --seed 0 [--refs data/p1_refs]
  python scripts/run.py analyze             # tables for RQ1-RQ4 from results/p1/*.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s2sr.models import build_model, count_params  # noqa: E402

log = logging.getLogger("s2sr")


def cfg(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())


def mtf_table() -> dict:
    m = cfg("mtf")
    if m.get("placeholder"):
        log.warning("configs/mtf.yaml still holds PLACEHOLDER MTF values; results are not final")
    return m["mtf_nyquist"]


# ---------------------------------------------------------------- data
def cmd_download(a):
    from s2sr.data.stac_download import download_region

    rc, dc = cfg("regions"), cfg("data")
    out = ROOT / dc["raw_dir"]
    records = []
    for name, region in rc["regions"].items():
        if region.get("source", "stac") != "stac":
            log.info("%s: source=%s, not downloaded", name, region["source"])
            continue
        records += download_region(name, region, rc, out)
    pd.DataFrame(records).to_csv(out / "scenes.csv", index=False)


def cmd_blocks(a):
    import geopandas as gpd
    from pyproj import Transformer
    from shapely.geometry import box

    from s2sr.data.blocks import make_blocks

    rc, dc = cfg("regions"), cfg("data")
    out = ROOT / dc["blocks_dir"]
    out.mkdir(parents=True, exist_ok=True)
    bc = dc["blocks"]
    for name, region in rc["regions"].items():
        if region.get("source", "stac") != "stac":
            continue  # regions with another source are not tiled here
        if region.get("bbox") is None:
            raise ValueError(f"region {name}: bbox not set")
        t = Transformer.from_crs("EPSG:4326", f"EPSG:{region['utm_epsg']}", always_xy=True)
        x0, y0 = t.transform(region["bbox"][0], region["bbox"][1])
        x1, y1 = t.transform(region["bbox"][2], region["bbox"][3])
        blocks = make_blocks((x0, y0, x1, y1), bc["size_m"], bc["buffer_m"], bc["fractions"],
                             bc["seed"], heldout=region["role"] == "heldout")
        gdf = gpd.GeoDataFrame(
            [dict(block=b.bid, row=b.row, col=b.col, split=b.split) for b in blocks],
            geometry=[box(b.x0, b.y0, b.x1, b.y1) for b in blocks], crs=f"EPSG:{region['utm_epsg']}")
        gdf.to_file(out / f"{name}.gpkg", driver="GPKG")
        log.info("%s: %s", name, gdf["split"].value_counts().to_dict())


def _load_blocks(path):
    import geopandas as gpd

    from s2sr.data.blocks import Block

    gdf = gpd.read_file(path)
    return [Block(int(r.row), int(r.col), r.split, *r.geometry.bounds) for r in gdf.itertuples()]


def cmd_patches(a):
    from s2sr.data.patches import ShardWriter, cut_scene

    dc = cfg("data")
    raw, out = ROOT / dc["raw_dir"], ROOT / dc["patch_dir"]
    out.mkdir(parents=True, exist_ok=True)
    pc = dc["patches"]
    scenes = pd.read_csv(raw / "scenes.csv")
    writers = {s: ShardWriter(out / f"{s}.h5", pc["size_20m"]) for s in ("train", "val", "test", "test_ood")}
    try:
        for rec in scenes.to_dict("records"):
            blocks = _load_blocks(ROOT / dc["blocks_dir"] / f"{rec['region']}.gpkg")
            n = cut_scene(raw / rec["region"] / rec["scene"], rec, blocks, writers,
                          pc["size_20m"], pc["stride_20m"], pc["min_valid"])
            log.info("%s: %d patches", rec["scene"], n)
    finally:
        for w in writers.values():
            w.close()


# ---------------------------------------------------------------- models
def cmd_count_params(a):
    mtf = mtf_table()
    n = {}
    for name, spec in cfg("methods")["methods"].items():
        if spec["arch"] in ("bicubic", "atprk", "dsen2_global"):
            continue   # not trained here
        n[name] = count_params(build_model(spec, mtf))
        print(f"{name:14s} {n[name] / 1e6:6.2f} M")
    ref = n["dsen2"]
    off = {k: v for k, v in n.items() if v and abs(v - ref) / ref > 0.10}
    if off:
        raise SystemExit(f"parameter counts outside dsen2 +-10 %: {off}")


def cmd_train(a):
    from s2sr.train import train

    spec = cfg("methods")["methods"][a.method]
    tc = cfg("train")
    train(a.method, spec, tc, mtf_table(), ROOT / cfg("data")["patch_dir"],
          ROOT / "runs" / "p1" / a.method / f"seed{a.seed}", a.seed)


def _load_trained(method, seed, mtf, device):
    spec = cfg("methods")["methods"][method]
    model = build_model(spec, mtf).to(device)
    if getattr(model, "trainable", False):
        ckpt = ROOT / "runs" / "p1" / method / f"seed{seed}" / "best.pt"
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return model


def cmd_eval(a):
    from s2sr.data.dataset import NativeDataset, WaldDataset
    from s2sr.evaluate import RefDataset, eval_p1, eval_p2, eval_refs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtf = mtf_table()
    pdir = ROOT / cfg("data")["patch_dir"]
    model = _load_trained(a.method, a.seed, mtf, device)
    out = ROOT / "results" / "p1"
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for split in ("test", "test_ood"):
        frames.append(eval_p1(model, WaldDataset(pdir / f"{split}.h5", mtf), a.method, a.seed, split, device))
        frames.append(eval_p2(model, NativeDataset(pdir / f"{split}.h5"), a.method, a.seed, split, device, mtf))
        log.info("%s seed %d: %s done", a.method, a.seed, split)
    for name in ("sen2neon", "sen2venus"):
        d = ROOT / a.refs / name
        if (d / "tiles.csv").exists():
            ds = RefDataset(d)
            if len(ds) == 0:
                raise SystemExit(f"{d}: no reference tiles with status ok")
            frames.append(eval_refs(model, ds, a.method, a.seed, device, mtf))
            log.info("%s seed %d: %s done", a.method, a.seed, name)
        else:
            log.warning("no prepared references in %s", d)
    tmp = out / f".{a.method}_seed{a.seed}.csv.tmp"
    pd.concat(frames).to_csv(tmp, index=False)
    tmp.replace(out / f"{a.method}_seed{a.seed}.csv")


def expected_results() -> list[str]:
    seeds = cfg("train")["seeds"]
    out = []
    for name, spec in cfg("methods")["methods"].items():
        trainable = spec["arch"] not in ("bicubic", "atprk", "dsen2_global")
        out += [f"{name}_seed{s}.csv" for s in (seeds if trainable else [0])]
    return out


def cmd_analyze(a):
    from s2sr.analysis import analyze

    res = ROOT / "results" / "p1"
    res.mkdir(parents=True, exist_ok=True)
    missing = [f for f in expected_results() if not (res / f).exists()]
    if missing and not a.allow_incomplete:
        (res / "MISSING.txt").write_text("\n".join(missing) + "\n")
        raise SystemExit(f"{len(missing)} result files missing (see {res / 'MISSING.txt'}); "
                         "rerun the failed jobs or pass --allow-incomplete")
    analyze(res, cfg("methods")["methods"], allow_incomplete=a.allow_incomplete)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for n in ("download", "blocks", "patches", "count-params"):
        sub.add_parser(n)
    an = sub.add_parser("analyze")
    an.add_argument("--allow-incomplete", action="store_true")
    t = sub.add_parser("train")
    t.add_argument("--method", required=True)
    t.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("eval")
    e.add_argument("--method", required=True)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--refs", default="data/p1_refs", help="directory written by scripts/p1_prepare_refs.py")
    a = p.parse_args()
    {"download": cmd_download, "blocks": cmd_blocks, "patches": cmd_patches,
     "count-params": cmd_count_params, "train": cmd_train, "eval": cmd_eval,
     "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
