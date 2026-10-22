"""Analyses for RQ1-RQ4 of the P1 paper from results/p1/<method>_seed<k>.csv (written by `run.py eval`).

Datasets: "test" and "test_ood" (Vietnamese blocks; P1, P2) and "sen2neon" / "sen2venus" (real-reference
tiles; P1 and P2 on the same tiles, P3 against the registered reference). Primary metrics are HP-corr and edge
error, averaged over the bands the reference provides (all six for SEN2NEON, B5-B8A for SEN2VENuS; the same
band set is used for P1 and P2 of that dataset so that the comparison is matched in bands as well as tiles).

Outputs in results/p1/analysis: CSV tables and summary.md.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .bands import RED_EDGE_BANDS, SWIR_BANDS, TARGET_BANDS
from .stats import (friedman_units, pairwise_units, rank_agreement_units, scale_gap_units, unit_matrix)

REF_BANDS = {"sen2neon": TARGET_BANDS, "sen2venus": RED_EDGE_BANDS}
PRIMARY = [("hp_corr", True), ("edge", False)]
SECONDARY = [("rmse_band", False), ("hp_gain", None)]
RQ3_PAIRS = [("edsr_g", "edsr_u"), ("rcan_g", "rcan_u"), ("edsr_g", "edsr_g_scr"), ("edsr_g_scr", "edsr_u")]
RQ4_PAIRS = [("dsen2", "dsen2_global")]


def add_means(df: pd.DataFrame) -> pd.DataFrame:
    """Band-averaged columns <metric>_mean over the band set of each dataset (red-edge / SWIR subsets too)."""
    df = df.reset_index(drop=True)
    for ds, d in df.groupby("dataset"):
        bands = REF_BANDS.get(ds, TARGET_BANDS)
        for metric in ("hp_corr", "edge", "hp_gain", "rmse_band"):
            for tag, bs in (("mean", bands), ("re", [b for b in bands if b in RED_EDGE_BANDS]),
                            ("swir", [b for b in bands if b in SWIR_BANDS])):
                cols = [f"{metric}_{b}" for b in bs if f"{metric}_{b}" in d]
                if cols:
                    # mean over the bands with a value (edges can be missing in a flat band); a sample needs at
                    # least half of the bands, otherwise it is NaN
                    n_ok = d[cols].notna().sum(axis=1)
                    df.loc[d.index, f"{metric}_{tag}"] = d[cols].mean(axis=1).where(n_ok * 2 >= len(cols))
                    df.loc[d.index, f"{metric}_{tag}_nbands"] = n_ok
    return df


def _sel(df, protocol, dataset):
    return df[(df.protocol == protocol) & (df.dataset == dataset)]


def method_table(d: pd.DataFrame, cols: list[str], methods: list[str]) -> pd.DataFrame:
    """Mean over units (seed-averaged unit means) and seed sd of the method scores."""
    out = {}
    for c in cols:
        if c not in d or d[c].isna().all():
            continue
        g = d.dropna(subset=[c]).groupby(["method", "seed", "unit"])[c].mean().groupby(["method", "seed"]).mean()
        out[c] = g.groupby("method").mean()
        out[c + "_seed_sd"] = g.groupby("method").std()
    t = pd.DataFrame(out)
    return t.loc[[m for m in methods if m in t.index]]


def analyze(res_dir: Path, methods_cfg: dict, allow_incomplete: bool = False) -> None:
    out = res_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(res_dir.glob("*_seed*.csv"))
    res = add_means(pd.concat([pd.read_csv(p, dtype={"unit": str, "key": str}) for p in files], ignore_index=True))
    for proto, ds in [("P1", "test"), ("P2", "test"), ("P3", "sen2neon"), ("P1", "sen2neon")]:
        if not len(_sel(res, proto, ds)):
            raise SystemExit(f"no {proto} rows for {ds}: the evaluation is incomplete")
    if not len(_sel(res, "P3", "sen2venus")):
        if not allow_incomplete:
            raise SystemExit("no SEN2VENuS P3 rows (pass --allow-incomplete to analyse SEN2NEON only)")
        print("warning: no SEN2VENuS P3 rows; SEN2VENuS analyses are skipped")
    order = list(methods_cfg)
    ranked_cfg = [m for m, s in methods_cfg.items() if s.get("role", "ranked") == "ranked"]
    ranked = [m for m in ranked_cfg if m in set(res.method)]
    if len(ranked) < len(ranked_cfg) and not allow_incomplete:
        raise SystemExit(f"ranked methods missing from the results: {sorted(set(ranked_cfg) - set(ranked))}")
    md = [f"# P1 analysis\n\nResult files: {len(files)}; methods: {', '.join(sorted(set(res.method)))}\n",
          f"Ranked methods ({len(ranked)}): {', '.join(ranked)}\n"]

    # ---- descriptive tables
    p1cols = ["psnr", "ssim", "sam", "ergas", "mae_ndre", "mae_ndmi", "hp_corr_mean", "edge_mean"]
    for ds in ("test", "test_ood"):
        t = method_table(_sel(res, "P1", ds), p1cols, order)
        if len(t):
            t.to_csv(out / f"table_p1_{ds}.csv")
            md += [f"\n## P1 reduced scale, {ds}\n", t.round(4).to_markdown(), "\n"]
        t = method_table(_sel(res, "P2", ds), ["psnr", "sam", "hp_corr_mean", "edge_mean"], order)
        if len(t):
            t.to_csv(out / f"table_p2_{ds}.csv")
            md += [f"\n## P2 consistency, {ds}\n", t.round(4).to_markdown(), "\n"]
    for ds, bands in REF_BANDS.items():
        d3 = _sel(res, "P3", ds)
        if not len(d3):
            continue
        cols = ([f"hp_corr_{b}" for b in bands] + ["hp_corr_mean", "edge_mean", "hp_gain_mean", "rmse_band_mean",
                "sam", "ndre_hp_corr"] + (["ndmi_hp_corr"] if ds == "sen2neon" else []))
        t = method_table(d3, cols, order)
        t.to_csv(out / f"table_p3_{ds}.csv")
        md += [f"\n## P3 real reference, {ds} ({d3.unit.nunique()} sites, {d3.key.nunique()} tiles)\n",
               t.round(4).to_markdown(), "\n"]
        for proto in ("P1", "P2"):
            t = method_table(_sel(res, proto, ds), ["hp_corr_mean", "edge_mean"], order)
            if len(t):
                t.to_csv(out / f"table_{proto.lower()}_{ds}.csv")
                md += [f"\n### {proto} on the {ds} tiles (site-matched)\n", t.round(4).to_markdown(), "\n"]

    # ---- RQ1: rank agreement
    rows, rank_tabs = [], []
    comparisons = []   # (name, protocol a, protocol b, band-set tag): tags keep the band sets matched
    for ds in REF_BANDS:
        tag = "mean" if ds == "sen2neon" else "re"
        comparisons += [(f"P1({ds}) vs P3({ds})", _sel(res, "P1", ds), _sel(res, "P3", ds), tag, True),
                        (f"P2({ds}) vs P3({ds})", _sel(res, "P2", ds), _sel(res, "P3", ds), tag, True),
                        (f"P1(test) vs P3({ds})", _sel(res, "P1", "test"), _sel(res, "P3", ds), tag, False)]
    comparisons.append(("P1(test) vs P2(test)", _sel(res, "P1", "test"), _sel(res, "P2", "test"), "mean", True))
    for name, da, db, tag, paired in comparisons:
        if not len(da) or not len(db):
            continue
        for metric, hib in PRIMARY + [("rmse_band", False)]:
            col = f"{metric}_{tag}"
            if col not in da or col not in db or da[col].isna().all() or db[col].isna().all():
                continue
            r, ranks = rank_agreement_units(da, db, col, ranked, higher_is_better=hib, paired=paired)
            if r["n_methods"] < len(ranked_cfg) and not allow_incomplete:
                raise SystemExit(f"{name} / {col}: only {r['n_methods']} of {len(ranked_cfg)} ranked methods")
            rows.append(dict(comparison=name, **r))
            rank_tabs.append(ranks.assign(comparison=name, metric=metric))
    if rows:
        tau = pd.DataFrame(rows)
        tau.to_csv(out / "rank_agreement.csv", index=False)
        rk = pd.concat(rank_tabs)
        rk.to_csv(out / "rank_changes.csv")
        md += ["\n## RQ1 rank agreement (Kendall tau, unit bootstrap 95% CI, one-sided permutation p)\n",
               tau.round(3).to_markdown(index=False), "\n",
               "\n### Methods whose rank changes by 2 or more (primary metric HP-corr)\n",
               rk[(rk.metric == "hp_corr") & (rk.rank_change.abs() >= 2)].round(4).to_markdown(), "\n"]

    # ---- RQ2: scale gap on site-matched tiles
    for ds, bands in REF_BANDS.items():
        d1, d3 = _sel(res, "P1", ds), _sel(res, "P3", ds)
        if len(d1) and len(d3):
            g = scale_gap_units(d1, d3, bands + (["re", "swir"] if ds == "sen2neon" else []), ranked)
            if not len(g):
                md.append(f"\nRQ2 {ds}: no site-matched units\n")
                continue
            g.to_csv(out / f"scale_gap_{ds}.csv", index=False)
            piv = g.pivot(index="method", columns="band", values="gap").loc[[m for m in ranked if m in set(g.method)]]
            if {"re", "swir"} <= set(piv.columns):    # red-edge vs SWIR: positive = larger loss on SWIR
                piv["swir_minus_re"] = piv["swir"] - piv["re"]
            md += [f"\n## RQ2 scale gap HP-corr(P1) - HP-corr(P3), {ds} (site-matched)\n",
                   piv.round(4).to_markdown(), "\n"]

    # ---- RQ3 / RQ4 paired unit-level comparisons, and Friedman
    pair_rows = []
    for (proto, ds), d in res.groupby(["protocol", "dataset"]):
        present = set(d.method)
        for metric in ("hp_corr_mean", "edge_mean"):
            if metric not in d or d[metric].isna().all():
                continue
            for rq, pairs in (("RQ3", RQ3_PAIRS), ("RQ4", RQ4_PAIRS)):
                ok = [p for p in pairs if p[0] in present and p[1] in present]
                if ok:
                    pr = pairwise_units(d, metric, sorted({m for p in ok for m in p}), ok)
                    pair_rows.append(pr.assign(rq=rq, protocol=proto, dataset=ds, metric=metric))
            rk = [m for m in ranked if m in present]
            if len(rk) >= 3 and len(unit_matrix(d, metric, rk)) >= 3:
                fr = friedman_units(d, metric, rk)
                md.append(f"\nFriedman {proto}/{ds}/{metric}: chi2 = {fr['statistic']:.2f}, "
                          f"p = {fr['p']:.3g} ({fr['n_units']} units, {fr['n_methods']} methods)\n")
                if metric in ("hp_corr_mean", "edge_mean"):   # post-hoc: all ranked pairs, Holm over the pairs
                    pair_rows.append(pairwise_units(d, metric, rk).assign(rq="all_pairs", protocol=proto,
                                                                          dataset=ds, metric=metric))
    if pair_rows:
        pr = pd.concat(pair_rows)
        pr.to_csv(out / "paired_rq3_rq4.csv", index=False)
        md += ["\n## RQ3 / RQ4 paired unit-level differences (a - b)\n",
               pr[pr.rq != "all_pairs"][["rq", "protocol", "dataset", "metric", "a", "b", "n_units", "mean_diff",
                                         "ci_low", "ci_high", "units_a_better", "p_signflip_holm"]
                                        ].round(4).to_markdown(index=False), "\n",
               "\nAll-pairs post-hoc comparisons (Holm within each protocol/dataset): paired_rq3_rq4.csv, "
               "rq == 'all_pairs'.\n"]
    (out / "summary.md").write_text("\n".join(md))
    print(f"analysis written to {out}")


__all__ = ["analyze", "add_means", "method_table"]
