r"""LaTeX tables of the P1 paper, generated from results/p1_final so that no number is typed by hand.
Rules are those of the World Scientific journal class (ws-vjcs.cls): \toprule, \colrule, \botrule.

  python scripts/p1_tables.py --res results/p1 --out results/p1/tables
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s2sr.analysis import add_means  # noqa: E402

ORDER = ["bicubic", "atprk", "dsen2", "dsen2_ca_hpf", "lfc_guided", "edsr_u", "edsr_g", "rcan_u", "rcan_g",
         "edsr_g_scr", "dsen2_global"]
NAMES = {"bicubic": "Bicubic", "atprk": "ATPRK", "dsen2": "DSen2", "dsen2_ca_hpf": "DSen2+CA+HPF",
         "lfc_guided": "LFC-guided", "edsr_u": "EDSR-U", "edsr_g": "EDSR-G", "rcan_u": "RCAN-U",
         "rcan_g": "RCAN-G", "edsr_g_scr": "EDSR-G-scr$^\\dagger$", "dsen2_global": "DSen2-global$^\\dagger$"}
COMP = {"P1(sen2neon) vs P3(sen2neon)": "P1 vs P3, SEN2NEON",
        "P2(sen2neon) vs P3(sen2neon)": "P2 vs P3, SEN2NEON",
        "P1(test) vs P3(sen2neon)": "P1 (Vietnam) vs P3, SEN2NEON",
        "P1(sen2venus) vs P3(sen2venus)": "P1 vs P3, SEN2VEN$\\mu$S",
        "P2(sen2venus) vs P3(sen2venus)": "P2 vs P3, SEN2VEN$\\mu$S",
        "P1(test) vs P3(sen2venus)": "P1 (Vietnam) vs P3, SEN2VEN$\\mu$S",
        "P1(test) vs P2(test)": "P1 vs P2, Vietnam"}


def scores(r, protocol, ds, col):
    """Mean over units of seed-averaged unit means, and the sd over seeds of the method score."""
    x = r[(r.protocol == protocol) & (r.dataset == ds)].dropna(subset=[col])
    s = x.groupby(["method", "seed", "unit"])[col].mean().groupby(["method", "seed"]).mean()
    return s.groupby("method").mean(), s.groupby("method").std()


def fmt(v, d=3):
    """Number with a typographic minus sign."""
    return "--" if pd.isna(v) else f"{v:.{d}f}".replace("-", "$-$")


def bold_best(vals: dict, higher: bool, ranked):
    cand = {m: v for m, v in vals.items() if m in ranked and not pd.isna(v)}
    return max(cand, key=cand.get) if higher else min(cand, key=cand.get)


def table_p1(r, out, ranked):
    cols = [("psnr", "PSNR", True, 2), ("ssim", "SSIM", True, 4), ("hp_corr_mean", "HP-corr", True, 4),
            ("edge_mean", "Edge (px)", False, 3), ("mae_ndre", "MAE NDRE", False, 4),
            ("mae_ndmi", "MAE NDMI", False, 4)]
    vals = {c: scores(r, "P1", "test", c)[0] for c, *_ in cols}
    best = {c: bold_best(vals[c].to_dict(), h, ranked) for c, _, h, _ in cols}
    lines = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             "Method & " + " & ".join(n for _, n, _, _ in cols) + r" \\ \colrule"]
    for m in ORDER:
        cells = []
        for c, _, _, d in cols:
            s = fmt(vals[c].get(m), d)
            cells.append(rf"\textbf{{{s}}}" if best[c] == m else s)
        lines.append(f"{NAMES[m]} & " + " & ".join(cells) + r" \\" + (r" \colrule" if m == "rcan_g" else ""))
    lines += [r"\botrule", r"\end{tabular}"]
    (out / "tab_p1.tex").write_text("\n".join(lines) + "\n")


def table_p3(r, out, ranked):
    blocks = [("sen2neon", "mean", "SEN2NEON (31 sites, all six bands)", True),
              ("sen2venus", "re", r"SEN2VEN$\mu$S (five sites, B5--B8A)", False)]
    lines = [r"\begin{tabular}{lccccccc}", r"\toprule",
             r" & \multicolumn{2}{c}{HP-corr} & & \multicolumn{4}{c}{real resolution (P3)} \\",
             r"\cline{2-3}\cline{5-8}",
             r"Method & P1 & P3 & $\Delta$ & Edge (px) & HP-gain & NDRE & NDMI \\"]
    for ds, tag, title, ndmi in blocks:
        p1 = scores(r, "P1", ds, f"hp_corr_{tag}")[0]
        p3 = scores(r, "P3", ds, f"hp_corr_{tag}")[0]
        ed = scores(r, "P3", ds, f"edge_{tag}")[0]
        gn = scores(r, "P3", ds, f"hp_gain_{tag}")[0]
        nd = scores(r, "P3", ds, "ndre_hp_corr")[0]
        nm = scores(r, "P3", ds, "ndmi_hp_corr")[0] if ndmi else pd.Series(dtype=float)
        b1, b3 = bold_best(p1.to_dict(), True, ranked), bold_best(p3.to_dict(), True, ranked)
        be, bn = bold_best(ed.to_dict(), False, ranked), bold_best(nd.to_dict(), True, ranked)
        lines[-1] += r" \colrule"
        lines += [rf"\multicolumn{{8}}{{l}}{{\emph{{{title}}}}} \\"]
        for m in ORDER:
            c = [rf"\textbf{{{fmt(p1.get(m))}}}" if m == b1 else fmt(p1.get(m)),
                 rf"\textbf{{{fmt(p3.get(m))}}}" if m == b3 else fmt(p3.get(m)),
                 fmt(p1.get(m) - p3.get(m)) if m in p1 and m in p3 else "--",
                 rf"\textbf{{{fmt(ed.get(m))}}}" if m == be else fmt(ed.get(m)),
                 fmt(gn.get(m), 2),
                 rf"\textbf{{{fmt(nd.get(m))}}}" if m == bn else fmt(nd.get(m)),
                 fmt(nm.get(m)) if ndmi else "--"]
            lines.append(f"{NAMES[m]} & " + " & ".join(c) + r" \\")
    lines += [r"\botrule", r"\end{tabular}"]
    (out / "tab_p3.tex").write_text("\n".join(lines) + "\n")


def table_tau(res, out):
    """One row per comparison and metric (tau, 95 % CI, permutation p), to fit the 5 in VJCS text width."""
    t = pd.read_csv(res / "analysis" / "rank_agreement.csv")
    t["m"] = t.metric.str.replace("_mean", "").str.replace("_re", "")
    lines = [r"\begin{tabular}{llccc}", r"\toprule",
             r"Comparison & Metric & $\tau$ & 95\,\% CI & $p$ \\ \colrule"]
    rows = []
    for comp, label in COMP.items():
        first = True
        for m, mname in (("hp_corr", "HP-corr"), ("edge", "Edge error"), ("rmse_band", "RMSE")):
            row = t[(t.comparison == comp) & (t.m == m)]
            if not len(row):
                continue
            row = row.iloc[0]
            rows.append(f"{label if first else ''} & {mname} & ${row.tau:+.2f}$ & "
                        f"$[{row.ci_low:+.2f}, {row.ci_high:+.2f}]$ & {row.p_perm:.2f} \\\\")
            first = False
        if rows:
            rows[-1] += r" \colrule" if comp != list(COMP)[-1] else ""
    lines += rows + [r"\botrule", r"\end{tabular}"]
    (out / "tab_tau.tex").write_text("\n".join(lines) + "\n")


BANDS = ["B05", "B06", "B07", "B8A", "B11", "B12"]
BAND_ROWS = ["bicubic", "dsen2", "dsen2_ca_hpf", "lfc_guided", "edsr_g", "rcan_g", "dsen2_global"]


def table_bands(r, out):
    """Per-band HP-corr at real resolution and scale gap on the site-matched SEN2NEON tiles (Appendix C)."""
    p1 = {b: scores(r, "P1", "sen2neon", f"hp_corr_{b}")[0] for b in BANDS}
    p3 = {b: scores(r, "P3", "sen2neon", f"hp_corr_{b}")[0] for b in BANDS}
    head = "Method & " + " & ".join(b.replace("B0", "B") for b in BANDS) + r" \\ \colrule"
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r"\multicolumn{7}{@{}l}{\emph{HP-corr at real resolution (P3)}} \\", head]
    lines += [f"{NAMES[m]} & " + " & ".join(fmt(p3[b].get(m)) for b in BANDS) + r" \\" for m in BAND_ROWS]
    lines[-1] += r" \colrule"
    lines += [r"\multicolumn{7}{@{}l}{\emph{Scale gap $\Delta$ = HP-corr(P1) $-$ HP-corr(P3)}} \\", head]
    lines += [f"{NAMES[m]} & " + " & ".join(fmt(p1[b].get(m) - p3[b].get(m)) for b in BANDS) + r" \\"
              for m in BAND_ROWS]
    lines += [r"\botrule", r"\end{tabular}"]
    (out / "tab_bands.tex").write_text("\n".join(lines) + "\n")


def site_level(r, res):
    """Seed-averaged site means per method, protocol and metric on the real-reference tiles (released CSV)."""
    cols = ["hp_corr_mean", "edge_mean", "hp_gain_mean"] + [f"hp_corr_{b}" for b in BANDS]
    x = r[r.dataset.isin(["sen2neon", "sen2venus"]) & r.protocol.isin(["P1", "P2", "P3", "P3u"])]
    s = (x.groupby(["dataset", "protocol", "method", "seed", "unit"])[cols].mean()
         .groupby(["dataset", "protocol", "method", "unit"]).mean())
    s.round(6).to_csv(res / "analysis" / "site_level.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results/p1")
    ap.add_argument("--out", default="results/p1/tables")
    a = ap.parse_args()
    res, out = Path(a.res), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import yaml

    ranked = [m for m, s in yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/methods.yaml")
                                           .read_text())["methods"].items() if s.get("role") == "ranked"]
    r = add_means(pd.concat([pd.read_csv(f, dtype={"unit": str, "key": str})
                             for f in glob.glob(str(res / "*_seed*.csv"))], ignore_index=True))
    table_p1(r, out, ranked)
    table_p3(r, out, ranked)
    table_tau(res, out)
    table_bands(r, out)
    site_level(r, res)
    print("tables written to", out)


if __name__ == "__main__":
    main()
