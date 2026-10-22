"""Figures of the P1 paper from results/p1_final (written by `run.py eval` and `run.py analyze`).

  python scripts/p1_figures.py --res results/p1 --out results/p1/figures

fig_rank_slope.pdf : method ranks by HP-corr, site-matched reduced scale (P1) -> real resolution (P3)
fig_mechanism.pdf  : detail amplitude (HP-gain) vs structural fidelity (HP-corr) at real resolution, guided methods,
                     centred on each site's mean (paired comparison)
fig_scale_gap.pdf  : HP-corr(P1) - HP-corr(P3) per method, red-edge vs SWIR (SEN2NEON, site-matched)
Colour encodes the method family (validated palette: retrained guided, original DSen2 weights, unguided or
scrambled guide); classical baselines are grey; every mark carries a direct label.
Figures are drawn at the 5 in text width of the VJCS (World Scientific) layout so that no reduction is needed, with
lettering of at least 6.5 pt in a Helvetica-metric font, as the journal's figure instructions ask.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s2sr.analysis import add_means  # noqa: E402

NAMES = {"bicubic": "Bicubic", "atprk": "ATPRK", "dsen2": "DSen2", "dsen2_ca_hpf": "DSen2+CA+HPF",
         "lfc_guided": "LFC-guided", "edsr_u": "EDSR-U", "edsr_g": "EDSR-G", "rcan_u": "RCAN-U",
         "rcan_g": "RCAN-G", "edsr_g_scr": "EDSR-G-scr", "dsen2_global": "DSen2-global"}
FAMILY = {"dsen2": "g", "dsen2_ca_hpf": "g", "lfc_guided": "g", "edsr_g": "g", "rcan_g": "g",
          "dsen2_global": "w", "edsr_u": "u", "rcan_u": "u", "edsr_g_scr": "u", "bicubic": "c", "atprk": "c"}
COLOR = {"g": "#2a78d6", "w": "#eb6834", "u": "#1baf7a", "c": "#8a8a85"}
MARKER = {"g": "o", "w": "D", "u": "s", "c": "^"}
LABEL = {"g": "guided, retrained (Wald)", "w": "DSen2, original global weights", "u": "unguided / scrambled guide",
         "c": "classical"}
INK, MUTED, GRID = "#2b2b29", "#6b6a64", "#e4e3de"
TAGS = {"sen2neon": ("mean", "SEN2NEON (all six bands)"), "sen2venus": ("re", "SEN2VENµS (B5-B8A)")}


def style():
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
                         "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
                         "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold",
                         "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "pdf.fonttype": 42, "lines.linewidth": 2})


def site_scores(r, protocol, ds, col):
    """Seed-averaged site means per method: DataFrame units x methods."""
    x = r[(r.protocol == protocol) & (r.dataset == ds)]
    g = x.groupby(["method", "seed", "unit"])[col].mean().groupby(["method", "unit"]).mean()
    return g.unstack("method").dropna(axis=0)


def boot_ci(v, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return np.percentile(b, [2.5, 97.5])


def fig_rank_slope(r, out):
    fig, axes = plt.subplots(2, 1, figsize=(5.0, 5.6))
    for ax, (ds, (tag, title)) in zip(axes, TAGS.items()):
        a = site_scores(r, "P1", ds, f"hp_corr_{tag}").mean()
        b = site_scores(r, "P3", ds, f"hp_corr_{tag}").mean()
        ms = [m for m in NAMES if m in a.index and m in b.index]
        ra, rb = (-a[ms]).rank(method="min"), (-b[ms]).rank(method="min")
        for m in ms:
            f = FAMILY[m]
            ax.plot([0, 1], [ra[m], rb[m]], color=COLOR[f], lw=1.6, alpha=0.9, zorder=2)
            ax.scatter([0, 1], [ra[m], rb[m]], color=COLOR[f], marker=MARKER[f], s=28, zorder=3,
                       edgecolor="white", linewidth=0.8)
            ax.text(-0.05, ra[m], f"{NAMES[m]}  {a[m]:.3f}", ha="right", va="center", fontsize=6.5, color=INK)
            ax.text(1.05, rb[m], f"{b[m]:.3f}  {NAMES[m]}", ha="left", va="center", fontsize=6.5, color=INK)
        ax.set_xlim(-0.75, 1.75)
        ax.set_ylim(len(ms) + 0.6, 0.4)
        ax.set_xticks([0, 1], ["reduced scale\n(site-matched P1)", "real resolution\n(P3)"])
        ax.set_yticks([])
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_title(title, fontsize=8)
    handles = [plt.Line2D([], [], color=COLOR[f], marker=MARKER[f], lw=1.6, label=LABEL[f]) for f in "gwuc"]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "fig_rank_slope.pdf")
    plt.close(fig)


GUIDED = ["rcan_g", "lfc_guided", "edsr_g", "dsen2_ca_hpf", "dsen2", "dsen2_global"]


def fig_mechanism(r, out):
    """Guided methods only. Values are centred on each site's mean over these methods, so that the 95 % CIs
    (bootstrap over sites) reflect the paired, within-site comparison that the analysis tests."""
    fig, axes = plt.subplots(2, 1, figsize=(5.0, 5.0))
    for ax, (ds, (tag, title)) in zip(axes, TAGS.items()):
        gain = site_scores(r, "P3", ds, f"hp_gain_{tag}")[GUIDED]
        corr = site_scores(r, "P3", ds, f"hp_corr_{tag}")[GUIDED]
        gc, cc = gain.sub(gain.mean(axis=1), axis=0), corr.sub(corr.mean(axis=1), axis=0)
        pts = {}
        for m in GUIDED:
            f = FAMILY[m]
            gx, cy = gc[m].to_numpy(), cc[m].to_numpy()
            (xlo, xhi), (ylo, yhi) = boot_ci(gx), boot_ci(cy)
            ax.errorbar(gx.mean(), cy.mean(), xerr=[[gx.mean() - xlo], [xhi - gx.mean()]],
                        yerr=[[cy.mean() - ylo], [yhi - cy.mean()]], fmt=MARKER[f], color=COLOR[f], ms=5,
                        mec="white", mew=0.8, elinewidth=1, capsize=0, zorder=3)
            pts[m] = (gx.mean(), cy.mean())
        # labels: the global model beside its point, the retrained cluster in a column with leader lines
        gx0, gy0 = pts["dsen2_global"]
        ax.annotate(NAMES["dsen2_global"], (gx0, gy0), xytext=(6, -10), textcoords="offset points", fontsize=6.5)
        retr = sorted([m for m in GUIDED if m != "dsen2_global"], key=lambda m: -pts[m][1])
        xs = [pts[m][0] for m in retr]
        ys = [pts[m][1] for m in retr]
        x_lab = min(xs) - 0.42 * (max(xs) - gx0)
        y_top, y_bot = max(ys) + 0.15 * (gy0 - min(ys)), min(ys)
        for k, m in enumerate(retr):
            yl = y_top + (y_bot - y_top) * k / (len(retr) - 1)
            ax.annotate(NAMES[m], pts[m], xytext=(x_lab, yl), textcoords="data", fontsize=6.5, ha="right",
                        va="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=1, shrinkB=3))
        ax.axhline(0, color=GRID, lw=0.8, zorder=0)
        ax.axvline(0, color=GRID, lw=0.8, zorder=0)
        ax.set_xlabel("HP-gain relative to the site mean")
        ax.set_ylabel("HP-corr relative to the site mean")
        ax.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_mechanism.pdf")
    plt.close(fig)


def fig_scale_gap(res_dir: Path, out):
    g = pd.read_csv(res_dir / "analysis" / "scale_gap_sen2neon.csv")
    g = g[g.band.isin(["re", "swir"])]
    ms = [m for m in NAMES if m in set(g.method)]
    fig, ax = plt.subplots(figsize=(5.0, 2.5))
    for i, m in enumerate(ms):
        f = FAMILY[m]
        for band, mk, dx in (("re", "o", -0.12), ("swir", "s", 0.12)):
            row = g[(g.method == m) & (g.band == band)].iloc[0]
            ax.errorbar(i + dx, row.gap, yerr=[[row.gap - row.ci_low], [row.ci_high - row.gap]], fmt=mk,
                        color=COLOR[f], mfc=COLOR[f] if band == "re" else "white", ms=5, elinewidth=1, capsize=0)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(ms)), [NAMES[m] for m in ms], rotation=30, ha="right")
    ax.set_ylabel("HP-corr(P1) $-$ HP-corr(P3)")
    ax.grid(axis="x", visible=False)
    h = [plt.Line2D([], [], color=INK, marker="o", lw=0, label="red-edge (B5-B8A)"),
         plt.Line2D([], [], color=INK, marker="s", mfc="white", lw=0, label="SWIR (B11, B12)")]
    ax.legend(handles=h, frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig_scale_gap.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results/p1")
    ap.add_argument("--out", default="results/p1/figures")
    a = ap.parse_args()
    res, out = Path(a.res), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    style()
    r = add_means(pd.concat([pd.read_csv(f, dtype={"unit": str, "key": str})
                             for f in glob.glob(str(res / "*_seed*.csv"))], ignore_index=True))
    fig_rank_slope(r, out)
    fig_mechanism(r, out)
    fig_scale_gap(res, out)
    print("figures written to", out)


if __name__ == "__main__":
    main()
