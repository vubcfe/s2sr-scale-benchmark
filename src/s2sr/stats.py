"""Statistics for the paper's analyses (RQ1-RQ4).

rank_agreement : Kendall tau between method rankings under two protocols, with a bootstrap CI
                 over samples (methods' mean scores are recomputed on each resample).
friedman       : omnibus test across methods on paired samples.
pairwise_wilcoxon_holm : all method pairs, Wilcoxon signed-rank, Holm-corrected p-values.
Unit-level versions (used by s2sr.analysis): rank_agreement_units, friedman_units, pairwise_units,
scale_gap_units (HP-corr P1 - P3 on site-matched tiles, relative to bicubic).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def _wide(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """samples x methods matrix of a metric, averaged over seeds; only complete rows kept."""
    w = df.groupby(["idx", "method"])[metric].mean().unstack("method")
    return w.dropna()


def rank_agreement(df_a: pd.DataFrame, df_b: pd.DataFrame, metric: str, lower_is_better: bool = True,
                   n_boot: int = 2000, seed: int = 0) -> dict:
    """Kendall tau between rankings from two protocols (different sample sets are allowed)."""
    wa, wb = _wide(df_a, metric), _wide(df_b, metric)
    methods = sorted(set(wa.columns) & set(wb.columns))
    wa, wb = wa[methods].to_numpy(), wb[methods].to_numpy()
    sign = 1.0 if lower_is_better else -1.0
    tau = stats.kendalltau(sign * wa.mean(0), sign * wb.mean(0)).statistic
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        ia = rng.integers(0, len(wa), len(wa))
        ib = rng.integers(0, len(wb), len(wb))
        boots.append(stats.kendalltau(wa[ia].mean(0), wb[ib].mean(0)).statistic)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return dict(metric=metric, n_methods=len(methods), tau=float(tau), ci_low=float(lo), ci_high=float(hi))


def friedman(df: pd.DataFrame, metric: str) -> dict:
    w = _wide(df, metric)
    res = stats.friedmanchisquare(*[w[c].to_numpy() for c in w.columns])
    return dict(metric=metric, n_samples=len(w), statistic=float(res.statistic), p=float(res.pvalue))


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (m - k) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def pairwise_wilcoxon_holm(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    w = _wide(df, metric)
    rows = []
    for a, b in combinations(w.columns, 2):
        d = w[a] - w[b]
        p = stats.wilcoxon(w[a], w[b]).pvalue if (d != 0).any() else 1.0
        rows.append(dict(a=a, b=b, median_diff=float(d.median()), p=float(p)))
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"].tolist())
    return out


# ---------------------------------------------------------------- unit-level statistics (P1 paper, Sec. 3.6)
# Samples within a block (Vietnamese tests) or a site (real references) are spatially correlated, so all
# inference resamples units. A score is first averaged over the samples of a unit for each (method, seed).

def unit_seed_means(df: pd.DataFrame, metric: str, methods: list[str] | None = None) -> pd.DataFrame:
    """(unit, method, seed) means of a metric; units missing any method are dropped."""
    d = df.dropna(subset=[metric])
    if methods is not None:
        d = d[d["method"].isin(methods)]
    g = d.groupby(["unit", "method", "seed"])[metric].mean().reset_index()
    per_unit = g.groupby("unit")["method"].nunique()
    n_methods = g["method"].nunique()
    return g[g["unit"].isin(per_unit[per_unit == n_methods].index)]


def method_scores(g: pd.DataFrame) -> pd.Series:
    """Score per method: mean over units of the seed-averaged unit means."""
    return g.groupby(["unit", "method"])[metric_col(g)].mean().unstack("method").mean()


def metric_col(g: pd.DataFrame) -> str:
    return [c for c in g.columns if c not in ("unit", "method", "seed")][0]


def _tau(a: np.ndarray, b: np.ndarray) -> float:
    return float(stats.kendalltau(a, b).statistic)


def _cube(g: pd.DataFrame, units, methods) -> np.ndarray:
    """(unit, method, seed) array of unit means; NaN where a method has fewer seeds."""
    col = metric_col(g)
    seeds = sorted(g["seed"].unique())
    arr = np.full((len(units), len(methods), len(seeds)), np.nan)
    ui = {u: k for k, u in enumerate(units)}
    mi = {m: k for k, m in enumerate(methods)}
    si = {s: k for k, s in enumerate(seeds)}
    for u, m, s, v in g[["unit", "method", "seed", col]].itertuples(index=False):
        arr[ui[u], mi[m], si[s]] = v
    return arr


def _seed_picks(cube: np.ndarray, rng) -> list[np.ndarray]:
    """Resampled seed indices per method (only among the seeds the method has)."""
    picks = []
    for m in range(cube.shape[1]):
        avail = np.where(~np.isnan(cube[:, m, :]).all(axis=0))[0]
        picks.append(rng.choice(avail, len(avail), replace=True))
    return picks


def _boot_scores(cube: np.ndarray, unit_idx: np.ndarray, rng, picks: list[np.ndarray] | None = None) -> np.ndarray:
    """One hierarchical bootstrap replicate: the given units, and seeds resampled within each method (`picks`
    lets paired comparisons use the same resampled seeds for both protocols)."""
    picks = picks if picks is not None else _seed_picks(cube, rng)
    out = np.empty(cube.shape[1])
    sub = cube[unit_idx]
    for m in range(cube.shape[1]):
        out[m] = np.nanmean(sub[:, m, picks[m]], axis=1).mean()
    return out


def pair_rows(df_a: pd.DataFrame, df_b: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the (method, seed, key) cells with a value in both protocols (site-matched comparisons are paired
    down to the tile)."""
    ka = set(map(tuple, df_a.dropna(subset=[metric])[["method", "seed", "key"]].astype(str).to_numpy()))
    kb = set(map(tuple, df_b.dropna(subset=[metric])[["method", "seed", "key"]].astype(str).to_numpy()))
    both = ka & kb
    keep = lambda d: d[[tuple(r) in both for r in d[["method", "seed", "key"]].astype(str).to_numpy()]]  # noqa
    return keep(df_a), keep(df_b)


def _perm_taus(x: np.ndarray, y: np.ndarray, n_perm: int, rng, chunk: int = 20000) -> np.ndarray:
    """Kendall tau of x with random permutations of y (vectorised; scipy's tau-b when there are ties)."""
    n = len(x)
    if len(np.unique(x)) < n or len(np.unique(y)) < n:
        return np.array([_tau(x, rng.permutation(y)) for _ in range(n_perm)])
    i, j = np.triu_indices(n, 1)
    sx = np.sign(x[j] - x[i])
    out = []
    for start in range(0, n_perm, chunk):
        m = min(chunk, n_perm - start)
        yp = y[np.argsort(rng.random((m, n)), axis=1)]
        out.append((np.sign(yp[:, j] - yp[:, i]) * sx).sum(1) / len(i))
    return np.concatenate(out)


def rank_agreement_units(df_a: pd.DataFrame, df_b: pd.DataFrame, metric: str, methods: list[str],
                         higher_is_better: bool = True, n_boot: int = 1000, n_perm: int = 200000,
                         seed: int = 0, paired: bool | None = None) -> tuple[dict, pd.DataFrame]:
    """Kendall tau between the method rankings of two protocols, with a hierarchical bootstrap CI over units
    and a one-sided Monte Carlo permutation test of agreement beyond chance. Also returns the per-method ranks.
    paired=True (site-matched protocols on the same tiles): both sides are restricted to their common units and
    the units are resampled jointly; paired=None decides from the data (identical unit sets)."""
    if paired:
        df_a, df_b = pair_rows(df_a, df_b, metric)
        if not len(df_a):
            raise ValueError(f"{metric}: no tiles with a value in both protocols")
    ga, gb = unit_seed_means(df_a, metric, methods), unit_seed_means(df_b, metric, methods)
    common = sorted(set(ga["method"]) & set(gb["method"]))
    ga, gb = ga[ga["method"].isin(common)], gb[gb["method"].isin(common)]
    sa, sb = method_scores(ga)[common], method_scores(gb)[common]
    tau = _tau(sa.to_numpy(), sb.to_numpy())
    rng = np.random.default_rng(seed)
    if paired:
        common_u = sorted(set(ga["unit"]) & set(gb["unit"]))
        if not common_u:
            raise ValueError(f"{metric}: no common units for a paired comparison")
        ga, gb = ga[ga["unit"].isin(common_u)], gb[gb["unit"].isin(common_u)]
        sa, sb = method_scores(ga)[common], method_scores(gb)[common]
        tau = _tau(sa.to_numpy(), sb.to_numpy())
    ua, ub = sorted(ga["unit"].unique()), sorted(gb["unit"].unique())
    paired = bool(paired) if paired is not None else ua == ub
    ca, cb = _cube(ga, ua, common), _cube(gb, ub, common)
    same_seeds = paired and ca.shape[2] == cb.shape[2]
    boots = []
    for _ in range(n_boot):
        ia = rng.integers(0, len(ua), len(ua))
        ib = ia if paired else rng.integers(0, len(ub), len(ub))
        pa = _seed_picks(ca, rng)
        pb = pa if same_seeds else _seed_picks(cb, rng)
        boots.append(_tau(_boot_scores(ca, ia, rng, pa), _boot_scores(cb, ib, rng, pb)))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    x, y = sa.to_numpy(), sb.to_numpy()
    p = (1 + (_perm_taus(x, y, n_perm, rng) >= tau - 1e-12).sum()) / (n_perm + 1)
    sign = -1.0 if higher_is_better else 1.0
    ranks = pd.DataFrame(dict(score_a=sa, score_b=sb,
                              rank_a=(sign * sa).rank(method="min").astype(int),
                              rank_b=(sign * sb).rank(method="min").astype(int)))
    ranks["rank_change"] = ranks["rank_b"] - ranks["rank_a"]
    res = dict(metric=metric, n_methods=len(common), n_units_a=len(ua), n_units_b=len(ub), paired=paired,
               tau=tau, ci_low=float(lo), ci_high=float(hi), p_perm=float(p))
    return res, ranks.sort_values("rank_a")


def unit_matrix(df: pd.DataFrame, metric: str, methods: list[str]) -> pd.DataFrame:
    """units x methods matrix of seed-averaged unit means."""
    g = unit_seed_means(df, metric, methods)
    return g.groupby(["unit", "method"])[metric].mean().unstack("method").dropna()


def friedman_units(df: pd.DataFrame, metric: str, methods: list[str]) -> dict:
    w = unit_matrix(df, metric, methods)
    res = stats.friedmanchisquare(*[w[c].to_numpy() for c in w.columns])
    return dict(metric=metric, n_units=len(w), n_methods=w.shape[1], statistic=float(res.statistic),
                p=float(res.pvalue))


def sign_flip_p(d: np.ndarray, n_mc: int = 200000, seed: int = 0) -> float:
    """Two-sided sign-flip test of mean(d) = 0 over paired unit differences (exact for n <= 16)."""
    d = np.asarray(d, float)
    n, obs = len(d), abs(d.mean())
    if n <= 16:
        signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    else:
        signs = np.random.default_rng(seed).choice([-1, 1], size=(n_mc, n))
    return float((np.abs((signs * d).mean(1)) >= obs - 1e-15).mean())


def pairwise_units(df: pd.DataFrame, metric: str, methods: list[str], pairs: list[tuple[str, str]] | None = None
                   ) -> pd.DataFrame:
    """Paired unit-level comparisons: mean difference, hierarchical bootstrap CI (units jointly, seeds within
    each method), Wilcoxon and sign-flip p over the seed-averaged unit differences (Holm over the pairs)."""
    pairs = pairs or list(combinations([m for m in methods if m in set(df["method"])], 2))
    rng = np.random.default_rng(0)
    rows = []
    for a, b in pairs:      # units complete for this pair only
        w = unit_matrix(df, metric, [a, b])
        if len(w) < 2:
            continue
        units = sorted(w.index)
        g = unit_seed_means(df, metric, [a, b])
        g = g[g["unit"].isin(units)]
        d = (w.loc[units, a] - w.loc[units, b]).to_numpy()
        cube = _cube(g, units, [a, b])
        boots = []
        for _ in range(2000):
            s = _boot_scores(cube, rng.integers(0, len(units), len(units)), rng)
            boots.append(s[0] - s[1])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        pw = stats.wilcoxon(d).pvalue if (d != 0).any() else 1.0
        rows.append(dict(a=a, b=b, n_units=len(d), mean_diff=float(d.mean()), ci_low=float(lo), ci_high=float(hi),
                         units_a_better=int((d > 0).sum()), p_wilcoxon=float(pw), p_signflip=sign_flip_p(d)))
    out = pd.DataFrame(rows)
    if len(out):
        out["p_wilcoxon_holm"] = holm(out["p_wilcoxon"].tolist())
        out["p_signflip_holm"] = holm(out["p_signflip"].tolist())
    return out


def scale_gap_units(p1: pd.DataFrame, p3: pd.DataFrame, bands: list[str], methods: list[str],
                    baseline: str = "bicubic", metric: str = "hp_corr", n_boot: int = 2000) -> pd.DataFrame:
    """Delta_b = score_P1 - score_P3 per method and band on site-matched tiles (same units), and
    Delta_b / Delta_b(baseline). CIs from a hierarchical bootstrap: shared units jointly, seeds within methods."""
    rng = np.random.default_rng(0)
    rows = []
    for band in bands:
        col = f"{metric}_{band}"
        if col not in p1 or col not in p3:
            continue
        q1, q3 = pair_rows(p1, p3, col)
        g1, g3 = unit_seed_means(q1, col, methods), unit_seed_means(q3, col, methods)
        units = sorted(set(g1["unit"]) & set(g3["unit"]))
        ms = sorted(set(g1["method"]) & set(g3["method"]))
        if not units or not ms:
            continue
        c1 = _cube(g1[g1["unit"].isin(units) & g1["method"].isin(ms)], units, ms)
        c3 = _cube(g3[g3["unit"].isin(units) & g3["method"].isin(ms)], units, ms)
        point = np.nanmean(c1, axis=2).mean(0) - np.nanmean(c3, axis=2).mean(0)
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(units), len(units))
            pk = _seed_picks(c1, rng)
            boots.append(_boot_scores(c1, idx, rng, pk) - _boot_scores(c3, idx, rng, pk))
        boots = np.array(boots)
        b_i = ms.index(baseline) if baseline in ms else None
        for k, m in enumerate(ms):
            # the ratio is interpretable only when interpolation itself loses fidelity at real resolution
            rel = point[k] / point[b_i] if b_i is not None and point[b_i] > 1e-3 else np.nan
            rows.append(dict(method=m, band=band, n_units=len(units), gap=float(point[k]),
                             ci_low=float(np.percentile(boots[:, k], 2.5)),
                             ci_high=float(np.percentile(boots[:, k], 97.5)), gap_rel=float(rel)))
    return pd.DataFrame(rows)
