"""Metrics for ranking, classification, calibration and hotspot enrichment; bootstrap by cluster.

Ranking (per structure): a candidate hits when DCA <= 4 A (or DCC <= 4 A). top-k = any hit in the first k;
top-N = first n_sites; top-(N+2) = first n_sites + 2 (the DeepPocket / P2Rank convention). A structure whose
candidates contain no hit counts as a miss for every method (the ceiling is reported separately).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def per_structure(df: pd.DataFrame, score_col: str, label_col: str = "label", ks=(1, 3, 5)) -> pd.DataFrame:
    rows = []
    for pdb, g in df.groupby("pdb", sort=False):
        g = g.sort_values(score_col, ascending=False, kind="stable")
        lab = g[label_col].to_numpy().astype(bool)
        n = int(g["n_sites"].iloc[0]) if "n_sites" in g else 1
        first = int(np.argmax(lab)) + 1 if lab.any() else None
        rows.append(dict(pdb=pdb, cluster30=g["cluster30"].iloc[0] if "cluster30" in g else pdb,
                         **{f"top{k}": bool(lab[:k].any()) for k in ks},
                         topN=bool(lab[:n].any()), topN2=bool(lab[:n + 2].any()), first=first,
                         mrr=(1.0 / first) if first else 0.0, ceiling=bool(lab.any())))
    return pd.DataFrame(rows)


def boot_ci(values: np.ndarray, clusters: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95 % percentile CI of `values`, resampling whole clusters."""
    rng = np.random.default_rng(seed)
    groups = pd.Series(values).groupby(pd.Series(clusters)).apply(np.asarray).tolist()
    est = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean() for _ in range(n)]
    return float(np.mean(values)), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def paired_boot(a: np.ndarray, b: np.ndarray, clusters: np.ndarray, n: int = 2000, seed: int = 0) -> dict:
    """Paired cluster bootstrap of the difference a - b (same structures, same order): mean, CI, one-sided p."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    groups = pd.Series(d).groupby(pd.Series(clusters)).apply(np.asarray).tolist()
    est = np.array([np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean() for _ in range(n)])
    return dict(diff=float(d.mean()), lo=float(np.percentile(est, 2.5)), hi=float(np.percentile(est, 97.5)),
                p_le_0=float((est <= 0).mean()))


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, bool); s = np.asarray(s, float)
    if y.all() or (~y).all():
        return float("nan")
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, bool)
    if not y.any():
        return float("nan")
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-width probability bins."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1); out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            out += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(out)


def enrichment_at(y: np.ndarray, s: np.ndarray, frac: float = 0.1) -> float:
    """Fraction of positives among the top `frac` of points divided by the base rate (1 = random)."""
    y = np.asarray(y, bool); s = np.asarray(s, float)
    if not y.any() or len(y) == 0:
        return float("nan")
    k = max(1, int(round(frac * len(y))))
    top = np.argsort(-s)[:k]
    return float(y[top].mean() / y.mean())


def permutation_control(y: np.ndarray, s: np.ndarray, fn, n: int = 100, seed: int = 0) -> tuple[float, float]:
    """Metric `fn(y, s)` against the distribution of the same metric with shuffled scores: (observed, p-value)."""
    rng = np.random.default_rng(seed)
    obs = fn(y, s)
    null = np.array([fn(y, rng.permutation(s)) for _ in range(n)])
    return float(obs), float((null >= obs).mean())


def nms_by_score(df: pd.DataFrame, score_col: str, radius: float, center_col: str = "center") -> pd.DataFrame:
    """Keep, per structure, the best-scoring prediction and then any prediction farther than `radius` from all kept.

    A candidate generator proposes several sub-sites of one cavity; metrics that count predictions (top-1, top-N)
    punish that, while the candidate ceiling rewards it. Merging *after* ranking separates the two effects, and the
    radius becomes a reported choice instead of a hidden one.
    """
    keep = []
    for _, g in df.groupby("pdb", sort=False):
        g = g.sort_values(score_col, ascending=False, kind="stable")
        chosen: list[np.ndarray] = []
        for idx, r in g.iterrows():
            c = r[center_col]
            c = np.array([float(x) for x in str(c).split(";")]) if isinstance(c, str) else np.asarray(c, float)
            if all(np.linalg.norm(c - k) > radius for k in chosen):
                chosen.append(c); keep.append(idx)
    return df.loc[keep]


def redundancy(df: pd.DataFrame, score_col: str, site_col: str = "site_idx", label_col: str = "label",
               top: int | None = None) -> dict:
    """How many predictions point at a site another, better-ranked prediction already found.

    The LIGYSIS comparison showed this dominates the apparent ranking of pocket methods: 67 % of VN-EGNN's
    predictions were redundant (one site predicted up to seven times), and removing redundancy moved several methods
    by 5-13 points of recall. A method that reports no redundancy statistic cannot be compared with one that does, so
    we report it for ourselves: `fraction` of hitting predictions that are repeats, and `sites_per_hit_prediction`.
    `top=n` restricts the count to the first n predictions per structure (n = n_sites + 2 is the useful case).
    """
    rep = hit = 0
    for _, g in df.groupby("pdb", sort=False):
        g = g.sort_values(score_col, ascending=False, kind="stable")
        if top is not None:
            n = int(g["n_sites"].iloc[0]) if "n_sites" in g else 1
            g = g.head(top if isinstance(top, int) else n + 2)
        seen = set()
        for _, r in g.iterrows():
            if not r[label_col]:
                continue
            hit += 1
            s_i = r.get(site_col)
            if s_i in seen:
                rep += 1
            else:
                seen.add(s_i)
    return dict(n_hitting_predictions=int(hit), n_redundant=int(rep),
                fraction=float(rep / hit) if hit else float("nan"),
                sites_per_hit_prediction=float((hit - rep) / hit) if hit else float("nan"))


def summarize(per: pd.DataFrame, cols=("top1", "top3", "topN", "topN2", "mrr")) -> dict:
    out = {}
    for c in cols:
        m, lo, hi = boot_ci(per[c].to_numpy(float), per["cluster30"].to_numpy())
        out[c] = dict(mean=m, lo=lo, hi=hi)
    out["n"] = int(len(per)); out["ceiling"] = float(per["ceiling"].mean())
    return out


def fmt(stat: dict) -> str:
    return f"{stat['mean']:.3f} [{stat['lo']:.3f}, {stat['hi']:.3f}]"


def by_group(rr, groups: dict, score_col: str = "score", label_col: str = "label", id_col: str = "pdb",
             cols=("top1", "top3", "topN", "topN2", "ceiling"), min_n: int = 10) -> dict:
    """`per_structure` restricted to each named group of structure ids.

    One aggregate number hides heterogeneity that is the actual finding: DeepDrug3D's shape-only ablation loses
    about 0.13 on nucleotide pockets and nothing on haem pockets, and our own metal measurement is a 7.3x
    enrichment that yields an AUC of 0.544 -- both invisible in a mean over all structures. No paper in this
    literature stratifies a site-prediction result by pocket chemical class, so this is reported alongside the
    aggregate rather than instead of it.

    Groups smaller than `min_n` structures are returned with their count and no metrics: at n < 10 the standard
    error on a success rate is wider than any effect we could claim.
    """
    out = {}
    for name, ids in groups.items():
        sub = rr[rr[id_col].isin(set(ids))]
        n = sub[id_col].nunique()
        if n < min_n:
            out[name] = dict(n=int(n))
            continue
        per = per_structure(sub, score_col, label_col=label_col)
        out[name] = dict(n=int(n), **{k: float(per[k].mean()) for k in cols if k in per})
    return out


def composition(df, id_col: str = "pdb", site_col: str = "n_sites", cluster_col: str = "cluster30") -> dict:
    """Who is in a subset, not how well it scored: structure and cluster counts, and the site-count distribution.

    Reported beside every homology-filtered subset because filtering changes *which* structures survive, and the
    site-count mix alone moves top-1 -- a structure with one site is a different problem from one with five. An
    earlier version of this analysis read a closing gap between subsets as leakage when it was composition, which
    is why this is printed without being asked for rather than computed when someone suspects it.
    """
    one = df.drop_duplicates(id_col)
    n = one[site_col] if site_col in one else None
    return dict(structures=int(one[id_col].nunique()),
                clusters=int(one[cluster_col].nunique()) if cluster_col in one else None,
                mean_sites=float(n.mean()) if n is not None and len(n) else None,
                single_site_fraction=float((n == 1).mean()) if n is not None and len(n) else None,
                site_counts={int(k): int(v) for k, v in n.value_counts().sort_index().items()}
                            if n is not None and len(n) else {})
