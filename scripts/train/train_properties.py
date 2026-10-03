#!/usr/bin/env python
"""Tabular baseline for the pocket-property head: predict the property classes of a site from its candidate features.

The network's property head has to beat something, and the honest something is the same information in tabular form:
the 109 geometry / chemistry / interaction-partner features of the candidate that sits on the site, fed to one
LightGBM classifier per class. Protocol as everywhere: 5-fold cross-validation by 30 %-identity cluster, out-of-fold
predictions only, per-class AUROC and average precision with a cluster bootstrap, expected calibration error, and a
label-permutation control that must destroy the signal.

A site is matched to the candidate whose centre is nearest to it (within --match, default 6 A); sites with no
candidate nearby are reported separately as unmatched, never silently dropped.

Usage: python scripts/train/train_properties.py [--ds data/processed] [--tag native] [--seeds 3]
Output: docs/results/properties_<tag>.md / .json
"""
import argparse, json, pathlib, sys
import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from equicave import labels as LB, metrics as M, pocket_features as pf, tables  # noqa: E402


def parse_center(s: str) -> np.ndarray:
    return np.array([float(x) for x in str(s).split(";")])


def match_sites(sites: pd.DataFrame, cand: pd.DataFrame, max_d: float) -> pd.DataFrame:
    """Join each site to its nearest candidate within `max_d` A; keeps the distance as `match_dist`."""
    cand = cand.copy()
    cand["_xyz"] = cand["center"].map(parse_center)
    rows = []
    for pdb, g in sites.groupby("pdb", sort=False):
        c = cand[cand["pdb"] == pdb]
        if c.empty:
            continue
        C = np.vstack(c["_xyz"].to_numpy())
        for _, s in g.iterrows():
            d = np.linalg.norm(C - parse_center(s["center"]), axis=1)
            j = int(np.argmin(d))
            if d[j] <= max_d:
                row = c.iloc[j].to_dict()
                row.pop("_xyz", None)
                rows.append({**row, **{k: s[k] for k in s.index if k in LB.PROPERTY_CLASSES or k in ("site",)},
                             "match_dist": float(d[j]), "cluster30": s["cluster30"], "fold": int(s["fold"])})
    return pd.DataFrame(rows)


def oof_predict(df: pd.DataFrame, feats: list[str], y: np.ndarray, seed: int) -> np.ndarray:
    import lightgbm as lgb
    p = np.full(len(df), np.nan)
    for k in sorted(df["fold"].unique()):
        te = (df["fold"] == k).to_numpy(); tr = ~te
        if y[tr].sum() < 5 or y[tr].sum() == tr.sum():
            p[te] = y[tr].mean() if tr.any() else 0.0
            continue
        m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=10, feature_fraction=0.8,
                           bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=4, seed=seed,
                           is_unbalance=True), lgb.Dataset(df.loc[tr, feats], y[tr]), num_boost_round=200)
        p[te] = m.predict(df.loc[te, feats])
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default=str(REPO / "data/processed")); ap.add_argument("--tag", default="native")
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--match", type=float, default=6.0)
    ap.add_argument("--out", default=str(REPO / "docs/results"))
    a = ap.parse_args()
    ds = pathlib.Path(a.ds)
    sites = pd.read_csv(ds / "labels_sites.csv")
    cand = tables.read_table(ds, f"candidates_{a.tag}")
    df = match_sites(sites, cand, a.match)
    unmatched = len(sites) - len(df)
    feats = [f for f in pf.FEATURES if f in df]
    print(f"{len(df)} sites matched to a candidate within {a.match} A ({unmatched} unmatched), "
          f"{df['cluster30'].nunique()} clusters, {len(feats)} features")
    results = {}
    for cls in LB.PROPERTY_CLASSES:
        if cls not in df:
            continue
        y = df[cls].to_numpy(float)
        if y.sum() < 10:
            results[cls] = dict(n_positive=int(y.sum()), note="fewer than 10 positives: not evaluated")
            continue
        preds = [oof_predict(df, feats, y, s) for s in range(a.seeds)]
        p = np.mean(preds, axis=0)
        au, lo, hi = M.boot_ci(np.array([M.auroc(y, p)]), np.array(["all"]))  # placeholder, replaced below
        auroc = M.auroc(y, p); ap_ = M.average_precision(y, p)
        # cluster bootstrap of the two metrics, resampling whole clusters
        rng = np.random.default_rng(0); cl = df["cluster30"].to_numpy()
        groups = [np.where(cl == c)[0] for c in pd.unique(cl)]
        aus, aps = [], []
        for _ in range(400):
            idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            if y[idx].sum() == 0 or y[idx].sum() == len(idx):
                continue
            aus.append(M.auroc(y[idx], p[idx])); aps.append(M.average_precision(y[idx], p[idx]))
        perm_auroc, perm_p = M.permutation_control(y, p, M.auroc, n=100)
        results[cls] = dict(n_positive=int(y.sum()), prevalence=float(y.mean()), auroc=auroc,
                            auroc_lo=float(np.percentile(aus, 2.5)) if aus else None,
                            auroc_hi=float(np.percentile(aus, 97.5)) if aus else None,
                            ap=ap_, ap_lo=float(np.percentile(aps, 2.5)) if aps else None,
                            ap_hi=float(np.percentile(aps, 97.5)) if aps else None,
                            ap_over_prevalence=(ap_ / y.mean()) if y.mean() else None,
                            ece=M.ece(y, p), permutation_p=perm_p, seeds=a.seeds)
        print(f"  {cls:18s} n+={int(y.sum()):4d} AUROC {auroc:.3f} AP {ap_:.3f} (prevalence {y.mean():.3f}) "
              f"ECE {results[cls]['ece']:.3f} permutation p {perm_p:.3f}")
    ev = [v for v in results.values() if "auroc" in v]
    macro = dict(auroc=float(np.mean([v["auroc"] for v in ev])), ap=float(np.mean([v["ap"] for v in ev])),
                 ap_over_prevalence=float(np.mean([v["ap_over_prevalence"] for v in ev])), n_classes=len(ev))
    lines = [f"# Pocket property classes: tabular baseline ({a.tag})", "",
             f"{len(df)} ligand sites matched to their candidate within {a.match} Å ({unmatched} sites had no candidate "
             f"that close), {df['cluster30'].nunique()} sequence clusters, {len(feats)} candidate features, "
             f"one LightGBM classifier per class, out-of-fold predictions from 5-fold cross-validation by cluster, "
             f"{a.seeds} seeds averaged. This is the baseline the network's property head has to beat.", "",
             "| class | positives | prevalence | AUROC | AP | AP / prevalence | ECE | permutation p |",
             "|---|---|---|---|---|---|---|---|"]
    for cls, v in results.items():
        if "auroc" not in v:
            lines.append(f"| {cls} | {v['n_positive']} | — | — | — | — | — | {v.get('note', '')} |"); continue
        lines.append(f"| {cls} | {v['n_positive']} | {v['prevalence']:.3f} | {v['auroc']:.3f} "
                     f"[{v['auroc_lo']:.3f}, {v['auroc_hi']:.3f}] | {v['ap']:.3f} [{v['ap_lo']:.3f}, {v['ap_hi']:.3f}] | "
                     f"{v['ap_over_prevalence']:.2f}× | {v['ece']:.3f} | {v['permutation_p']:.3f} |")
    lines += ["", f"Macro average over the {macro['n_classes']} evaluated classes: AUROC {macro['auroc']:.3f}, "
                  f"AP {macro['ap']:.3f}, AP over prevalence {macro['ap_over_prevalence']:.2f}×.",
              "", "A permutation p above 0.05 means the signal for that class is not distinguishable from chance with "
                  "this sample size, and the class is reported as such rather than quietly averaged in."]
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"properties_{a.tag}.md").write_text("\n".join(lines) + "\n")
    (out / f"properties_{a.tag}.json").write_text(json.dumps(dict(n_sites=len(df), unmatched=unmatched,
        n_clusters=int(df["cluster30"].nunique()), features=feats, match=a.match, seeds=a.seeds,
        per_class=results, macro=macro), indent=1))
    print("\n".join(lines[-3:]))


if __name__ == "__main__":
    main()
