#!/usr/bin/env python
"""Train the per-point ligandability model and emit its candidate aggregates out of fold.

The model answers one question about a single empty cavity point -- would a ligand heavy atom sit here? -- from the
point's own interaction potential, enclosure and local composition (`point_score.POINT_FEATURES`). It is trained by
30 %-identity cluster fold, so a point is only ever scored by a model that never saw its structure, and the per-fold
models are then applied to every candidate of their held-out fold to produce the `point_score.POINT` aggregates the
ranker consumes (`train_ranker.py --features-extra points`).

Reported: out-of-fold AUROC and average precision per point, a permutation control, and -- the number that matters
for the ranker -- top-1 / top-N of ordering candidates by the additive point sum alone, which is the comparison
against our pocket-level features on the same structures.

Usage: python scripts/train/train_point_model.py [--tag native] [--leaves 63] [--rounds 400]
Output: models/point_<tag>_fold<k>.txt, models/point_<tag>.txt, data/processed/point_features_<tag>.csv.gz,
        docs/results/point_model_<tag>.md / .json
"""
import argparse, json, pathlib, sys
import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from equicave import metrics as M, point_score as ps, tables  # noqa: E402


def fit(X, y, leaves, rounds, seed, lr=0.05):
    import lightgbm as lgb
    pos = max(1, int(y.sum()))
    ds = lgb.Dataset(X, label=y, feature_name=list(ps.POINT_FEATURES), free_raw_data=False)
    params = dict(objective="binary", metric="average_precision", num_leaves=leaves, learning_rate=lr,
                  min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  scale_pos_weight=float((len(y) - pos) / pos) ** 0.5,   # square root: reweight without swamping
                  verbosity=-1, seed=seed, num_threads=2)
    return lgb.train(params, ds, num_boost_round=rounds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default=str(REPO / "data/processed")); ap.add_argument("--tag", default="native")
    ap.add_argument("--leaves", type=int, default=63); ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", default=str(REPO / "models")); ap.add_argument("--out", default=str(REPO / "docs/results"))
    a = ap.parse_args()
    ds = pathlib.Path(a.ds)
    # float32 on read, not after: at the largest scale the float64 frame alone is over six gigabytes.
    df = tables.read_table(ds, f"points_{a.tag}",
                           dtype={f: "float32" for f in ps.POINT_FEATURES} | {"occ": "int8", "fold": "int16"})
    feats = [f for f in ps.POINT_FEATURES if f in df]
    if len(feats) != len(ps.POINT_FEATURES):
        sys.exit(f"points_{a.tag} has {len(feats)} of {len(ps.POINT_FEATURES)} point features; rebuild it")
    print(f"{len(df)} points, {df['pdb'].nunique()} structures, {df['occ'].mean():.4f} positive, "
          f"{df['cluster30'].nunique()} clusters")

    X = df[feats].to_numpy(np.float32); y = df["occ"].to_numpy(np.int8)
    oof = np.full(len(df), np.nan)
    models = pathlib.Path(a.models); models.mkdir(exist_ok=True)
    for k in sorted(df["fold"].unique()):
        tr, va = (df["fold"] != k).to_numpy(), (df["fold"] == k).to_numpy()
        b = fit(X[tr], y[tr], a.leaves, a.rounds, a.seed)
        oof[va] = b.predict(X[va])
        b.save_model(str(models / f"point_{a.tag}_fold{k}.txt"))
        print(f"  fold {k}: {tr.sum()} train / {va.sum()} val points, val AP {M.average_precision(y[va], oof[va]):.3f}")
    full = fit(X, y, a.leaves, a.rounds, a.seed)                 # for inference on structures outside the manifest
    full.save_model(str(models / f"point_{a.tag}.txt"))
    (models / f"point_{a.tag}.txt.features.json").write_text(json.dumps(list(feats)))

    rng = np.random.default_rng(a.seed)
    res = dict(n_points=int(len(df)), n_structures=int(df["pdb"].nunique()), positive=float(y.mean()),
               auroc=M.auroc(y, oof), ap=M.average_precision(y, oof),
               ap_permuted=M.average_precision(y, rng.permutation(oof)),
               importance={f: int(v) for f, v in sorted(zip(feats, full.feature_importance("gain")),
                                                        key=lambda kv: -kv[1])[:15]})

    # candidate aggregates from the out-of-fold point scores, and what ranking by the sum alone is worth
    df = df.assign(_s=oof)
    rows = []
    for (pdb, center), g in df.groupby(["pdb", "center"], sort=False):
        pts = g[["x", "y", "z"]].to_numpy(float)
        cen = np.array([float(v) for v in center.split(";")])
        rows.append(dict(pdb=pdb, center=center, fold=int(g["fold"].iloc[0]),
                         **ps.aggregate(g["_s"].to_numpy(float), pts, cen)))
    agg = pd.DataFrame(rows)
    tables.write_table(agg, ds, f"point_features_{a.tag}")
    print(f"wrote point_features_{a.tag} for {len(agg)} candidates")

    try:
        cand = tables.read_table(ds, f"candidates_{a.tag}")[["pdb", "center", "label", "n_sites", "cluster30", "nat_score"]]
        j = cand.merge(agg, on=["pdb", "center"], how="inner")
        print(f"{len(j)} candidates joined to the candidate table")
        for col in ("pts_sum", "pts_blob_max", "pts_max", "nat_score"):
            if col in j:
                st = M.summarize(M.per_structure(j, col))
                res.setdefault("ranking_by_single_feature", {})[col] = st
    except FileNotFoundError:
        print(f"  candidates_{a.tag} not found; the single-feature ranking check is skipped")

    lines = [f"# Per-point ligandability model ({a.tag})", "",
             f"{res['n_points']} cavity grid points from {res['n_structures']} structures, "
             f"{res['positive']:.4f} of them within {json.loads((ds / f'points_{a.tag}.meta.json').read_text())['occ_radius']} A "
             f"of a ligand heavy atom. Out-of-fold by 30 %-identity cluster, {len(ps.POINT_FEATURES)} point features.", "",
             f"| metric | value |", "|---|---|",
             f"| point AUROC | {res['auroc']:.3f} |", f"| point average precision | {res['ap']:.3f} |",
             f"| average precision, scores permuted | {res['ap_permuted']:.3f} |",
             f"| positive rate (the trivial baseline) | {res['positive']:.4f} |"]
    if "ranking_by_single_feature" in res:
        lines += ["", "Ordering candidates by one aggregate alone, on the structures of this table. `nat_score` is the "
                  "generator's own geometric score and is the baseline to beat; the learned ranker uses all 118 "
                  "pocket features and is reported in `ranker_<tag>.md`.", "",
                  "| ordering | top-1 | top-3 | top-N | top-(N+2) | MRR | ceiling | n |", "|---|---|---|---|---|---|---|"]
        for col, st in res["ranking_by_single_feature"].items():
            lines.append(f"| {col} | {M.fmt(st['top1'])} | {M.fmt(st['top3'])} | {M.fmt(st['topN'])} | "
                         f"{M.fmt(st['topN2'])} | {M.fmt(st['mrr'])} | {st['ceiling']:.3f} | {st['n']} |")
    lines += ["", "Gain of the fifteen most used point features:", "",
              "| feature | gain |", "|---|---|"] + [f"| {f} | {v} |" for f, v in res["importance"].items()]
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"point_model_{a.tag}.md").write_text("\n".join(lines) + "\n")
    (out / f"point_model_{a.tag}.json").write_text(json.dumps(res, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
