#!/usr/bin/env python
"""Step 3: learn to rank native candidates (LightGBM LambdaRank) with an honest protocol.

Protocol: 5-fold cross-validation by 30 %-identity cluster (manifest `fold`), several seeds, baselines on exactly the
same structures, 95 % bootstrap CI over clusters, paired cluster bootstrap of every gain against the geometry
baseline. Success: DCA <= 4 A. Baselines: native order (detector score = largest closed cavity first), cavity volume,
buriedness. External finders (fpocket, P2Rank) are compared only if their candidate tables exist
(scripts/baselines/, optional).

Usage: python scripts/train/train_ranker.py [--ds data/processed] [--tag native] [--seeds 5] [--ablate]
       [--model models/ranker_native.txt] [--features-extra net]
Output: docs/results/ranker_<tag>.json and .md
"""
import argparse, json, pathlib, sys
import numpy as np
import pandas as pd
import lightgbm as lgb

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from equicave import metrics as M, pocket_features as pf  # noqa: E402

PARAMS = dict(objective="lambdarank", metric="ndcg", eval_at=[1, 3], learning_rate=0.05, num_leaves=15, min_data_in_leaf=20,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=4,
              lambdarank_truncation_level=5)


def fit(train: pd.DataFrame, feats, seed: int, rounds: int = 300):
    train = train.sort_values("pdb", kind="stable")
    grp = train.groupby("pdb", sort=False).size().to_numpy()
    return lgb.train(dict(PARAMS, seed=seed), lgb.Dataset(train[feats], train["label"], group=grp), num_boost_round=rounds)


def cv_scores(df: pd.DataFrame, feats, seed: int) -> np.ndarray:
    s = np.full(len(df), np.nan)
    for k in sorted(df["fold"].unique()):
        tr = df["fold"] != k
        s[~tr.to_numpy()] = fit(df[tr], feats, seed).predict(df.loc[~tr, feats])
    return s


def evaluate(df: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    return M.per_structure(df.assign(_s=score), "_s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default=str(REPO / "data/processed")); ap.add_argument("--tag", default="native")
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--model", default=""); ap.add_argument("--features-extra", default="", help="e.g. net: add NET_FEATURES present in the table")
    ap.add_argument("--out", default=str(REPO / "docs/results"))
    a = ap.parse_args()
    ds = pathlib.Path(a.ds)
    df = pd.read_csv(ds / f"candidates_{a.tag}.csv").reset_index(drop=True)
    geo = json.loads((ds / f"candidates_{a.tag}.geometry.json").read_text())
    feats = list(pf.FEATURES)
    if a.features_extra == "net":
        feats += [f for f in pf.NET_FEATURES if f in df]
    feats = [f for f in feats if f in df]
    print(f"{df['pdb'].nunique()} structures, {df['cluster30'].nunique()} clusters, {len(df)} candidates, "
          f"ceiling {(df.groupby('pdb')['label'].max() == 1).mean():.3f}, {len(feats)} features")

    results, per_method = {}, {}
    baselines = {"native order (detector score)": -df["nat_rank"].to_numpy(float),
                 "largest cavity first": df["cav_volume"].to_numpy(float),
                 "most buried first": df["buried_mean"].to_numpy(float)}
    for name, s in baselines.items():
        per_method[name] = [evaluate(df, s)]
    per_method["LightGBM LambdaRank (native features)"] = [evaluate(df, cv_scores(df, feats, sd)) for sd in range(a.seeds)]
    if a.ablate:
        for g, cols in pf.GROUPS.items():
            sub = [f for f in feats if f not in cols]
            if len(sub) < len(feats):
                per_method[f"  without {g}"] = [evaluate(df, cv_scores(df, sub, sd)) for sd in range(max(2, a.seeds // 2))]
        per_method["  native features only"] = [evaluate(df, cv_scores(df, [f for f in pf.NATIVE if f in feats], sd)) for sd in range(max(2, a.seeds // 2))]
    ref = per_method["native order (detector score)"][0]
    for name, runs in per_method.items():
        avg = pd.concat(runs).groupby("pdb", sort=False).agg({c: "mean" for c in ("top1", "top3", "topN", "topN2", "mrr", "ceiling")} | {"cluster30": "first"}).reset_index()
        st = M.summarize(avg)
        st["seed_sd_top1"] = float(np.std([r["top1"].mean() for r in runs])) if len(runs) > 1 else 0.0
        st["paired_vs_native_order"] = {c: M.paired_boot(avg.set_index("pdb").loc[ref["pdb"], c].to_numpy(float), ref[c].to_numpy(float), ref["cluster30"].to_numpy())
                                        for c in ("top1", "topN2")}
        results[name] = st
    lines = ["| method | top-1 | top-3 | top-N | top-(N+2) | MRR | gain top-1 vs native order |", "|---|---|---|---|---|---|---|"]
    for name, st in results.items():
        g = st["paired_vs_native_order"]["top1"]
        lines.append(f"| {name} | {M.fmt(st['top1'])} | {M.fmt(st['top3'])} | {M.fmt(st['topN'])} | {M.fmt(st['topN2'])} | {M.fmt(st['mrr'])} | {g['diff']:+.3f} [{g['lo']:+.3f}, {g['hi']:+.3f}] |")
    table = "\n".join(lines)
    print(table)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"ranker_{a.tag}.json").write_text(json.dumps(dict(n_structures=int(df['pdb'].nunique()), n_clusters=int(df['cluster30'].nunique()),
                                                              n_candidates=len(df), seeds=a.seeds, features=feats, geometry=geo, results=results), indent=1))
    (out / f"ranker_{a.tag}.md").write_text(f"# Ranker results ({a.tag})\n\n{df['pdb'].nunique()} structures, {df['cluster30'].nunique()} clusters, "
                                            f"{len(df)} candidates, ceiling {(df.groupby('pdb')['label'].max() == 1).mean():.3f}; "
                                            f"5-fold CV by cluster, {a.seeds} seeds; 95 % CI by cluster bootstrap; DCA <= 4 A.\n\n{table}\n")
    if a.model:
        m = fit(df, feats, 0); m.save_model(a.model)
        json.dump(feats, open(a.model + ".features.json", "w")); json.dump(pf.geometry(), open(a.model + ".geometry.json", "w"))
        print(f"model saved to {a.model}")


if __name__ == "__main__":
    main()
