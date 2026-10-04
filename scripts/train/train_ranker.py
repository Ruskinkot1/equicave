#!/usr/bin/env python
"""Step 3: learn to rank candidates (LightGBM LambdaRank) with an honest protocol.

Protocol: 5-fold cross-validation by 30 %-identity cluster (manifest `fold`), several seeds whose scores are averaged
(an ensemble, which is what one would ship), baselines on exactly the same structures, 95 % bootstrap CI over
clusters, and a paired cluster bootstrap of every gain against the detector order. Success: DCA <= 4 A.

Three things beyond a plain LambdaRank fit, each switchable so its contribution can be measured:
  --graded        relevance 2 for DCA <= 2 A, 1 for <= 4 A, 0 otherwise, instead of a binary label. Ranking a centre
                  that sits on the ligand above one that merely touches it is what top-1 actually rewards.
  --zscore        every feature is also given as a z-score within its own structure. Pocket ranking is a comparison
                  inside one protein, so "deeper than the other cavities of this protein" is the informative form.
  --ensemble      the final score is the mean over seeds (reported next to the per-seed mean).
  calibration     a LambdaRank score is not a probability, so an isotonic regression maps it to P(the candidate hits a
                  ligand). It is fitted per fold on the *other* folds' out-of-fold scores, never on the fold it
                  scores, and the expected calibration error is reported before and after.

Baselines: detector order, largest cavity, most buried. External finders (fpocket, P2Rank) are compared by
scripts/eval/compare_methods.py on the same structures.

Usage: python scripts/train/train_ranker.py [--ds data/processed] [--tag native] [--seeds 5] [--ablate]
       [--model models/ranker_native.txt] [--features-extra net] [--no-graded] [--no-zscore]
Output: docs/results/ranker_<tag>.json and .md
"""
import argparse, json, pathlib, sys
import numpy as np
import pandas as pd
import lightgbm as lgb

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from equicave import metrics as M, pocket_features as pf, tables  # noqa: E402

PARAMS = dict(objective="lambdarank", metric="ndcg", eval_at=[1, 3], learning_rate=0.05, num_leaves=31, min_data_in_leaf=20,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=4,
              lambdarank_truncation_level=10, label_gain=[0, 1, 3])
TARGET = "relevance"


def add_relevance(df: pd.DataFrame, graded: bool = True) -> pd.DataFrame:
    """Graded relevance: 2 if the centre is within 2 A of a ligand atom, 1 within 4 A, 0 otherwise."""
    df = df.copy()
    df[TARGET] = df["label"].astype(int)
    if graded and "dca" in df:
        df[TARGET] = np.where(df["dca"] <= 2.0, 2, df["label"].astype(int))
    return df


def add_zscores(df: pd.DataFrame, feats: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Each feature also as a z-score inside its own structure: ranking is a comparison within one protein."""
    g = df.groupby("pdb", sort=False)[feats]
    z = (df[feats] - g.transform("mean")) / (g.transform("std") + 1e-6)
    z.columns = [f"{c}_z" for c in feats]
    return pd.concat([df, z.fillna(0.0)], axis=1), feats + list(z.columns)


def fit(train: pd.DataFrame, feats, seed: int, rounds: int = 400):
    train = train.sort_values("pdb", kind="stable")
    grp = train.groupby("pdb", sort=False).size().to_numpy()
    y = train[TARGET] if TARGET in train else train["label"]
    return lgb.train(dict(PARAMS, seed=seed), lgb.Dataset(train[feats], y, group=grp), num_boost_round=rounds)


def cv_scores(df: pd.DataFrame, feats, seed: int) -> np.ndarray:
    s = np.full(len(df), np.nan)
    for k in sorted(df["fold"].unique()):
        tr = df["fold"] != k
        s[~tr.to_numpy()] = fit(df[tr], feats, seed).predict(df.loc[~tr, feats])
    return s


def evaluate(df: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    return M.per_structure(df.assign(_s=score), "_s")


def calibrate(df: pd.DataFrame, score: np.ndarray) -> tuple[np.ndarray, dict]:
    """Isotonic map from LambdaRank score to P(hit), fitted out of fold; returns (probabilities, calibration report).

    Fold k is calibrated by an isotonic regression fitted on the out-of-fold scores of every other fold, so no
    candidate is calibrated by a model that saw it. Ranking is unchanged within a structure (the map is monotone);
    what changes is that the number can be read as a probability.
    """
    from sklearn.isotonic import IsotonicRegression
    p = np.full(len(df), np.nan)
    folds = sorted(df["fold"].unique())
    for k in folds:
        te = (df["fold"] == k).to_numpy()
        tr = ~te & np.isfinite(score)
        if tr.sum() < 50 or df.loc[tr, "label"].nunique() < 2:
            p[te] = 1 / (1 + np.exp(-score[te])); continue
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(score[tr], df.loc[tr, "label"].to_numpy(float))
        p[te] = iso.predict(score[te])
    y = df["label"].to_numpy(float)
    raw = 1 / (1 + np.exp(-score))
    return p, dict(ece_raw_sigmoid=M.ece(y, raw), ece_isotonic=M.ece(y, p),
                   brier_raw_sigmoid=float(np.mean((raw - y) ** 2)), brier_isotonic=float(np.mean((p - y) ** 2)),
                   mean_predicted=float(np.nanmean(p)), base_rate=float(y.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default=str(REPO / "data/processed")); ap.add_argument("--tag", default="native")
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--model", default="")
    ap.add_argument("--features-extra", default="", help="comma-separated: net (network scores), esm (language-model features)")
    ap.add_argument("--out", default=str(REPO / "docs/results"))
    ap.add_argument("--no-graded", action="store_true"); ap.add_argument("--no-zscore", action="store_true")
    ap.add_argument("--restrict-to", default="", help="a manifest whose structures are the only ones used (e.g. the cleaned one)")
    a = ap.parse_args()
    ds = pathlib.Path(a.ds)
    df = tables.read_table(ds, f"candidates_{a.tag}").reset_index(drop=True)
    if a.restrict_to:
        import csv as _csv
        keep = {r["pdb"] for r in _csv.DictReader(open(a.restrict_to))}
        before, before_cl = df["pdb"].nunique(), df["cluster30"].nunique()
        df = df[df["pdb"].isin(keep)].reset_index(drop=True)
        print(f"restricted to {pathlib.Path(a.restrict_to).name}: {df['pdb'].nunique()}/{before} structures, "
              f"{df['cluster30'].nunique()}/{before_cl} clusters")
    geo = json.loads((ds / f"candidates_{a.tag}.geometry.json").read_text())
    feats = list(pf.FEATURES) + ([f for f in pf.PEPTIDE if f in df] if a.tag == "peptide" else [])
    extras = [x for x in a.features_extra.split(",") if x]
    if "net" in extras:
        feats += [f for f in pf.NET_FEATURES if f in df]
    if "esm" in extras:                     # protein-language-model features per candidate, no network needed
        ef = ds / f"esm_features_{a.tag}.csv.gz"
        if not ef.exists():
            sys.exit(f"{ef.name} missing: run scripts/train/build_esm_features.py --tag {a.tag}")
        e = pd.read_csv(ef).drop(columns=["fold"], errors="ignore")
        before = len(df)
        df = df.merge(e, on=["pdb", "center"], how="left")
        esm_cols = [c for c in e.columns if c.startswith("esm_")]
        df[esm_cols] = df[esm_cols].fillna(0.0)
        missing = int(df[esm_cols[0]].eq(0.0).sum()) if esm_cols else 0
        print(f"ESM features: {len(esm_cols)} columns joined to {before} candidates ({missing} without an embedding)")
        feats += esm_cols
        pf.GROUPS = dict(pf.GROUPS, esm=esm_cols)       # so --ablate measures what the language model is worth
    feats = [f for f in dict.fromkeys(feats) if f in df]
    df = add_relevance(df, graded=not a.no_graded)
    base_feats = list(feats)
    if not a.no_zscore:
        df, feats = add_zscores(df, feats)
    print(f"{df['pdb'].nunique()} structures, {df['cluster30'].nunique()} clusters, {len(df)} candidates, "
          f"ceiling {(df.groupby('pdb')['label'].max() == 1).mean():.3f}, {len(feats)} features")

    results, per_method = {}, {}
    baselines = {"native order (detector score)": -df["nat_rank"].to_numpy(float),
                 "largest cavity first": df["cav_volume"].to_numpy(float),
                 "most buried first": df["buried_mean"].to_numpy(float)}
    for name, s in baselines.items():
        per_method[name] = [evaluate(df, s)]
    main_name = f"LightGBM LambdaRank ({len(feats)} features)"
    seed_scores = [cv_scores(df, feats, sd) for sd in range(a.seeds)]
    per_method[main_name] = [evaluate(df, s) for s in seed_scores]
    ens = np.mean(seed_scores, axis=0)
    per_method[f"{main_name}, seed ensemble"] = [evaluate(df, ens)]
    prob, calib = calibrate(df, ens)
    per_method[f"{main_name}, calibrated"] = [evaluate(df, prob)]
    print("calibration: " + json.dumps({k: round(v, 4) for k, v in calib.items()}))
    if a.ablate:
        for g, cols in pf.GROUPS.items():
            drop = set(cols) | {f"{c}_z" for c in cols}
            sub = [f for f in feats if f not in drop]
            if len(sub) < len(feats) and sub:
                per_method[f"  without {g}"] = [evaluate(df, cv_scores(df, sub, sd)) for sd in range(max(2, a.seeds // 2))]
        per_method["  native features only"] = [evaluate(df, cv_scores(df, [f for f in feats if f.startswith("nat_")], sd)) for sd in range(max(2, a.seeds // 2))]
        per_method["  without the within-structure z-scores"] = [evaluate(df, cv_scores(df, base_feats, sd)) for sd in range(max(2, a.seeds // 2))]
        if not a.no_graded:
            binary = df.assign(**{TARGET: df["label"].astype(int)})
            per_method["  binary relevance instead of graded"] = [evaluate(binary, cv_scores(binary, feats, sd)) for sd in range(max(2, a.seeds // 2))]
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
                                                              n_candidates=len(df), seeds=a.seeds, features=feats, n_features=len(feats),
                                                              graded=not a.no_graded, zscore=not a.no_zscore, calibration=calib,
                                                              geometry=geo, results=results), indent=1))
    (out / f"ranker_{a.tag}.md").write_text(
        f"# Ranker results ({a.tag})\n\n{df['pdb'].nunique()} structures, {df['cluster30'].nunique()} clusters, "
        f"{len(df)} candidates, ceiling {(df.groupby('pdb')['label'].max() == 1).mean():.3f}; {len(feats)} features; "
        f"5-fold CV by 30 %-identity cluster, {a.seeds} seeds; 95 % CI by cluster bootstrap; success DCA <= 4 A; "
        f"graded relevance {not a.no_graded}, within-structure z-scores {not a.no_zscore}.\n\n{table}\n\n"
        f"Calibration of the seed-ensemble score (isotonic, fitted out of fold): "
        f"ECE {calib['ece_isotonic']:.4f} after versus {calib['ece_raw_sigmoid']:.4f} for a plain sigmoid of the score; "
        f"Brier {calib['brier_isotonic']:.4f} versus {calib['brier_raw_sigmoid']:.4f}; "
        f"base rate {calib['base_rate']:.3f}, mean predicted {calib['mean_predicted']:.3f}.\n")
    if a.model:
        pathlib.Path(a.model).parent.mkdir(parents=True, exist_ok=True)
        fit(df, feats, 0).save_model(a.model)
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ok = np.isfinite(ens)
        iso.fit(ens[ok], df.loc[ok, "label"].to_numpy(float))
        json.dump(dict(x=iso.X_thresholds_.tolist(), y=iso.y_thresholds_.tolist(), report=calib),
                  open(a.model + ".calibration.json", "w"))
        json.dump(feats, open(a.model + ".features.json", "w")); json.dump(pf.geometry(), open(a.model + ".geometry.json", "w"))
        json.dump(dict(graded=not a.no_graded, zscore=not a.no_zscore, base_features=base_feats), open(a.model + ".preprocess.json", "w"))
        print(f"model saved to {a.model} ({len(feats)} features)")


if __name__ == "__main__":
    main()
