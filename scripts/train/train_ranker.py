#!/usr/bin/env python
"""Step 3: learn to rank candidates (LightGBM LambdaRank) with an honest protocol.

Protocol: 5-fold cross-validation by 30 %-identity cluster (manifest `fold`), several seeds whose scores are averaged
(an ensemble, which is what one would ship), baselines on exactly the same structures, 95 % bootstrap CI over
clusters, and a paired cluster bootstrap of every gain against the detector order. Success: DCA <= 4 A.

Seven things beyond a plain LambdaRank fit, each switchable so its contribution can be measured:
  --graded        relevance 2 for DCA <= 2 A, 1 for <= 4 A, 0 otherwise, instead of a binary label. Ranking a centre
                  that sits on the ligand above one that merely touches it is what top-1 actually rewards.
  --zscore        every feature is also given as a z-score within its own structure. Pocket ranking is a comparison
                  inside one protein, so "deeper than the other cavities of this protein" is the informative form.
  --ensemble      the final score is the mean over seeds (reported next to the per-seed mean).
  --margins       besides the z-score of a feature, its margin: the value minus the best value among the *other*
                  candidates of the same structure. A z-score says "unusual here", a margin says "better than the
                  best alternative", which is what top-1 rewards; they are not the same number.
  --objectives    one model per objective and their rank-average: LambdaRank (ordering), binary classification
                  (is this candidate a site) and regression on -DCA (how close it is). The three disagree in
                  different places, so averaging their within-structure ranks is usually better than any one.
  --stack         a logistic regression over the out-of-fold predictions of those models, fitted per fold on the
                  other folds, i.e. real stacking rather than a fixed average.
  --search N      nested random search over N hyperparameter draws, selected on *inner* folds of the training part
                  only, so the outer estimate stays honest.
  --set-ranker    a permutation-equivariant set transformer (`equicave.set_ranker`) that scores all candidates of a
                  structure jointly with a listwise loss. Gradient boosting scores each candidate alone and has to be
                  told about the competition through hand-built z-scores and margins; the transformer learns the
                  comparison, and it can consume raw embeddings that a tree model cannot use.
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


def add_margins(df: pd.DataFrame, feats: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Each feature minus the best value among the other candidates of the same structure.

    For a structure with values x_1..x_n the margin of candidate i is x_i - max_{j != i} x_j, computed from the two
    largest values per group so it costs one sort. A positive margin means "no other candidate here is better on
    this feature", which is exactly the comparison top-1 scores.
    """
    out = {}
    g = df.groupby("pdb", sort=False)
    for c in feats:
        top2 = g[c].transform(lambda v: v.nlargest(2).iloc[-1] if len(v) > 1 else v.iloc[0])
        mx = g[c].transform("max")
        best_other = np.where(df[c].to_numpy() >= mx.to_numpy(), top2.to_numpy(), mx.to_numpy())
        out[f"{c}_m"] = df[c].to_numpy() - best_other
    m = pd.DataFrame(out, index=df.index).fillna(0.0)
    return pd.concat([df, m], axis=1), feats + list(m.columns)


def fit(train: pd.DataFrame, feats, seed: int, rounds: int = 400, objective: str = "lambdarank", params: dict | None = None):
    """One model. `objective`: lambdarank (ordering), binary (is a site), regression (on -DCA)."""
    train = train.sort_values("pdb", kind="stable")
    base = dict(PARAMS, seed=seed, **(params or {}))
    if objective == "lambdarank":
        y = train[TARGET] if TARGET in train else train["label"]
        ds = lgb.Dataset(train[feats], y, group=train.groupby("pdb", sort=False).size().to_numpy())
    elif objective == "binary":
        base = {k: v for k, v in base.items() if k not in ("metric", "eval_at", "lambdarank_truncation_level", "label_gain")}
        base.update(objective="binary", metric="auc", is_unbalance=True)
        ds = lgb.Dataset(train[feats], train["label"])
    elif objective == "regression":
        base = {k: v for k, v in base.items() if k not in ("metric", "eval_at", "lambdarank_truncation_level", "label_gain")}
        base.update(objective="regression", metric="l2")
        ds = lgb.Dataset(train[feats], -train["dca"].clip(upper=20.0))
    else:
        raise ValueError(objective)
    return lgb.train(base, ds, num_boost_round=rounds)


def cv_scores(df: pd.DataFrame, feats, seed: int, objective: str = "lambdarank", params: dict | None = None,
              rounds: int = 400) -> np.ndarray:
    s = np.full(len(df), np.nan)
    for k in sorted(df["fold"].unique()):
        tr = df["fold"] != k
        s[~tr.to_numpy()] = fit(df[tr], feats, seed, rounds, objective, params).predict(df.loc[~tr, feats])
    return s


def cascade_scores(df: pd.DataFrame, feats, first: np.ndarray, k: int, seed: int, rounds: int = 300) -> np.ndarray:
    """A second stage that only ever sees the top `k` candidates of the first stage, and reorders those.

    Measured on the COACH420 structures that are not similar to our training set: when the first-ranked candidate is
    wrong, the right answer is the second-ranked one in 24 of 53 cases and within the first five in 41 of 53, while
    96 % of those structures do have a correct candidate somewhere. The mistake is therefore a choice between a few
    plausible pockets, not a failure to find the pocket -- and a model trained on all thirty candidates spends most
    of its capacity separating obvious non-sites from sites, which is not that choice.

    This stage is trained only on the restricted lists, where typically exactly one candidate is correct, so the
    gradient is entirely about the discrimination that decides top-1. It also sees what the first stage thought: the
    first-stage score, its rank in the structure and its margin to the best candidate.

    Leakage: `first` must already be out of fold (every row scored by a model that did not see it), and this stage is
    cross-validated on the same folds, so no candidate is reordered by a model that saw its structure. Returns a
    score for every row -- the second stage's score for the candidates it reordered, shifted above the rest, and the
    first stage's own ordering below them.
    """
    d = df.assign(_first=first)
    r = d.groupby("pdb")["_first"].rank(ascending=False, method="first")
    top = (r <= k).to_numpy()
    best = d.groupby("pdb")["_first"].transform("max")
    d["_first_rank"] = r / r.groupby(d["pdb"]).transform("max")
    d["_first_margin"] = d["_first"] - best
    sub = d[top].reset_index(drop=True)
    sfeats = list(feats) + ["_first", "_first_rank", "_first_margin"]
    s2 = cv_scores(sub, sfeats, seed, "lambdarank", rounds=rounds)
    # the reordered candidates keep their places at the head of the list; everything below keeps the first ordering
    out = np.empty(len(d))
    lo, hi = np.nanmin(s2), np.nanmax(s2)
    span = (hi - lo) or 1.0
    out[top] = 1.0 + (s2 - lo) / span
    rest = d.loc[~top, "_first"].to_numpy(float)
    if len(rest):
        rlo, rhi = rest.min(), rest.max()
        out[~top] = (rest - rlo) / ((rhi - rlo) or 1.0)       # strictly below every reordered candidate
    return out


def within_structure_rank(df: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    """Scores replaced by their rank inside each structure, scaled to [0, 1]: comparable across models."""
    s = pd.Series(score, index=df.index)
    r = s.groupby(df["pdb"]).rank(pct=True, method="average")
    return r.to_numpy()


def random_params(rng) -> dict:
    return dict(learning_rate=float(rng.choice([0.03, 0.05, 0.08, 0.12])),
                num_leaves=int(rng.choice([15, 31, 63, 127])),
                min_data_in_leaf=int(rng.choice([5, 10, 20, 40])),
                feature_fraction=float(rng.choice([0.5, 0.65, 0.8, 1.0])),
                bagging_fraction=float(rng.choice([0.6, 0.8, 1.0])),
                lambda_l2=float(rng.choice([0.5, 1.0, 5.0, 20.0])),
                lambdarank_truncation_level=int(rng.choice([3, 5, 10, 20])))


def nested_search(df: pd.DataFrame, feats, n_draws: int, seed: int = 0, rounds: int = 400):
    """Random search selected on inner folds of the training part only; returns out-of-fold scores and the picks.

    For each outer fold the training part is split by its own cluster folds, every draw is scored by top-1 on those
    inner folds, and only the winner is refitted on the whole training part to score the outer fold. The outer
    estimate therefore never sees a hyperparameter chosen with its own data.
    """
    rng = np.random.default_rng(seed)
    draws = [random_params(rng) for _ in range(n_draws)]
    out = np.full(len(df), np.nan)
    picks = []
    for k in sorted(df["fold"].unique()):
        te = (df["fold"] == k).to_numpy()
        tr = df[~te]
        inner = sorted(tr["fold"].unique())
        best, best_score = None, -1.0
        for p in draws:
            sc = np.full(len(tr), np.nan)
            for j in inner[:3]:                        # three inner folds keep the search affordable
                itr = (tr["fold"] != j).to_numpy()
                sc[~itr] = fit(tr[itr], feats, seed, rounds, "lambdarank", p).predict(tr.loc[~itr, feats])
            m = M.per_structure(tr.assign(_s=sc).dropna(subset=["_s"]), "_s")
            v = float(m["top1"].mean())
            if v > best_score:
                best, best_score = p, v
        picks.append(dict(fold=int(k), top1_inner=round(best_score, 4), **best))
        out[te] = fit(tr, feats, seed, rounds, "lambdarank", best).predict(df.loc[te, feats])
    return out, picks


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
    ap.add_argument("--features-extra", default="", help="comma-separated: net (network scores), "
                    "points (per-point ligandability aggregates), esm (language-model features)")
    ap.add_argument("--out", default=str(REPO / "docs/results"))
    ap.add_argument("--no-graded", action="store_true"); ap.add_argument("--no-zscore", action="store_true")
    ap.add_argument("--restrict-to", default="", help="a manifest whose structures are the only ones used (e.g. the cleaned one)")
    ap.add_argument("--margins", action="store_true", help="add per-structure margin features (value minus the best other)")
    ap.add_argument("--points-tag", default="", help="which point_features_<tag> table to join; defaults to --tag. "
                    "The detector is deterministic, so a point table built for one candidate tag joins any other "
                    "table of the same manifest on (pdb, center)")
    ap.add_argument("--cascade", default="", help="comma-separated k: second-stage re-rankers over the top k "
                    "candidates of the first stage, e.g. 3,5")
    ap.add_argument("--objectives", action="store_true", help="also fit binary and regression objectives and rank-average")
    ap.add_argument("--set-ranker", action="store_true", help="also fit the permutation-equivariant set transformer")
    ap.add_argument("--set-dim", type=int, default=96); ap.add_argument("--set-layers", type=int, default=2)
    ap.add_argument("--set-epochs", type=int, default=40)
    ap.add_argument("--stack", action="store_true", help="logistic regression over the model ranks (real stacking)")
    ap.add_argument("--search", type=int, default=0, help="nested random hyperparameter search of N draws")
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
    if "points" in extras:                  # aggregates of the learned per-point ligandability score
        pfile = ds / f"point_features_{a.points_tag or a.tag}.csv.gz"
        if not pfile.exists():
            sys.exit(f"{pfile.name} missing: run scripts/train/build_points.py then train_point_model.py --tag {a.tag}")
        pt = pd.read_csv(pfile).drop(columns=["fold"], errors="ignore")
        before = len(df)
        df = df.merge(pt, on=["pdb", "center"], how="left")
        assert len(df) == before, "point features duplicated a candidate row"
        cols = [c for c in pt.columns if c in pf.POINT_AGG]
        missing = int(df[cols[0]].isna().sum()) if cols else 0
        df[cols] = df[cols].fillna(0.0)
        print(f"point features: {len(cols)} columns joined to {before} candidates ({missing} without a point score)")
        feats += cols

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
    if a.margins:
        base_only = [f for f in base_feats if f in df]
        df, feats = add_margins(df, base_only)
        feats = list(dict.fromkeys(feats + [f for f in df.columns if f.endswith("_z")]))
        print(f"margins added: {len(feats)} features")
    main_name = f"LightGBM LambdaRank ({len(feats)} features)"
    seed_scores = [cv_scores(df, feats, sd) for sd in range(a.seeds)]
    per_method[main_name] = [evaluate(df, s) for s in seed_scores]
    ens = np.mean(seed_scores, axis=0)
    per_method[f"{main_name}, seed ensemble"] = [evaluate(df, ens)]
    prob, calib = calibrate(df, ens)
    per_method[f"{main_name}, calibrated"] = [evaluate(df, prob)]
    print("calibration: " + json.dumps({k: round(v, 4) for k, v in calib.items()}))

    extra_scores = {}
    for k in [int(x) for x in a.cascade.split(",") if x.strip()]:
        sc = np.mean([cascade_scores(df, feats, ens, k, sd) for sd in range(max(1, a.seeds // 2))], axis=0)
        extra_scores[f"cascade{k}"] = sc
        per_method[f"cascade re-ranker over the top {k}"] = [evaluate(df, sc)]
    if a.objectives:
        for obj in ("binary", "regression"):
            sc = np.mean([cv_scores(df, feats, sd, obj) for sd in range(max(1, a.seeds // 2))], axis=0)
            extra_scores[obj] = sc
            per_method[f"LightGBM {obj}"] = [evaluate(df, sc)]
        ranks = [within_structure_rank(df, s_) for s_ in [ens] + list(extra_scores.values())]
        per_method["rank average of the three objectives"] = [evaluate(df, np.mean(ranks, axis=0))]
    if a.set_ranker:
        from equicave.set_ranker import SetRanker
        sc = np.full(len(df), np.nan)
        for k in sorted(df["fold"].unique()):
            te = (df["fold"] == k).to_numpy(); tr = ~te
            m = SetRanker(n_features=len(feats), dim=a.set_dim, layers=a.set_layers, epochs=a.set_epochs,
                          seed=0).fit(df.loc[tr, feats].to_numpy(float), df.loc[tr, "pdb"].to_numpy(),
                                      df.loc[tr, TARGET].to_numpy(float), df.loc[tr, "label"].to_numpy(float),
                                      log=print)
            sc[te] = m.predict(df.loc[te, feats].to_numpy(float), df.loc[te, "pdb"].to_numpy())
        extra_scores["set"] = sc
        per_method[f"set transformer (listwise, {a.set_layers} layers)"] = [evaluate(df, sc)]
        per_method["set transformer + LambdaRank, rank average"] = [
            evaluate(df, (within_structure_rank(df, sc) + within_structure_rank(df, ens)) / 2)]
    if a.stack and extra_scores:
        from sklearn.linear_model import LogisticRegression
        cols = np.column_stack([within_structure_rank(df, s_) for s_ in [ens] + list(extra_scores.values())])
        st_sc = np.full(len(df), np.nan)
        for k in sorted(df["fold"].unique()):
            te = (df["fold"] == k).to_numpy(); tr = ~te & np.isfinite(cols).all(1)
            lr = LogisticRegression(max_iter=2000).fit(cols[tr], df.loc[tr, "label"])
            st_sc[te] = lr.predict_proba(cols[te])[:, 1]
        per_method["stacked (logistic regression over the model ranks)"] = [evaluate(df, st_sc)]
    if a.search:
        sc, picks = nested_search(df, feats, a.search)
        per_method[f"LambdaRank, nested search over {a.search} draws"] = [evaluate(df, sc)]
        print("hyperparameters chosen per fold: " + json.dumps(picks))
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
