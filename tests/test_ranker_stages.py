"""The ranker's cross-validation discipline and the cascade stage.

These functions decide every published ranking number, and a leak in them inflates all of them silently, so the
properties are asserted rather than assumed: a fold's rows are only ever scored by a model that did not see them,
and the cascade keeps the candidates it reorders above the ones it does not.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts/train"))
import train_ranker as TR  # noqa: E402


def toy(n_structures=40, n_cands=6, seed=0):
    """Structures whose correct candidate is the one with the largest `good` feature, plus pure noise columns."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_structures):
        best = rng.integers(n_cands)
        for j in range(n_cands):
            rows.append(dict(pdb=f"P{i:03d}", cluster30=f"C{i // 2:03d}", fold=i % 5,
                             good=(2.0 if j == best else 0.0) + rng.normal(0, 0.3),
                             noise=rng.normal(), nat_rank=j + 1,
                             label=int(j == best), n_sites=1))
    return pd.DataFrame(rows)


def test_cv_scores_never_lets_a_model_see_the_rows_it_scores(monkeypatch):
    """Every row's score must come from a fit whose training rows excluded that row's fold."""
    df = toy()
    seen = []

    real_fit = TR.fit

    def spy(train, feats, seed, rounds=400, objective="lambdarank", params=None):
        seen.append(set(train["fold"].unique()))
        return real_fit(train, feats, seed, rounds, objective, params)

    monkeypatch.setattr(TR, "fit", spy)
    s = TR.cv_scores(df, ["good", "noise"], seed=0, rounds=30)
    assert not np.isnan(s).any()
    folds = sorted(df["fold"].unique())
    assert len(seen) == len(folds)
    for k, train_folds in zip(folds, seen):
        assert k not in train_folds                     # the held-out fold was never in the training rows


def top1(df, score) -> float:
    """`train_ranker.evaluate` returns one row per structure; the reported number is the mean of a column."""
    return float(TR.evaluate(df, score)["top1"].mean())


def test_cv_scores_learns_the_signal_and_not_the_noise():
    df = toy()
    good = TR.cv_scores(df, ["good", "noise"], seed=0, rounds=60)
    noise_only = TR.cv_scores(df, ["noise"], seed=0, rounds=60)
    assert top1(df, good) > 0.6 > top1(df, noise_only)


def test_cascade_reorders_only_the_top_k_and_keeps_them_above_the_rest():
    df = toy()
    first = df["good"].to_numpy(float) * 0.0 + np.arange(len(df)) % 7      # an arbitrary first-stage ordering
    k = 3
    out = TR.cascade_scores(df, ["good", "noise"], first, k, seed=0, rounds=30)
    assert len(out) == len(df) and np.isfinite(out).all()
    d = df.assign(_f=first)
    rank = d.groupby("pdb")["_f"].rank(ascending=False, method="first").to_numpy()
    top = rank <= k
    assert top.sum() == df["pdb"].nunique() * k
    assert out[top].min() >= out[~top].max()            # every reordered candidate outranks every untouched one


def test_cascade_cannot_lower_the_ceiling_or_top_k_recall():
    """Reordering within the top k must not lose a hit that was already inside the top k."""
    df = toy(seed=3)
    first = TR.cv_scores(df, ["good", "noise"], seed=0, rounds=60)
    out = TR.cascade_scores(df, ["good", "noise"], first, 3, seed=0, rounds=30)
    a, b = TR.evaluate(df, first), TR.evaluate(df, out)
    assert b["ceiling"].mean() == pytest.approx(a["ceiling"].mean())
    assert b["top3"].mean() == pytest.approx(a["top3"].mean(), abs=1e-9)


# --- the learner is a choice, and the default is the one the published numbers used -------------------------------

def test_learner_default_and_rejection():
    """LightGBM stays the default: every number in docs/results is its, and an arm needs a reference."""
    assert TR.LEARNER == "lgbm" and TR.LEARNERS == ("lgbm", "catboost")
    df = toy(8, 4)
    feats = [c for c in df.columns if c.startswith(("good", "noise"))]
    with pytest.raises(ValueError, match="learner must be one of"):
        TR.fit(df, feats, 0, rounds=5, learner="xgboost")


def test_learner_flag_is_exposed():
    src = (pathlib.Path(__file__).resolve().parents[1] / "scripts/train/train_ranker.py").read_text()
    assert '"--learner"' in src and "choices=list(LEARNERS)" in src
    assert "LEARNER = a.learner" in src                      # the flag actually reaches the module-level default


def test_catboost_ranker_fits_and_orders():
    """Skipped where catboost is absent. The CatBoost path was written but never executed in that case."""
    pytest.importorskip("catboost")
    df = toy(30, 5)
    feats = [c for c in df.columns if c.startswith(("good", "noise"))]
    m = TR.fit(df, feats, 0, rounds=30, learner="catboost")
    s = m.predict(df[feats])
    assert len(s) == len(df) and np.isfinite(s).all()
    # the correct candidate of each structure carries the largest `good`, so a fitted ranker must prefer it
    hit = [int(np.argmax(g["pred"].to_numpy()) == int(np.argmax(g["good"].to_numpy())))
           for _, g in df.assign(pred=s).groupby("pdb", sort=False)]
    assert np.mean(hit) > 0.6
