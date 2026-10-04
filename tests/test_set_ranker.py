"""The set ranker must be permutation-equivariant and must learn a comparison a per-candidate model cannot."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from equicave.set_ranker import SetRanker  # noqa: E402


def _toy(n_struct=60, n_cand=8, seed=0):
    """In each structure the winner is the candidate with the largest feature 0 *relative to its own structure*,
    and the absolute scale of feature 0 varies per structure, so only a within-set comparison can find it."""
    rng = np.random.default_rng(seed)
    X, groups, rel, lab = [], [], [], []
    for s in range(n_struct):
        scale = rng.uniform(0.5, 50.0)
        f0 = rng.uniform(0, 1, n_cand) * scale
        noise = rng.normal(size=(n_cand, 3))
        X.append(np.column_stack([f0, noise]))
        groups += [s] * n_cand
        win = int(np.argmax(f0))
        r = np.zeros(n_cand); r[win] = 2.0
        rel.append(r); lab.append((r > 0).astype(float))
    return np.vstack(X), np.array(groups), np.concatenate(rel), np.concatenate(lab)


def test_permutation_equivariance():
    X, g, rel, lab = _toy(n_struct=4, n_cand=6)
    m = SetRanker(n_features=X.shape[1], dim=32, layers=2, heads=4, epochs=1)
    m._standardise(X, fit=True)
    s1 = m.predict(X, g)
    perm = np.concatenate([np.random.default_rng(k).permutation(6) + 6 * k for k in range(4)])
    s2 = m.predict(X[perm], g[perm])
    assert np.allclose(s1[perm], s2, atol=1e-5), "scores must follow the permutation of the candidates"


def test_learns_a_within_set_comparison():
    X, g, rel, lab = _toy(n_struct=120, n_cand=8, seed=1)
    m = SetRanker(n_features=X.shape[1], dim=48, layers=2, heads=4, epochs=40, batch=8, seed=0)
    m.fit(X, g, rel, lab)
    s = m.predict(X, g)
    top1 = np.mean([rel[g == k][int(np.argmax(s[g == k]))] > 0 for k in np.unique(g)])
    assert top1 > 0.7, f"top-1 on a learnable comparison is only {top1:.2f}"
    assert m.n_parameters() < 200_000


def test_handles_a_structure_with_no_positive():
    X, g, rel, lab = _toy(n_struct=20, n_cand=5, seed=2)
    rel[g == 0] = 0.0; lab[g == 0] = 0.0           # one structure where nothing hits
    m = SetRanker(n_features=X.shape[1], dim=32, layers=1, heads=4, epochs=5)
    m.fit(X, g, rel, lab)
    s = m.predict(X, g)
    assert np.isfinite(s).all()
