"""The per-point ligandability features and their candidate aggregates."""
import numpy as np
import pytest
from scipy.spatial import cKDTree

from equicave import detect, labels as LB, pocket_features as pf, point_score as ps
from .conftest import ball_with_pocket


def fake_structure(xyz):
    """A structure dict with the fields point_features reads: carbon atoms, one residue per nine atoms."""
    n = len(xyz)
    return dict(xyz=xyz, element=np.array(["C"] * n), resname=np.array(["ALA"] * n),
                resid=np.array([f"A:{i // 9}" for i in range(n)]), backbone=np.zeros(n, bool),
                bfactor=np.zeros(n), atom=np.array(["CB"] * n), chain=np.array(["A"] * n))


def features_for(xyz, pts):
    st = fake_structure(xyz)
    tp = cKDTree(xyz)
    trees = {k: (cKDTree(xyz[m]) if m.sum() else None) for k, m in LB.protein_atom_types(st).items()}
    return ps.point_features(pts, st, tp, trees)


def test_point_features_are_finite_and_named():
    xyz = ball_with_pocket()
    pts = np.array([[0.0, 0.0, 6.0], [0.0, 0.0, 20.0]])          # inside the pocket, and far outside the protein
    X = features_for(xyz, pts)
    assert X.shape == (2, len(ps.POINT_FEATURES))
    assert np.isfinite(X).all()
    i_bur = ps.POINT_FEATURES.index("p_buried")
    assert X[0, i_bur] > X[1, i_bur]                              # the pocket point is the enclosed one
    i_n8 = ps.POINT_FEATURES.index("p_n8")
    assert X[0, i_n8] > X[1, i_n8] == 0


def test_point_features_are_rotation_invariant():
    """Every per-point feature is a distance, a count or a fraction, so a rotation must not move any of them."""
    xyz = ball_with_pocket()
    pts = np.array([[0.0, 0.0, 6.0], [0.0, 1.0, 5.0], [1.0, 0.0, 6.5]])
    rng = np.random.default_rng(0)
    q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    q = q * np.sign(np.linalg.det(q))
    a = features_for(xyz, pts)
    b = features_for(xyz @ q.T, pts @ q.T)
    assert np.allclose(a, b, atol=1e-5)


def test_labels_mark_only_occupied_points():
    pts = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    lig = np.array([[0.5, 0.0, 0.0]])
    assert list(ps.point_labels(pts, lig, radius=2.0)) == [1, 0]
    assert list(ps.point_labels(pts, np.zeros((0, 3)))) == [0, 0]


def test_aggregate_keys_match_the_declared_group_and_are_the_ranker_group():
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(40, 3)) * 2
    out = ps.aggregate(rng.random(40), pts, pts.mean(0))
    assert sorted(out) == sorted(ps.POINT) == sorted(pf.POINT_AGG) == sorted(pf.GROUPS["points"])
    assert all(np.isfinite(v) for v in out.values())
    empty = ps.aggregate(np.zeros(0), np.zeros((0, 3)), np.zeros(3))
    assert sorted(empty) == sorted(ps.POINT) and set(empty.values()) == {0.0}


def test_the_sum_grows_with_the_number_of_ligandable_points_and_the_blob_is_contiguous():
    """pts_sum is the additive score; pts_blob_max only counts a connected patch, which is the point of having both."""
    near = np.stack(np.meshgrid(*[np.arange(3.0)] * 3, indexing="ij"), -1).reshape(-1, 3)   # 27 adjacent points
    scattered = near * 10.0                                                                 # same count, far apart
    hot = np.ones(len(near))
    a, b = ps.aggregate(hot, near, near.mean(0)), ps.aggregate(hot, scattered, scattered.mean(0))
    assert a["pts_sum"] == b["pts_sum"] == len(near)
    assert a["pts_blob_max"] == len(near) and b["pts_blob_max"] == 1.0
    half = ps.aggregate(np.r_[hot[:14], np.zeros(13)], near, near.mean(0))
    assert half["pts_sum"] < a["pts_sum"] and half["pts_n_50"] == 14


def test_weighted_center_moves_towards_the_high_scoring_points():
    pts = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    c = ps.weighted_center(np.array([0.1, 0.9]), pts)
    assert c[0] == pytest.approx(9.0)
    assert np.allclose(ps.weighted_center(np.zeros(2), pts), pts.mean(0))


def test_aggregates_can_be_produced_for_real_candidates_at_inference():
    """pocket_features.point_aggregates must return one complete row per candidate, in candidate order."""
    xyz = ball_with_pocket()
    cands = detect.detect_sites(xyz, min_buried=8, nms=6.0, max_sites=5, fill_min_buried=6)
    assert cands

    class ConstantModel:                       # stands in for the LightGBM booster
        def predict(self, X):
            return np.full(len(X), 0.75)

    rows = pf.point_aggregates(cands, fake_structure(xyz), ConstantModel())
    assert len(rows) == len(cands)
    assert all(sorted(r) == sorted(pf.POINT_AGG) for r in rows)
    assert all(r["pts_max"] == pytest.approx(0.75) for r in rows if r["pts_n_points"])


def test_the_interaction_columns_are_not_swapped():
    """Column j of every potential block must be POT_SPEC[j]; a swap would mislabel 51 features without any symptom.

    The test asks the chemistry directly: a lone receptor acceptor 3 A away satisfies a *donor* placed at the point
    (the `hbd` channel) and, at 3.5 A, a halogen; it says nothing about a donor being available, so `hba` stays 0.
    """
    assert list(pf.POT_CLASSES) == [n for n, _ in pf.POT_SPEC]
    names = [n for n, _ in pf.POT_SPEC]
    trees = {k: None for k in ("donor", "acceptor", "cation", "anion", "aromatic", "hydrophobic")}
    trees["acceptor"] = cKDTree(np.array([[3.0, 0.0, 0.0]]))
    C, D, A = pf.point_potential(np.zeros((1, 3)), trees)
    assert A[0][names.index("hbd")] == 1 and C[0][names.index("hbd")] == 1
    assert A[0][names.index("halogen")] == 1
    assert A[0][names.index("hba")] == 0 and C[0][names.index("hba")] == 0
    assert all(A[0][names.index(n)] == 0 for n in ("hydrophobic", "aromatic", "cation", "anion"))
    assert D[0][names.index("hbd")] == pytest.approx(3.0)

    # the same ordering must hold in the per-point feature row the model consumes
    xyz = ball_with_pocket()
    X = features_for(xyz, np.array([[0.0, 0.0, 6.0]]))
    for j, name in enumerate(names):
        assert ps.POINT_FEATURES[j] == f"p_n_{name}"
        assert ps.POINT_FEATURES[len(names) + j] == f"p_d_{name}"
        assert ps.POINT_FEATURES[2 * len(names) + j] == f"p_s_{name}"
    assert X.shape[1] == len(ps.POINT_FEATURES)
