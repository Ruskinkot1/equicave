import numpy as np
from scipy.spatial import cKDTree

from equicave import detect, pockets as pk
from tests.conftest import ball_with_pocket, shell, solid_ball


def test_buriedness_grid_matches_exact_rays():
    xyz = ball_with_pocket()
    lo, shape, dist = detect.lattice(xyz)
    occ = dist < pk.RAY_HIT
    bg = detect.buriedness_grid(occ)
    free = dist >= pk.CLASH
    idx = np.argwhere(free)
    rng = np.random.default_rng(0)
    sel = idx[rng.choice(len(idx), 300, replace=False)]
    pts = lo + sel * pk.GRID
    exact = pk._buried(pts, cKDTree(xyz))
    grid = bg[tuple(sel.T)]
    assert np.corrcoef(exact, grid)[0, 1] > 0.95
    assert np.abs(exact - grid).mean() < 2.0


def test_pocket_is_found_at_the_carved_cavity():
    xyz = ball_with_pocket(pocket_center=(0, 0, 6.0))
    cands = detect.detect_sites(xyz)
    assert cands, "no candidate"
    assert cands[0]["rank"] == 1 and len(cands) <= detect.MAX_SITES
    assert np.linalg.norm(cands[0]["center"] - np.array([0, 0, 6.0])) < 4.0
    assert cands[0]["score"] >= cands[-1]["score"]


def test_solid_ball_has_no_deep_candidate():
    cands = detect.detect_sites(solid_ball())
    assert all(c["n_points"] < 40 for c in cands)   # at most tiny surface dimples


def test_tube_cavity_is_split_by_nms_into_several_sites():
    xyz = shell(radius=7.0, zs=tuple(range(-14, 15, 3)), n_ring=30)
    cands = detect.detect_sites(xyz, nms=6.0)
    assert len(cands) >= 2
    centres = np.array([c["center"] for c in cands])
    d = np.linalg.norm(centres[:, None] - centres[None], axis=2) + np.eye(len(centres)) * 99
    assert d.min() > 3.0


def test_equivariance_of_candidates():
    xyz = ball_with_pocket()
    rng = np.random.default_rng(1)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    t = np.array([30.0, -12.0, 7.0])
    a = detect.detect_sites(xyz)[0]
    b = detect.detect_sites(xyz @ q.T + t)[0]
    # the lattice is axis aligned, so only approximate equivariance is expected (within one lattice step)
    assert np.linalg.norm((a["center"] @ q.T + t) - b["center"]) < 2.0


def test_candidate_table_features():
    rows = detect.candidate_table(detect.detect_sites(ball_with_pocket()))
    assert rows[0]["nat_rank"] == 1 and abs(rows[0]["nat_rel"] - 1.0) < 1e-9
    assert set(rows[0]) >= {"nat_score", "nat_rank", "nat_rel", "nat_npts", "nat_mean_bur", "n_cands", "center"}


def test_empty_and_tiny_inputs():
    assert detect.detect_sites(np.zeros((2, 3))) == []
    cands, field = detect.detect_sites(ball_with_pocket(), return_field=True)
    assert field["buried"].shape == field["free"].shape
