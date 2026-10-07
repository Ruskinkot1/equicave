"""Pocket geometry without external tools: free-space grid, 26-ray buriedness, cavity masks, SAS points, shape.

Everything here depends only on protein heavy-atom coordinates. The constants are shared by candidate detection
(`detect.py`), the ranker features (`pocket_features.py`) and the network's probe placement (`training/`), so
training and inference see the same geometry; `geometry()` records them next to every trained model.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

GRID = 1.0            # A: lattice step for every grid in the project
CLASH = 3.0           # A: a grid point closer than this to a protein heavy atom is not free space
RAY_LEN = 8.0         # A: a ray is "blocked" if it meets protein within this length ...
RAY_HIT = 2.0         # ... i.e. a point on the ray lies within RAY_HIT A of a heavy atom
RAY_STEP = 1.0        # A: sampling step along each ray
MIN_BURIED = 8        # of 26 rays: minimum closure for a point to belong to a cavity mask (cavity_box)
MARGIN = 1.5          # A: docking box margin around the cavity
MIN_SIDE = 10.0       # A: minimum docking box side

_DIRS = np.array([[x, y, z] for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if (x, y, z) != (0, 0, 0)], float)
_DIRS /= np.linalg.norm(_DIRS, axis=1, keepdims=True)          # the 26 lattice directions
N_RAYS = len(_DIRS)


def geometry() -> dict:
    """Constants that define cavities; stored with trained models and checked at inference."""
    return dict(GRID=GRID, CLASH=CLASH, RAY_LEN=RAY_LEN, RAY_HIT=RAY_HIT, RAY_STEP=RAY_STEP, MIN_BURIED=MIN_BURIED)


def _buried(points: np.ndarray, tree: cKDTree, ray_len: float = RAY_LEN, ray_hit: float = RAY_HIT,
            step: float = RAY_STEP, workers: int = 1) -> np.ndarray:
    """Number of the 26 rays from each point that meet a protein atom within `ray_len` A (LIGSITE-style closure)."""
    points = np.atleast_2d(points)
    n = np.zeros(len(points), int)
    if len(points) == 0:
        return n
    ks = np.arange(step, ray_len + 1e-6, step)
    for d in _DIRS:
        hit = np.zeros(len(points), bool)
        for k in ks:
            idx = ~hit
            if not idx.any():
                break
            hit[idx] = tree.query(points[idx] + k * d, distance_upper_bound=ray_hit, workers=workers)[0] < ray_hit
        n += hit
    return n


def free_grid(xyz: np.ndarray, step: float = GRID, clash: float = CLASH, margin: float = 4.0, tree=None):
    """Lattice points around the protein that are at least `clash` A from every heavy atom.

    Returns (points [M,3], origin, shape, index [M,3]). The lattice is absolute (multiples of `step`), so the same
    protein always gives the same points regardless of the caller.
    """
    tree = tree or cKDTree(xyz)
    lo = np.floor((xyz.min(0) - margin) / step) * step
    hi = xyz.max(0) + margin
    axes = [np.arange(l, h + step, step) for l, h in zip(lo, hi)]
    shape = tuple(len(a) for a in axes)
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    d, _ = tree.query(grid, distance_upper_bound=clash)
    keep = ~(d < clash)
    idx = np.stack(np.unravel_index(np.where(keep)[0], shape), 1)
    return grid[keep], lo, shape, idx


def cavity_box(seeds: np.ndarray, xyz: np.ndarray, radius: float, step: float = GRID, min_buried: int = MIN_BURIED,
               clash: float = CLASH):
    """Irregular cavity mask around `seeds` plus the axis-aligned box a docking program needs.

    Grid points within `radius` A of a seed, free of clashes, buried (>= `min_buried` of 26 rays) and connected to a
    seed form the cavity. Returns (box centre, box sides, cavity points).
    """
    seeds = np.atleast_2d(np.asarray(seeds, float))
    lo = np.floor((seeds.min(0) - radius - step) / step) * step
    hi = seeds.max(0) + radius + step
    axes = [np.arange(l, h + step, step) for l, h in zip(lo, hi)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1)
    flat = grid.reshape(-1, 3)
    tp = cKDTree(xyz)
    cand = (cKDTree(seeds).query(flat)[0] <= radius) & (tp.query(flat)[0] >= clash)
    ok = np.zeros(len(flat), bool)
    idx = np.where(cand)[0]
    if len(idx):
        ok[idx] = _buried(flat[idx], tp) >= min_buried
    free = ok.reshape(grid.shape[:3])
    start = np.zeros_like(free)
    for sd in seeds:
        i = tuple(np.clip(np.round((sd - lo) / step).astype(int), 0, np.array(free.shape) - 1))
        free[i] = True; start[i] = True
    lab, _ = ndimage.label(free)
    keep = np.isin(lab, np.unique(lab[start]))
    pts = np.vstack([grid[keep], seeds])
    pmin, pmax = pts.min(0), pts.max(0)
    return (pmin + pmax) / 2, np.maximum(pmax - pmin + 2 * MARGIN, MIN_SIDE), pts


def shape_axes(points: np.ndarray) -> list[float]:
    """Principal extents (A, longest first) of a point set."""
    c = np.atleast_2d(points) - np.atleast_2d(points).mean(0)
    if len(c) < 4:
        return [0.0, 0.0, 0.0]
    _, vec = np.linalg.eigh(np.cov(c.T))
    proj = c @ vec
    return sorted((float(x) for x in (proj.max(0) - proj.min(0))), reverse=True)


def sas_points(xyz: np.ndarray, radii: np.ndarray | None = None, probe: float = 1.4, n_sphere: int = 60,
               seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Solvent-accessible-surface point cloud (Shrake-Rupley sampling) and the index of the atom each point sits on.

    Points on a sphere of radius (vdW + probe) around each atom that are not inside any neighbour's sphere.
    Used by the surface module of the network; a point cloud, not a mesh.
    """
    xyz = np.asarray(xyz, float)
    radii = np.full(len(xyz), 1.7) if radii is None else np.asarray(radii, float)
    # Fibonacci sphere: even, deterministic
    i = np.arange(n_sphere) + 0.5
    phi = np.arccos(1 - 2 * i / n_sphere); theta = np.pi * (1 + 5 ** 0.5) * i
    unit = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)
    R = radii + probe
    tree = cKDTree(xyz)
    pts, owner = [], []
    maxr = R.max()
    for a in range(len(xyz)):
        sph = xyz[a] + R[a] * unit
        nb = tree.query_ball_point(xyz[a], R[a] + maxr)
        nb = [b for b in nb if b != a]
        if nb:
            d = np.linalg.norm(sph[:, None, :] - xyz[nb][None], axis=2)
            exposed = (d >= R[nb][None]).all(1)
        else:
            exposed = np.ones(n_sphere, bool)
        if exposed.any():
            pts.append(sph[exposed]); owner.append(np.full(int(exposed.sum()), a))
    if not pts:
        return np.zeros((0, 3)), np.zeros(0, int)
    return np.vstack(pts), np.concatenate(owner)


VDW = {"C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8, "SE": 1.9, "P": 1.8}


def vdw_radii(element: np.ndarray) -> np.ndarray:
    return np.array([VDW.get(str(e), 1.7) for e in element])


def dca(center, lig_xyz: np.ndarray) -> float:
    """Distance from a predicted centre to the nearest ligand heavy atom (success: <= 4 A)."""
    return float(np.linalg.norm(np.asarray(lig_xyz) - np.asarray(center), axis=1).min())


def dcc(center, lig_xyz: np.ndarray) -> float:
    """Distance from a predicted centre to the ligand centroid."""
    return float(np.linalg.norm(np.asarray(lig_xyz).mean(0) - np.asarray(center)))


def _voxels(points: np.ndarray, step: float = GRID) -> set:
    """A point cloud as a set of integer lattice cells, so two clouds can be intersected exactly."""
    if len(points) == 0:
        return set()
    return set(map(tuple, np.rint(np.asarray(points, float) / step).astype(np.int64)))


def site_voxels(lig_xyz: np.ndarray, free: np.ndarray, radius: float = 4.0, step: float = GRID) -> np.ndarray:
    """The true pocket's volume: the free lattice points within `radius` of a ligand heavy atom.

    DVO needs a volume for the answer, and which volume is a protocol choice rather than a given. scPDB ships
    cavity files and the papers that report DVO use those; we do not depend on scPDB, so the volume is defined
    from the ligand and our own lattice instead -- the free points the ligand actually occupies or touches. The
    radius is stated rather than tuned, and both sides of the ratio live on the same lattice, which is what makes
    the intersection exact instead of a nearest-neighbour approximation.
    """
    lig, free = np.asarray(lig_xyz, float), np.asarray(free, float)
    if len(free) == 0 or len(lig) == 0:
        return np.empty((0, 3))
    d = cKDTree(lig).query(free, k=1, distance_upper_bound=radius)[0]
    return free[np.isfinite(d)]


def dvo(pred_points: np.ndarray, true_points: np.ndarray, step: float = GRID) -> float:
    """Discretized volume overlap: the Jaccard index of two pocket volumes on the shared lattice.

    DCC and DCA say whether the prediction is in the right place; this says whether it is the right shape and
    size. A prediction can sit a single angstrom from the ligand centroid and still cover a tenth of the pocket,
    or engulf half the protein -- both score perfectly on distance and badly here. Returns 0.0 when either volume
    is empty, since an empty prediction overlaps nothing.
    """
    a, b = _voxels(pred_points, step), _voxels(true_points, step)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def pli(pred_points: np.ndarray, lig_xyz: np.ndarray, step: float = GRID) -> float:
    """Proportion of ligand inside: the fraction of ligand heavy atoms falling in the predicted volume.

    The asymmetric half of DVO. A pocket that covers the whole ligand but is twice too large scores 1.0 here and
    poorly on DVO; reporting both separates "did it miss part of the ligand" from "is it the wrong size".
    """
    lig = np.asarray(lig_xyz, float)
    if len(lig) == 0 or len(pred_points) == 0:
        return 0.0
    cells = _voxels(pred_points, step)
    hit = [tuple(c) in cells for c in np.rint(lig / step).astype(np.int64)]
    return float(np.mean(hit))
