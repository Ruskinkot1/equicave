"""Native candidate generator: binding-site candidates from protein geometry alone.

Algorithm (all on the absolute 1 A lattice of `pockets.free_grid`):
1. One nearest-atom distance query for every lattice point gives two masks: *free* (>= CLASH from heavy atoms) and
   *occupied* (< RAY_HIT from a heavy atom, the "wall" a ray can hit).
2. Buriedness of every free point = number of the 26 lattice directions along which the occupied mask is met within
   RAY_LEN A. This is the lattice form of `pockets._buried` (same rays, same length, same hit radius; the ray is sampled
   at lattice points instead of every 1 A along a unit vector).
3. Cavity points = free points with buriedness >= `min_buried`; 26-connected components of cavity points are cavities.
4. Each cavity is split into sub-sites by peaks of the smoothed buriedness field with non-maximum suppression of
   radius `nms` (default 6 A). Every cavity point goes to its nearest peak.
5. A candidate = one peak group with at least `min_points` points. Score = sum of buriedness over its points;
   centre = buriedness-weighted centroid. Candidates are returned best-first, at most `max_sites`.

Only coordinates are used: no sequence, no finder scores, no external program.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from . import pockets as pk

DETECT_MIN_BURIED = 14   # of 26 rays: cavity points for candidate detection (deeper than the cavity-mask MIN_BURIED)
NMS_RADIUS = 6.0         # A: minimum separation between sub-site peaks
MIN_POINTS = 12          # lattice points: smallest candidate that is kept
MAX_SITES = 30
SMOOTH_SIGMA = 1.5       # A: gaussian smoothing of the buriedness field before peak picking

# integer lattice offsets for the 26 directions and the number of steps that stay within RAY_LEN
_OFFS = np.array([[x, y, z] for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if (x, y, z) != (0, 0, 0)], int)


def _shift(a: np.ndarray, off) -> np.ndarray:
    """a shifted by -off so that out[i] = a[i + off]; outside the array = False."""
    out = np.zeros_like(a)
    src = [slice(max(o, 0), a.shape[k] + min(o, 0)) for k, o in enumerate(off)]
    dst = [slice(max(-o, 0), a.shape[k] + min(-o, 0)) for k, o in enumerate(off)]
    out[tuple(dst)] = a[tuple(src)]
    return out


def buriedness_grid(occupied: np.ndarray, step: float = pk.GRID, ray_len: float = pk.RAY_LEN) -> np.ndarray:
    """Per lattice point: number of the 26 directions whose ray meets `occupied` within `ray_len` A."""
    n = np.zeros(occupied.shape, np.int8)
    for off in _OFFS:
        length = np.linalg.norm(off) * step
        hit = np.zeros_like(occupied)
        for k in range(1, int(ray_len / length + 1e-9) + 1):
            hit |= _shift(occupied, off * k)
        n += hit
    return n


def lattice(xyz: np.ndarray, step: float = pk.GRID, margin: float = 4.0):
    """Absolute lattice around the protein: origin, shape, nearest-heavy-atom distance for every lattice point."""
    lo = np.floor((xyz.min(0) - margin) / step) * step
    hi = xyz.max(0) + margin
    axes = [np.arange(l, h + step, step) for l, h in zip(lo, hi)]
    shape = tuple(len(a) for a in axes)
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    d, _ = cKDTree(xyz).query(grid, workers=-1)
    return lo, shape, d.reshape(shape)


def detect_sites(xyz: np.ndarray, step: float = pk.GRID, clash: float = pk.CLASH, min_buried: int = DETECT_MIN_BURIED,
                 nms: float = NMS_RADIUS, min_points: int = MIN_POINTS, max_sites: int = MAX_SITES,
                 return_field: bool = False) -> list[dict] | tuple[list[dict], dict]:
    """Candidate binding sites of one protein (heavy-atom coordinates only). Best-first, at most `max_sites`.

    Each candidate: dict(center [3], peak [3], points [n,3], buried [n] (per point), score, n_points, mean_buried,
    max_buried, cavity (id of the parent connected cavity), cavity_points (size of that cavity), rank).
    With `return_field=True` the lattice (origin, shape, free mask, buriedness) is returned too.
    """
    xyz = np.asarray(xyz, float)
    if len(xyz) < 4:
        return ([], {}) if return_field else []
    lo, shape, dist = lattice(xyz, step)
    free = dist >= clash
    occupied = dist < pk.RAY_HIT
    bur = buriedness_grid(occupied, step)
    cav = free & (bur >= min_buried)
    lab, n_lab = ndimage.label(cav, structure=np.ones((3, 3, 3)))
    field = dict(origin=lo, shape=shape, free=free, buried=bur, dist=dist, labels=lab)
    if n_lab == 0:
        return ([], field) if return_field else []
    smooth = ndimage.gaussian_filter((bur * cav).astype(float), SMOOTH_SIGMA / step)
    cands = []
    sizes = ndimage.sum(cav, lab, index=np.arange(1, n_lab + 1))
    for cid in np.argsort(-sizes) + 1:
        if sizes[cid - 1] < min_points:
            continue
        idx = np.argwhere(lab == cid)
        pts = lo + idx * step
        b = bur[tuple(idx.T)].astype(float)
        s = smooth[tuple(idx.T)]
        order = np.argsort(-s, kind="stable")
        peaks = []
        tree_pts = cKDTree(pts)
        taken = np.zeros(len(pts), bool)
        for i in order:                                  # greedy non-maximum suppression on the smoothed field
            if taken[i]:
                continue
            peaks.append(i)
            taken[tree_pts.query_ball_point(pts[i], nms)] = True
        pk_xyz = pts[peaks]
        owner = cKDTree(pk_xyz).query(pts)[1] if len(peaks) > 1 else np.zeros(len(pts), int)
        for j, p in enumerate(peaks):
            m = owner == j
            if m.sum() < min_points:
                continue
            w = b[m]
            cands.append(dict(center=(pts[m] * w[:, None]).sum(0) / w.sum(), peak=pts[p], points=pts[m], buried=w,
                              score=float(w.sum()), n_points=int(m.sum()), mean_buried=float(w.mean()),
                              max_buried=float(w.max()), cavity=int(cid), cavity_points=int(sizes[cid - 1])))
    cands.sort(key=lambda c: -c["score"])
    cands = cands[:max_sites]
    for r, c in enumerate(cands, 1):
        c["rank"] = r
    return (cands, field) if return_field else cands


def candidate_table(cands: list[dict]) -> list[dict]:
    """Flat, JSON-friendly rows (no point arrays): the native features every candidate carries into the ranker."""
    mx = max([c["score"] for c in cands] + [1e-9])
    rows = []
    for c in cands:
        rows.append(dict(rank=c["rank"], center=[round(float(x), 2) for x in c["center"]],
                         nat_score=c["score"], nat_rank=c["rank"], nat_rel=c["score"] / mx, nat_npts=c["n_points"],
                         nat_mean_bur=c["mean_buried"], nat_max_bur=c["max_buried"], nat_cavity_npts=c["cavity_points"],
                         n_cands=len(cands)))
    return rows
