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
6. Two tiers: deep cavities (>= DETECT_MIN_BURIED) first; remaining slots are filled by shallower cavities
   (>= FILL_MIN_BURIED) that are not within `nms` of a deep candidate.

Only coordinates are used: no sequence, no finder scores, no external program.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from . import pockets as pk

DETECT_MIN_BURIED = 16   # of 26 rays: cavity points for candidate detection (deeper than the cavity-mask MIN_BURIED)
FILL_MIN_BURIED = 10     # of 26 rays: second tier; shallower cavities fill the remaining slots (0 = off)
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
                 fill_min_buried: int = FILL_MIN_BURIED, return_field: bool = False) -> list[dict] | tuple[list[dict], dict]:
    """Candidate binding sites of one protein (heavy-atom coordinates only). Best-first, at most `max_sites`.

    Two tiers: deep cavities (buriedness >= `min_buried`) come first, ordered by score; if slots remain, cavities of the
    shallower field (>= `fill_min_buried`) whose centre is farther than `nms` from every deep candidate are appended,
    also by score. `tier` in the record says which one produced the candidate.
    Each candidate: dict(center [3], peak [3], points [n,3], buried [n] (per point), score, n_points, mean_buried,
    max_buried, cavity (id of the parent connected cavity), cavity_points (size of that cavity), tier, rank).
    With `return_field=True` the lattice (origin, shape, free mask, buriedness) is returned too.
    """
    xyz = np.asarray(xyz, float)
    if len(xyz) < 4:
        return ([], {}) if return_field else []
    lo, shape, dist = lattice(xyz, step)
    free = dist >= clash
    occupied = dist < pk.RAY_HIT
    bur = buriedness_grid(occupied, step)
    field = dict(origin=lo, shape=shape, free=free, buried=bur, dist=dist)
    cands = _split_cavities(lo, step, free, bur, min_buried, nms, min_points, tier=1)
    cands.sort(key=lambda c: -c["score"])
    cands = cands[:max_sites]
    if fill_min_buried and 0 < fill_min_buried < min_buried and len(cands) < max_sites:
        extra = _split_cavities(lo, step, free, bur, fill_min_buried, nms, min_points, tier=2)
        extra.sort(key=lambda c: -c["score"])
        have = np.array([c["center"] for c in cands]).reshape(-1, 3)
        for c in extra:
            if len(cands) >= max_sites:
                break
            if len(have) == 0 or np.linalg.norm(have - c["center"], axis=1).min() > nms:
                cands.append(c); have = np.vstack([have, c["center"]])
    for r, c in enumerate(cands, 1):
        c["rank"] = r
    return (cands, field) if return_field else cands


def _split_cavities(lo, step, free, bur, min_buried, nms, min_points, tier):
    """Connected cavities of `free & bur >= min_buried`, split into peak groups (NMS). Unordered list of candidates."""
    cav = free & (bur >= min_buried)
    lab, n_lab = ndimage.label(cav, structure=np.ones((3, 3, 3)))
    if n_lab == 0:
        return []
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
                              max_buried=float(w.max()), cavity=int(cid), cavity_points=int(sizes[cid - 1]), tier=tier))
    return cands


def grow_volumes(cands: list[dict], radius: float = 4.0) -> list[dict]:
    """Give each candidate the cavity points within `radius` of its own, without changing how many there are.

    The detector partitions a cavity among its buriedness peaks, so one peak group covers a fraction of a pocket
    and the ligand's envelope spans several. That is why DVO is capped near 0.07 on the raw groups. Merging whole
    cavities fixes the volume and costs the candidate ceiling, measured at 1.000 falling to 0.921 on 38 COACH420
    structures -- below P2Rank's 0.925, which is the one place we lead. Growing instead lets volumes overlap: each
    candidate keeps its own centre and its place in the ranking, and only the extent it claims changes.

    Members are taken from the parent cavity alone, so a grown volume never crosses into a different pocket.
    """
    by_cavity: dict = {}
    for c in cands:
        by_cavity.setdefault(c.get("cavity", id(c)), []).append(c)
    out = []
    for c in cands:
        pool = by_cavity.get(c.get("cavity", id(c)), [c])
        if len(pool) == 1:
            out.append(dict(c)); continue
        allpts = np.unique(np.vstack([m["points"] for m in pool]), axis=0)
        keep = cKDTree(c["points"]).query(allpts, k=1, distance_upper_bound=radius)[0]
        pts = allpts[np.isfinite(keep)]
        out.append(dict(c, points=pts, n_points=len(pts)))
    return out


def merge_by_cavity(cands: list[dict]) -> list[dict]:
    """One prediction per connected cavity instead of one per buriedness peak.

    The detector splits each cavity at peaks of the smoothed buriedness field, so a single pocket arrives as
    several candidates: thirty per structure against P2Rank's eight and DeepSurf's two. Two measurements say that
    costs us. A candidate holds a median of 21 lattice points while a ligand's own envelope runs to several
    hundred, which caps DVO near 0.07 however well placed the prediction is; and top-N divides by the structure's
    site count, so fragments of one pocket eat the budget that other pockets need.

    Merging is not free and the arm exists to price it: two ligands sharing one connected cavity become one
    prediction, and the candidate ceiling can only fall. The merged record keeps the best member's centre -- the
    deepest peak, not the cavity's centroid, which for a long groove would sit in no pocket at all -- while the
    volume, the score and the point count become the whole cavity's.
    """
    by_cavity: dict = {}
    for c in cands:
        by_cavity.setdefault(c.get("cavity", id(c)), []).append(c)
    out = []
    for members in by_cavity.values():
        best = max(members, key=lambda c: c["score"])
        pts = np.vstack([m["points"] for m in members]) if len(members) > 1 else best["points"]
        pts = np.unique(pts, axis=0)
        bur = np.concatenate([m["buried"] for m in members]) if len(members) > 1 else best["buried"]
        out.append(dict(best, points=pts, buried=bur, n_points=len(pts),
                        score=float(sum(m["score"] for m in members)),
                        mean_buried=float(np.mean(bur)), max_buried=float(np.max(bur)),
                        n_merged=len(members)))
    out.sort(key=lambda c: -c["score"])
    for i, c in enumerate(out, 1):
        c["rank"] = i
    return out


def candidate_table(cands: list[dict]) -> list[dict]:
    """Flat, JSON-friendly rows (no point arrays): the native features every candidate carries into the ranker."""
    mx = max([c["score"] for c in cands] + [1e-9])
    rows = []
    for c in cands:
        rows.append(dict(rank=c["rank"], center=[round(float(x), 2) for x in c["center"]],
                         nat_score=c["score"], nat_rank=c["rank"], nat_rel=c["score"] / mx, nat_npts=c["n_points"],
                         nat_mean_bur=c["mean_buried"], nat_max_bur=c["max_buried"], nat_cavity_npts=c["cavity_points"],
                         nat_tier=c["tier"],
                         n_cands=len(cands)))
    return rows
