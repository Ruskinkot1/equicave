"""A learned ligandability score for a single empty point, and its aggregates over a candidate.

Measured on COACH420 (`docs/results/compare_coach420.md`), our candidate set contains the right answer for 0.992 of
the comparable structures against P2Rank's 0.935, yet our first prediction is correct less often. The difference is
not detection and not prediction merging; it is how a candidate is scored. Our pocket features summarise a cavity
with means and counts over its whole extent, which dilutes a small well-formed sub-pocket inside a large hole. P2Rank
instead classifies every surface point and adds the scores up, so a pocket's score grows with how many of its points
individually look ligandable (idea recorded in docs/PROVENANCE.md; the implementation, the features and the labels
here are our own and operate on cavity grid points rather than on a SAS triangulation).

This module supplies the two halves of that: `point_features` describes one empty point by what a ligand atom placed
there could bind to and how enclosed it is, and `aggregate` turns a vector of per-point probabilities into candidate
features -- the additive sum, the shape of the score distribution, the largest contiguous high-scoring blob, and the
displacement from the candidate centre to the score-weighted centroid. The model itself is trained by
scripts/train/train_point_model.py and applied out of fold, so no candidate is ever scored by a model that saw its
structure.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from . import pocket_features as pf
from . import pockets as pk

OCC_RADIUS = 2.0          # A: a grid point counts as ligandable when a ligand heavy atom is this close
HIGH = (0.3, 0.5, 0.7)    # score thresholds whose point counts become candidate features
BLOB_LINK = 1.8           # A: two high-scoring points are in the same blob within this distance (1 A grid diagonal)

# per-point inputs: the 21 interaction-potential numbers plus enclosure and local density
POINT_FEATURES = ([f"p_n_{c}" for c in pf.POT_CLASSES] + [f"p_d_{c}" for c in pf.POT_CLASSES] +
                  [f"p_s_{c}" for c in pf.POT_CLASSES] +
                  ["p_classes", "p_d_min", "p_n_total", "p_buried", "p_width", "p_n6", "p_n8", "p_n12",
                   "p_f_hydrophobic8", "p_f_polar8", "p_centrality"])

POINT = pf.POINT_AGG      # per-candidate aggregates of the per-point score; named once, in pocket_features


def point_features(points: np.ndarray, st: dict, tp: cKDTree, trees: dict, rg: float = 0.0,
                   centroid: np.ndarray | None = None) -> np.ndarray:
    """One row of `POINT_FEATURES` per point. `trees` is the per-atom-type KD-tree map `featurize` already builds."""
    pts = np.atleast_2d(np.asarray(points, float))
    n = len(pts)
    if n == 0:
        return np.zeros((0, len(POINT_FEATURES)), np.float32)
    xyz = st["xyz"]
    if centroid is None:
        centroid = xyz.mean(0)
    if not rg:
        rg = float(np.sqrt(((xyz - centroid) ** 2).sum(1).mean())) or 1.0
    C, D, A = pf.point_potential(pts, trees)
    bur = pk._buried(pts, tp).astype(np.float32)
    width = tp.query(pts)[0]
    n6 = np.array([len(x) for x in tp.query_ball_point(pts, 6.0)], np.float32)
    n8_idx = tp.query_ball_point(pts, 8.0)
    n8 = np.array([len(x) for x in n8_idx], np.float32)
    n12 = np.array([len(x) for x in tp.query_ball_point(pts, 12.0)], np.float32)
    hyd, pol = np.zeros(n, np.float32), np.zeros(n, np.float32)
    for i, idx in enumerate(n8_idx):
        if not len(idx):
            continue
        el = st["element"][idx]
        hyd[i] = float((el == "C").mean()); pol[i] = float(np.isin(el, ("N", "O")).mean())
    cent = np.linalg.norm(pts - centroid, axis=1) / rg
    cols = [C[:, j] for j in range(C.shape[1])] + [D[:, j] for j in range(D.shape[1])] + \
           [A[:, j] for j in range(A.shape[1])] + \
           [A.sum(1), D.min(1), C.sum(1), bur, width, n6, n8, n12, hyd, pol, cent]
    return np.column_stack(cols).astype(np.float32)


def point_labels(points: np.ndarray, ligand_xyz: np.ndarray, radius: float = OCC_RADIUS) -> np.ndarray:
    """1 where a ligand heavy atom really sits within `radius` of the grid point."""
    pts = np.atleast_2d(np.asarray(points, float))
    if len(pts) == 0 or ligand_xyz is None or len(ligand_xyz) == 0:
        return np.zeros(len(pts), np.int8)
    return (cKDTree(np.asarray(ligand_xyz, float)).query(pts)[0] <= radius).astype(np.int8)


def _blobs(pts: np.ndarray, w: np.ndarray, link: float = BLOB_LINK) -> tuple[float, float]:
    """Largest connected group of the given points by score weight: (its weight, its point count).

    A real sub-pocket is a contiguous patch of high-scoring points; scattered points that happen to score well are
    not. Connectivity over the grid separates the two, and the additive sum alone cannot.
    """
    if len(pts) == 0:
        return 0.0, 0.0
    pairs = cKDTree(pts).query_pairs(link, output_type="ndarray")
    parent = np.arange(len(pts))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    for i, j in pairs:
        a, b = find(i), find(j)
        if a != b:
            parent[a] = b
    root = np.array([find(i) for i in range(len(pts))])
    best_w, best_n = 0.0, 0.0
    for r in np.unique(root):
        m = root == r
        if w[m].sum() > best_w:
            best_w, best_n = float(w[m].sum()), float(m.sum())
    return best_w, best_n


def aggregate(scores: np.ndarray, points: np.ndarray, center: np.ndarray) -> dict:
    """Candidate features from the per-point scores of that candidate's own points."""
    s = np.asarray(scores, float).ravel(); pts = np.atleast_2d(np.asarray(points, float))
    if len(s) == 0:
        return {f: 0.0 for f in POINT}
    order = np.argsort(-s)
    top10 = order[:max(1, len(s) // 10)]
    high = s >= HIGH[1]
    bw, bn = _blobs(pts[high], s[high])
    w = s / max(1e-9, s.sum())
    out = {"pts_sum": float(s.sum()), "pts_mean": float(s.mean()), "pts_max": float(s.max()),
           "pts_p90": float(np.percentile(s, 90)), "pts_p50": float(np.percentile(s, 50)),
           "pts_top10_mean": float(s[top10].mean()), "pts_sd": float(s.std()),
           "pts_blob_max": bw, "pts_blob_sum": bn, "pts_n_points": float(len(s)),
           "pts_centroid_shift": float(np.linalg.norm((pts * w[:, None]).sum(0) - np.asarray(center, float))),
           "pts_sum_per_point": float(s.sum() / len(s))}
    for h in HIGH:
        out[f"pts_n_{int(h * 100)}"] = float((s >= h).sum())
        out[f"pts_f_{int(h * 100)}"] = float((s >= h).mean())
    return out


def weighted_center(scores: np.ndarray, points: np.ndarray) -> np.ndarray:
    """The score-weighted centroid of a candidate's points: an alternative centre, measured separately."""
    s = np.clip(np.asarray(scores, float).ravel(), 0.0, None); pts = np.atleast_2d(np.asarray(points, float))
    if len(s) == 0 or s.sum() <= 0:
        return pts.mean(0) if len(pts) else np.zeros(3)
    return (pts * (s / s.sum())[:, None]).sum(0)
