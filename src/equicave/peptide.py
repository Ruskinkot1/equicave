"""Peptide-binder site finding: shallow elongated grooves, which the small-molecule detector is not tuned for.

A peptide binds a groove, not a buried cavity: it keeps its own backbone, lies partly in solvent, extends 10-30 A and
makes backbone-to-backbone hydrogen bonds with the receptor. Three consequences for detection:

1. **Depth.** Groove points are closed on fewer sides. The peptide tier uses `GROOVE_MIN_BURIED` (default 10 of 26 rays)
   instead of the 16 used for small molecules, so the candidate set is much larger and must be filtered by shape, not depth.
2. **Shape.** A groove is anisotropic: its longest principal extent is several times the shortest. `GROOVE_MIN_LENGTH`
   and `GROOVE_MIN_ANISOTROPY` keep elongated patches and drop round dimples and large open faces.
3. **Chemistry of the wall.** Peptide recognition needs exposed receptor backbone N and O (beta-augmentation, PDZ/SH3/MHC
   grooves) and often alternating hydrophobic sub-pockets. `backbone_exposure` and `groove_features` measure both.

Candidates are produced by `detect_peptide_sites`. The shallow field of a protein is one connected sheet over most of
its surface, so a global split is meaningless: instead the smoothed buriedness field is scanned for local peaks with
non-maximum suppression (`PEAK_NMS`, default 5 A, denser than the 6 A used for cavities because grooves overlap), and
each peak grows into a candidate from the patch points within `SEG_LEN / 2` of it. A long groove therefore yields a
chain of overlapping candidates that together cover a bound peptide, and each candidate is kept only if it is
elongated (`GROOVE_MIN_ANISOTROPY`) and long enough (`GROOVE_MIN_LENGTH`). Scores are geometric only; ranking is
learned (`peptide.groove_features` feed the same LambdaRank model family as the small-molecule features).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from . import detect, pockets as pk

GROOVE_MIN_BURIED = 10        # of 26 rays: shallow enough for a solvent-exposed groove
GROOVE_MIN_POINTS = 30        # lattice points: a groove that could host >= 3 peptide residues
GROOVE_MIN_LENGTH = 9.0       # A: longest principal extent of a kept segment
GROOVE_MIN_ANISOTROPY = 1.8   # ax1 / ax3 of a kept segment
SEG_LEN = 14.0                # A: extent of one groove segment (about 4 extended peptide residues)
PEAK_NMS = 5.0                # A: minimum separation between groove peaks
SMOOTH_SIGMA = 2.0            # A: smoothing of the buriedness field before peak picking
MAX_SITES = 30
BACKBONE_R = 6.0              # A: radius in which exposed backbone N/O are counted


def backbone_exposure(points: np.ndarray, st: dict, radius: float = BACKBONE_R) -> dict[str, np.ndarray]:
    """Per point: counts of receptor backbone N, O and CA, and of side-chain carbons, within `radius`.

    Exposed backbone donors and acceptors are what a peptide's own backbone pairs with, so they separate a peptide
    groove from a buried small-molecule cavity, which is lined mostly by side chains.
    """
    xyz = st["xyz"]; bb = st["backbone"]; atom = st["atom"]; el = st["element"]
    sets = dict(bb_n=bb & (atom == "N"), bb_o=bb & (atom == "O"), bb_ca=bb & (atom == "CA"), sc_c=(~bb) & (el == "C"),
                sc_polar=(~bb) & np.isin(el, ["N", "O"]))
    out = {}
    for k, m in sets.items():
        if m.sum() == 0:
            out[k] = np.zeros(len(points)); continue
        t = cKDTree(xyz[m])
        out[k] = np.array([len(x) for x in t.query_ball_point(points, radius)], float)
    return out


def _peak_segments(pts: np.ndarray, bur: np.ndarray, smooth: np.ndarray, seg_len: float, nms: float, min_points: int):
    """Local groove segments of one patch: peaks of `smooth` with NMS, each grown to the points within seg_len / 2."""
    tree = cKDTree(pts)
    order = np.argsort(-smooth, kind="stable")
    taken = np.zeros(len(pts), bool)
    out = []
    for i in order:
        if taken[i]:
            continue
        taken[tree.query_ball_point(pts[i], nms)] = True
        m = np.array(tree.query_ball_point(pts[i], seg_len / 2.0))
        if len(m) >= min_points:
            out.append((pts[m], bur[m], pts[i]))
    return out


def detect_peptide_sites(xyz: np.ndarray, st: dict | None = None, step: float = pk.GRID, clash: float = pk.CLASH,
                         min_buried: int = GROOVE_MIN_BURIED, min_points: int = GROOVE_MIN_POINTS,
                         min_length: float = GROOVE_MIN_LENGTH, min_anisotropy: float = GROOVE_MIN_ANISOTROPY,
                         seg_len: float = SEG_LEN, nms: float = PEAK_NMS, max_sites: int = MAX_SITES,
                         return_field: bool = False):
    """Peptide-binding groove candidates of one receptor (heavy-atom coordinates only). Best-first.

    Each candidate: dict(center, peak, points, buried, score, n_points, mean_buried, length, width, depth_extent,
    anisotropy, axis, groove (parent patch id), tier=3, rank) and, when `st` is given, the backbone-exposure counts
    averaged over its points.
    Score = sum of buriedness x anisotropy (an elongated, moderately closed patch scores above a round dimple of the
    same volume and above a wide open face).
    """
    xyz = np.asarray(xyz, float)
    if len(xyz) < 4:
        return ([], {}) if return_field else []
    lo, shape, dist = detect.lattice(xyz, step)
    free = dist >= clash
    bur = detect.buriedness_grid(dist < pk.RAY_HIT, step)
    patch = free & (bur >= min_buried) & (bur < 26)
    lab, n = ndimage.label(patch, structure=np.ones((3, 3, 3)))
    field = dict(origin=lo, shape=shape, free=free, buried=bur, dist=dist, labels=lab)
    smooth_grid = ndimage.gaussian_filter((bur * patch).astype(float), SMOOTH_SIGMA / step)
    cands = []
    if n:
        sizes = ndimage.sum(patch, lab, index=np.arange(1, n + 1))
        for pid in np.argsort(-sizes) + 1:
            if sizes[pid - 1] < min_points:
                continue
            idx = np.argwhere(lab == pid)
            pts = lo + idx * step
            b = bur[tuple(idx.T)].astype(float)
            sm = smooth_grid[tuple(idx.T)]
            for spts, sb, peak in _peak_segments(pts, b, sm, seg_len, nms, min_points):
                ax = pk.shape_axes(spts)
                aniso = ax[0] / max(ax[2], 1.0)
                if ax[0] < min_length or aniso < min_anisotropy:
                    continue
                c = spts - spts.mean(0)
                _, vec = np.linalg.eigh(np.cov(c.T))
                cands.append(dict(center=(spts * sb[:, None]).sum(0) / sb.sum(), peak=peak, points=spts, buried=sb,
                                  score=float(sb.sum() * min(aniso, 6.0)), n_points=int(len(spts)), mean_buried=float(sb.mean()),
                                  max_buried=float(sb.max()), length=float(ax[0]), width=float(ax[1]), depth_extent=float(ax[2]),
                                  anisotropy=float(aniso), axis=vec[:, -1], groove=int(pid), cavity_points=int(sizes[pid - 1]), tier=3))
    cands.sort(key=lambda c: -c["score"])
    cands = cands[:max_sites]
    for r, c in enumerate(cands, 1):
        c["rank"] = r
        if st is not None:
            ex = backbone_exposure(c["points"], st)
            c.update({f"pep_{k}": float(v.mean()) for k, v in ex.items()})
    return (cands, field) if return_field else cands


def groove_features(cands: list[dict], st: dict) -> list[dict]:
    """Peptide-specific features per candidate (used in addition to the shared geometry / chemistry features)."""
    rows = []
    mx = max([c["score"] for c in cands] + [1e-9])
    for c in cands:
        ex = {f"pep_{k}": c.get(f"pep_{k}") for k in ("bb_n", "bb_o", "bb_ca", "sc_c", "sc_polar")}
        if any(v is None for v in ex.values()):
            e = backbone_exposure(c["points"], st)
            ex = {f"pep_{k}": float(v.mean()) for k, v in e.items()}
        bb = ex["pep_bb_n"] + ex["pep_bb_o"]
        rows.append(dict(pep_score=c["score"], pep_rank=c["rank"], pep_rel=c["score"] / mx, pep_npts=c["n_points"],
                         pep_mean_bur=c["mean_buried"], pep_length=c["length"], pep_width=c["width"],
                         pep_anisotropy=c["anisotropy"], pep_bb_total=bb,
                         pep_bb_ratio=bb / max(1.0, bb + ex["pep_sc_c"] + ex["pep_sc_polar"]), **ex,
                         n_pep_cands=len(cands)))
    return rows


def merge_with_small_molecule(pep: list[dict], small: list[dict], nms: float = 6.0, max_sites: int = MAX_SITES) -> list[dict]:
    """One candidate list for a structure that may bind either a peptide or a small molecule.

    Small-molecule candidates keep their order; groove candidates that are not within `nms` of one of them are appended.
    `tier` tells them apart (1 deep cavity, 2 shallow cavity, 3 groove), so the ranker can learn to prefer either.
    """
    out = [dict(c) for c in small]
    have = np.array([c["center"] for c in out]).reshape(-1, 3)
    for c in sorted(pep, key=lambda c: -c["score"]):
        if len(out) >= max_sites:
            break
        if len(have) == 0 or np.linalg.norm(have - c["center"], axis=1).min() > nms:
            out.append(dict(c)); have = np.vstack([have, c["center"]])
    out.sort(key=lambda c: (c["tier"], -c["score"]))
    for r, c in enumerate(out, 1):
        c["rank"] = r
    return out[:max_sites]
