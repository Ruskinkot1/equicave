"""Per-residue conservation, and the honest account of what can be computed where.

P2Rank's optional conservation feature is worth +2.0 points of top-(N+2) on LIGYSIS and +346 true positives at a
fixed 100-false-positive budget, measured on a benchmark whose training overlap is 0.5-9.7 %. It is the only
external signal in this field with a measured effect that holds on train-dissimilar structures, and it has no
dependence on a training set at all: it is computed per target from a multiple sequence alignment.

That last property is also why it is hard to get. An MSA needs a sequence database -- UniRef90 is about 100 GB --
and a search tool. Neither HMMER, MMseqs2 nor BLAST is installed in the container this was written in, and
ConSurf-DB, the precomputed alternative, answered 404. So this module defines the feature and two backends, and
is explicit about which is which:

* `from_file` reads precomputed per-residue scores in P2Rank's own format, one `index score` pair per line. This
  is the real thing, and it is what to use on a machine that can run an alignment.
* `from_esm_entropy` derives a proxy from the masked language model we already run: the entropy of ESM-2's
  per-position distribution, low where the model is certain which residue belongs. It correlates with
  conservation because both measure the same evolutionary pressure, but it is **not** conservation, it carries no
  alignment depth, and the +2.0 above does not transfer to it. Our own ablation also argues against it: removing
  ESM-2 entirely costs -0.010, so a scalar derived from the same model has little room.

Nothing here is enabled by default. The point is that the feature slot and its aggregation exist, so a real
conservation file can be dropped in without touching the ranker.
"""
from __future__ import annotations

import pathlib

import numpy as np

# Names the candidate featuriser adds when a conservation source is supplied.
FEATURES = ["cons_mean", "cons_max", "cons_min", "cons_top3", "cons_weighted"]


def from_file(path: str | pathlib.Path, n_res: int) -> np.ndarray | None:
    """Per-residue scores from a two-column file, as P2Rank consumes them. Higher means more conserved."""
    p = pathlib.Path(path)
    if not p.exists():
        return None
    out = np.full(n_res, np.nan)
    for line in p.read_text().splitlines():
        f = line.split()
        if len(f) < 2:
            continue
        try:
            i, v = int(f[0]), float(f[1])
        except ValueError:
            continue
        if 0 <= i < n_res:
            out[i] = v
    return out if np.isfinite(out).any() else None


def from_esm_entropy(logits: np.ndarray) -> np.ndarray:
    """A proxy, not conservation: the negated entropy of ESM-2's per-position distribution.

    Returned on the same orientation as a conservation score -- larger is more constrained -- so the two backends
    are interchangeable in the featuriser. The sign flip is the whole conversion; no calibration is implied and
    none would make this the measured quantity.
    """
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max(-1, keepdims=True)
    p = np.exp(x)
    p /= p.sum(-1, keepdims=True)
    return -(-(p * np.log(p + 1e-12)).sum(-1))


def aggregate(scores: np.ndarray, res_idx: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """Collapse the conservation of a candidate's lining residues into the five candidate features.

    `res_idx` are the residues lining this candidate and `weights` an optional per-residue weight, such as how
    much of the pocket surface each contributes. Residues with no score are dropped rather than imputed: a gap in
    an alignment is missing information, and filling it with the mean would invent agreement where there is none.
    """
    if len(res_idx) == 0:
        return {k: 0.0 for k in FEATURES}
    s = np.asarray(scores)[np.asarray(res_idx, dtype=int)]
    ok = np.isfinite(s)
    if not ok.any():
        return {k: 0.0 for k in FEATURES}
    s = s[ok]
    w = np.ones_like(s) if weights is None else np.asarray(weights, float)[ok]
    top3 = np.sort(s)[-3:]
    return dict(cons_mean=float(s.mean()), cons_max=float(s.max()), cons_min=float(s.min()),
                cons_top3=float(top3.mean()),
                cons_weighted=float((s * w).sum() / max(w.sum(), 1e-9)))
