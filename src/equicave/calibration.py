"""Post-hoc calibration of the per-point heads, and the one place it can change a ranking.

Calibration is normally cosmetic for ranking, and it is worth being explicit about why, because the obvious
objection is right: temperature scaling is a monotone map, so it cannot change an argmax. Rank a site by the
maximum or the mean of its probes' probabilities and calibration changes nothing at all.

It stops being cosmetic the moment a site's score is a *nonlinear* aggregate of per-point probabilities. P2Rank
ranks a cluster by the **sum of the squares** of its per-point ligandability, and sum of squares is not invariant
under a monotone reparameterisation: pushing probabilities toward 0 and 1 rewards a site with a few confident
points, pulling them toward the middle rewards a site with many lukewarm ones. So under that rule -- the rule the
strongest ranker in this field uses, and our own `site_agg` default -- calibration is a ranking intervention.

No paper in the binding-site literature calibrates pocket scores and reports the effect on ranked success, so the
sign is unknown. `sumsq_sites` and the metrics around it exist to measure it rather than to assume it.

Temperature is fitted by minimising the negative log likelihood on held-out logits, the standard single-parameter
form (Guo et al. 2017). One scalar, so it cannot overfit a split of any reasonable size, and it is fitted on
structures the number is not reported on -- see `split_ids`.
"""
from __future__ import annotations

import hashlib

import numpy as np


def _nll(t: float, logit: np.ndarray, y: np.ndarray) -> float:
    z = logit / max(t, 1e-6)
    # log(1 + exp(-z)) for y = 1 and log(1 + exp(z)) for y = 0, in the stable form
    return float(np.mean(np.logaddexp(0.0, -z) * y + np.logaddexp(0.0, z) * (1 - y)))


def at_bound(t: float, lo: float = 0.05, hi: float = 20.0, tol: float = 0.02) -> bool:
    """Did the search land on its own boundary? Then the value is a floor or a ceiling, not a fitted temperature.

    Worth reporting rather than hiding: a head so miscalibrated that NLL still falls at T = 20 has a problem a
    scalar cannot fix, and publishing 20.0 as "the temperature" would present a boundary as a measurement.
    """
    return bool(t <= lo * (1 + tol) or t >= hi * (1 - tol))


def temperature(logit: np.ndarray, y: np.ndarray, lo: float = 0.05, hi: float = 20.0, iters: int = 60) -> float:
    """The scalar T minimising NLL of sigmoid(logit / T) against y. T > 1 softens, T < 1 sharpens.

    Golden-section search rather than a gradient step: the objective is one-dimensional and convex in log T, the
    data here is a few hundred thousand probes, and a closed loop with no learning rate is one fewer thing that
    can silently fail to converge.
    """
    logit = np.asarray(logit, float).ravel(); y = np.asarray(y, float).ravel()
    if len(y) == 0 or y.min() == y.max():
        return 1.0                                  # one class only: nothing to calibrate against
    a, b = np.log(lo), np.log(hi)
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = _nll(np.exp(c), logit, y), _nll(np.exp(d), logit, y)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a); fc = _nll(np.exp(c), logit, y)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a); fd = _nll(np.exp(d), logit, y)
    return float(np.exp((a + b) / 2))


def apply(logit: np.ndarray, t: float) -> np.ndarray:
    """sigmoid(logit / t)."""
    z = np.asarray(logit, float) / max(float(t), 1e-6)
    return 1.0 / (1.0 + np.exp(-z))


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    """Per-bin count, mean predicted probability and observed frequency -- the curve behind the ECE number."""
    y = np.asarray(y, float).ravel(); p = np.asarray(p, float).ravel()
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        out.append(dict(lo=float(edges[i]), hi=float(edges[i + 1]), n=int(m.sum()),
                        p_mean=float(p[m].mean()) if m.any() else float("nan"),
                        y_mean=float(y[m].mean()) if m.any() else float("nan")))
    return out


def split_ids(ids: list[str], frac: float = 0.5, salt: str = "calib") -> tuple[set, set]:
    """Deterministic split of structure ids into (fit, report) by a hash of the id.

    Hashed rather than shuffled so the two halves do not move when the file order does, and so the temperature is
    never fitted on the structures its effect is reported on -- which would make a calibration gain unfalsifiable.
    """
    fit = {i for i in ids
           if int(hashlib.sha1(f"{salt}:{i}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac}
    return fit, {i for i in ids if i not in fit}


def sumsq_sites(occ_prob: np.ndarray, probe_pos: np.ndarray, conf: np.ndarray, center: np.ndarray,
                nms: float = 6.0, membership: float = 8.0, max_sites: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """P2Rank's ranking rule applied to our probes: sites seeded by confidence under NMS, scored by sum of squares.

    Returns (centres [S, 3], scores [S]) in descending score order. The seeds and the NMS radius are the
    evaluation's own, so the only thing that varies between a calibrated and an uncalibrated call is the score --
    which is the whole point: it isolates the effect of calibration on the ordering under a nonlinear aggregate.
    """
    occ_prob = np.asarray(occ_prob, float).ravel()
    probe_pos = np.asarray(probe_pos, float).reshape(-1, 3)
    center = np.asarray(center, float).reshape(-1, 3)
    order = np.argsort(-np.asarray(conf, float).ravel())
    seeds: list[int] = []
    for i in order:
        if all(np.linalg.norm(center[i] - center[j]) > nms for j in seeds):
            seeds.append(int(i))
        if len(seeds) >= max_sites:
            break
    if not seeds:
        return np.zeros((0, 3)), np.zeros(0)
    ctr = center[seeds]
    d = np.linalg.norm(probe_pos[None, :, :] - ctr[:, None, :], axis=2)
    w = (d <= membership).astype(float)
    score = (w * occ_prob[None, :] ** 2).sum(1)
    o = np.argsort(-score)
    return ctr[o], score[o]
