"""Two ways to give a prediction a volume, and the one that is free.

The detector partitions a cavity among its buriedness peaks, so a peak group covers a fraction of a pocket while
the ligand's envelope spans several -- which is why DVO on the raw groups is capped near 0.07 however well placed
the prediction is. Measured on 38 COACH420 structures: peak groups give DVO 0.178 at a candidate ceiling of 1.000
and 30 predictions per structure; growing each group to the cavity points within 6 A gives 0.294 at the same
ceiling and the same count; merging whole cavities gives 0.265 at 16.1 predictions and a ceiling of 0.921, which
is below P2Rank's 0.925 and so would cost the one place we lead.
"""
import numpy as np
import pytest

from equicave import detect


def cand(points, cavity, score=1.0):
    pts = np.asarray(points, float)
    return dict(points=pts, buried=np.full(len(pts), 20.0), cavity=cavity, score=score,
                n_points=len(pts), center=pts.mean(0), rank=1, mean_buried=20.0, max_buried=20.0,
                cavity_points=len(pts), tier=1, peak=pts[0])


def line(x0, n):
    return [[float(x0 + i), 0.0, 0.0] for i in range(n)]


def test_growing_keeps_every_candidate():
    cands = [cand(line(0, 3), "A"), cand(line(5, 3), "A"), cand(line(40, 3), "B")]
    assert len(detect.grow_volumes(cands, 6.0)) == 3


def test_growing_enlarges_a_fragment_from_its_own_cavity():
    cands = [cand(line(0, 3), "A"), cand(line(5, 3), "A")]
    out = detect.grow_volumes(cands, 6.0)
    assert out[0]["n_points"] > cands[0]["n_points"]
    assert out[0]["n_points"] == out[0]["points"].shape[0]


def test_growing_never_crosses_into_another_cavity():
    near = [[2.0, 0.0, 0.0]]                       # close in space, different cavity
    cands = [cand(line(0, 3), "A"), cand(near, "B")]
    out = detect.grow_volumes(cands, 10.0)
    assert out[0]["n_points"] == 3, "a grown volume must stay inside its own cavity"


def test_growing_leaves_a_lone_candidate_alone():
    cands = [cand(line(0, 3), "A")]
    assert detect.grow_volumes(cands, 6.0)[0]["n_points"] == 3


def test_merging_reduces_the_count_and_unions_the_points():
    cands = [cand(line(0, 3), "A", 2.0), cand(line(5, 3), "A", 1.0), cand(line(40, 3), "B", 3.0)]
    out = detect.merge_by_cavity(cands)
    assert len(out) == 2
    a = next(c for c in out if c["n_merged"] == 2)
    assert a["n_points"] == 6
    assert a["score"] == pytest.approx(3.0)        # the cavity's score is its members' together


def test_merging_keeps_the_best_member_s_centre_not_the_cavity_centroid():
    # For a long groove the centroid sits in no pocket at all, so the deepest peak's centre is kept instead.
    cands = [cand(line(0, 3), "A", 5.0), cand(line(30, 3), "A", 1.0)]
    out = detect.merge_by_cavity(cands)
    assert len(out) == 1
    assert out[0]["center"] == pytest.approx(cands[0]["center"])


def test_merging_reranks_from_one():
    cands = [cand(line(0, 3), "A", 1.0), cand(line(40, 3), "B", 9.0)]
    out = detect.merge_by_cavity(cands)
    assert [c["rank"] for c in out] == [1, 2]
    assert out[0]["cavity"] == "B"
