"""The conservation feature slot, and the distinction between its two backends.

Conservation is the only external signal in this field with a measured effect that survives on train-dissimilar
structures -- +2.0 top-(N+2) on LIGYSIS -- and the only one with no training-set dependence at all. It also needs
an alignment against a sequence database, which the container this was written in cannot host. So the slot takes
a real score file on a machine that can produce one, and an ESM-derived proxy where it cannot, and the proxy is
labelled as a proxy everywhere it appears.
"""
import numpy as np
import pytest

from equicave import conservation as C


def test_a_score_file_is_read_by_residue_index(tmp_path):
    f = tmp_path / "c.txt"
    f.write_text("0 0.9\n2 0.1\n# a comment\n\n")
    got = C.from_file(f, 4)
    assert got[0] == pytest.approx(0.9) and got[2] == pytest.approx(0.1)
    assert np.isnan(got[1]) and np.isnan(got[3]), "an unscored residue stays missing rather than zero"


def test_a_missing_or_empty_file_is_none(tmp_path):
    assert C.from_file(tmp_path / "nope.txt", 4) is None
    (tmp_path / "empty.txt").write_text("\n\n")
    assert C.from_file(tmp_path / "empty.txt", 4) is None


def test_the_esm_proxy_is_larger_where_the_model_is_certain():
    certain = np.array([[10.0, 0.0, 0.0, 0.0]])
    unsure = np.array([[1.0, 1.0, 1.0, 1.0]])
    assert C.from_esm_entropy(certain)[0] > C.from_esm_entropy(unsure)[0]


def test_the_proxy_is_oriented_like_a_conservation_score():
    # Both backends must agree on direction or the featuriser silently inverts the signal.
    flat = C.from_esm_entropy(np.ones((1, 20)))[0]
    peaked = C.from_esm_entropy(np.concatenate([[20.0], np.zeros(19)])[None])[0]
    assert peaked > flat


def test_aggregation_drops_unscored_residues_rather_than_imputing():
    s = np.array([0.8, np.nan, 0.2])
    got = C.aggregate(s, [0, 1, 2])
    assert got["cons_mean"] == pytest.approx(0.5)      # the mean of 0.8 and 0.2, not of three values
    assert got["cons_max"] == pytest.approx(0.8)
    assert got["cons_min"] == pytest.approx(0.2)


def test_aggregation_of_nothing_is_zero_not_an_error():
    assert C.aggregate(np.array([np.nan, np.nan]), [0, 1]) == {k: 0.0 for k in C.FEATURES}
    assert C.aggregate(np.array([1.0]), []) == {k: 0.0 for k in C.FEATURES}


def test_weighting_moves_the_weighted_mean_and_nothing_else():
    s = np.array([1.0, 0.0])
    even = C.aggregate(s, [0, 1])
    tilted = C.aggregate(s, [0, 1], weights=np.array([9.0, 1.0]))
    assert tilted["cons_weighted"] > even["cons_weighted"]
    assert tilted["cons_mean"] == pytest.approx(even["cons_mean"])


def test_top3_is_the_mean_of_the_three_most_conserved():
    s = np.array([0.1, 0.2, 0.9, 1.0, 0.8])
    assert C.aggregate(s, list(range(5)))["cons_top3"] == pytest.approx((0.9 + 1.0 + 0.8) / 3)
