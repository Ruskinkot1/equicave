"""DVO and PLI: whether a prediction is the right shape, not only in the right place.

DCC and DCA both collapse a pocket to its centre, so a prediction covering a tenth of the site and one engulfing
half the protein can score identically on them. The volume metrics are what the segmentation line of the
literature (Kalasanty, PUResNet) reports alongside DCC, and they are what separates those two cases.
"""
import numpy as np
import pytest

from equicave import pockets as pk


def cube(n=4, origin=(0.0, 0.0, 0.0)):
    g = np.indices((n, n, n)).reshape(3, -1).T.astype(float)
    return g + np.asarray(origin, float)


def test_dvo_of_a_volume_with_itself_is_one():
    c = cube()
    assert pk.dvo(c, c) == pytest.approx(1.0)


def test_dvo_of_disjoint_volumes_is_zero():
    assert pk.dvo(cube(), cube(origin=(50.0, 0.0, 0.0))) == 0.0


def test_dvo_is_the_jaccard_index():
    # 4^3 = 64 cells each, shifted by two along x: overlap is a 2x4x4 = 32 slab, union 64 + 64 - 32 = 96.
    got = pk.dvo(cube(), cube(origin=(2.0, 0.0, 0.0)))
    assert got == pytest.approx(32 / 96)


def test_dvo_punishes_a_prediction_that_is_too_large():
    true = cube(4)
    assert pk.dvo(cube(8), true) < pk.dvo(cube(4), true)


def test_dvo_punishes_a_prediction_that_is_too_small():
    true = cube(4)
    assert pk.dvo(cube(2), true) < 1.0


def test_an_empty_volume_overlaps_nothing():
    assert pk.dvo(np.empty((0, 3)), cube()) == 0.0
    assert pk.dvo(cube(), np.empty((0, 3))) == 0.0


def test_distance_and_volume_disagree_on_purpose():
    # The case the metric exists for: a prediction centred exactly on the ligand that covers almost none of it.
    true = cube(6)
    centre = true.mean(0)
    tiny = np.array([centre])
    assert pk.dcc(tiny.mean(0), true) == pytest.approx(0.0, abs=1e-9)   # perfect by distance
    assert pk.dvo(tiny, true) < 0.01                                    # and almost nothing by volume


def test_pli_counts_the_ligand_atoms_inside():
    pocket = cube(4)
    lig = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [50.0, 0.0, 0.0]])
    assert pk.pli(pocket, lig) == pytest.approx(2 / 3)


def test_pli_is_one_when_the_pocket_swallows_the_ligand():
    assert pk.pli(cube(10), np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])) == pytest.approx(1.0)


def test_pli_and_dvo_separate_coverage_from_size():
    # An oversized pocket covers the whole ligand (PLI 1.0) while scoring badly on DVO. Reporting one without the
    # other would call this a good prediction.
    true, lig = cube(3), cube(3)
    big = cube(9)
    assert pk.pli(big, lig) == pytest.approx(1.0)
    assert pk.dvo(big, true) < 0.1


def test_site_voxels_keeps_only_free_points_near_the_ligand():
    free = cube(10)
    lig = np.array([[1.0, 1.0, 1.0]])
    got = pk.site_voxels(lig, free, radius=2.0)
    assert len(got) > 0
    assert np.linalg.norm(got - lig, axis=1).max() <= 2.0 + 1e-9
    assert len(got) < len(free)


def test_site_voxels_handles_empty_input():
    assert len(pk.site_voxels(np.empty((0, 3)), cube())) == 0
    assert len(pk.site_voxels(np.array([[0.0, 0.0, 0.0]]), np.empty((0, 3)))) == 0
