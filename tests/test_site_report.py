"""The object a user receives, as distinct from the row a benchmark scores.

A benchmark needs one number per site: did the centre land within 4 A. Someone deciding whether to run a screen
needs the docking box, the lining residues, the chemistry on offer and a reason to believe any of it. These tests
pin the parts of that object a user would act on, because a wrong residue number or an unusable box is a mistake
that reaches further than a wrong metric.
"""
import numpy as np
import pytest

from equicave import site_report as SR


def grid(n=5, origin=(0.0, 0.0, 0.0)):
    g = np.indices((n, n, n)).reshape(3, -1).T.astype(float)
    return g + np.asarray(origin, float)


def protein(d=8.0):
    return np.array([[d, 0.0, 0.0], [-d, 0.0, 0.0], [0.0, d, 0.0], [0.0, -d, 0.0]])


def st_of(xyz, resids, resns, chains):
    return dict(xyz=np.asarray(xyz, float), resid=np.array(resids), resname=np.array(resns),
                chain=np.array(chains))


# --- the docking box -------------------------------------------------------------------------------------------

def test_the_box_contains_the_pocket_with_padding():
    p = grid(5)
    b = SR.docking_box(p, pad=4.0)
    lo = np.array(b["center"]) - np.array(b["size"]) / 2
    hi = np.array(b["center"]) + np.array(b["size"]) / 2
    assert (p.min(0) >= lo - 1e-6).all() and (p.max(0) <= hi + 1e-6).all()


def test_a_small_pocket_still_gets_a_searchable_box():
    # Vina's search degenerates below about 12 A on a side, so a one-point pocket must not produce a box of zero.
    b = SR.docking_box(np.array([[0.0, 0.0, 0.0]]), pad=4.0, min_size=12.0)
    assert min(b["size"]) >= 12.0


def test_an_empty_pocket_has_no_box():
    assert SR.docking_box(np.empty((0, 3))) == {}


# --- what fits -------------------------------------------------------------------------------------------------

def test_the_inscribed_radius_is_the_room_at_the_pocket_s_widest():
    # A point 8 A from every protein atom can hold a sphere of radius 8; volume alone cannot say that.
    assert SR.max_inscribed_radius(np.array([[0.0, 0.0, 0.0]]), protein(8.0)) == pytest.approx(8.0)


def test_inscribed_radius_handles_empty_input():
    assert SR.max_inscribed_radius(np.empty((0, 3)), protein()) == 0.0
    assert SR.max_inscribed_radius(grid(), np.empty((0, 3))) == 0.0


# --- the pharmacophore -------------------------------------------------------------------------------------------

def test_the_pharmacophore_is_a_distribution_over_the_seven_classes():
    p = SR.pharmacophore(np.zeros((10, 7)))
    assert set(p) == set(SR.PHARMACOPHORE)
    assert sum(p.values()) == pytest.approx(1.0, abs=1e-3)


def test_the_pharmacophore_names_the_class_the_model_is_confident_about():
    logits = np.full((10, 7), -5.0)
    logits[:, SR.PHARMACOPHORE.index("aromatic")] = 5.0
    p = SR.pharmacophore(logits)
    assert max(p, key=p.get) == "aromatic"


def test_the_pharmacophore_is_size_independent():
    # A pocket twice the size with the same character must read the same, or a user cannot compare two sites.
    logits = np.tile(np.array([2.0, -1.0, 0.0, 1.0, -2.0, -3.0, -4.0]), (6, 1))
    assert SR.pharmacophore(logits) == SR.pharmacophore(np.tile(logits, (2, 1)))


def test_the_hotspot_map_places_each_chemistry_where_it_is_favoured():
    pts = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    logits = np.full((2, 7), -5.0)
    logits[0, SR.PHARMACOPHORE.index("hbd")] = 5.0
    logits[1, SR.PHARMACOPHORE.index("hydrophobic_c")] = 5.0
    m = SR.hotspot_map(pts, logits, top=4)
    by_type = {d["type"]: d["center"] for d in m}
    assert by_type["hbd"] == [0.0, 0.0, 0.0]
    assert by_type["hydrophobic_c"] == [10.0, 0.0, 0.0]


def test_the_hotspot_map_drops_what_the_model_is_unsure_of():
    assert SR.hotspot_map(np.zeros((3, 3)), np.zeros((3, 7)), min_p=0.9) == []


def test_ligand_types_are_independent_not_a_softmax():
    # A site can be both deeply buried and aromatic, so these must not be forced to sum to one.
    t = SR.ligand_type(np.full(14, 3.0))
    assert all(v > 0.9 for v in t.values())
    assert sum(t.values()) > 1.0


# --- the residues ------------------------------------------------------------------------------------------------

def test_residues_come_back_in_author_numbering_nearest_first():
    st = st_of([[1.0, 0, 0], [20.0, 0, 0], [2.0, 0, 0]], ["A_145", "B_7", "A_146"],
               ["TYR", "GLY", "ASP"], ["A", "B", "A"])
    r = SR.lining_residues(np.array([[0.0, 0, 0]]), st, radius=5.0)
    assert [x["resi"] for x in r] == ["145", "146"], "the far residue must not be listed"
    assert r[0]["resn"] == "TYR" and r[0]["chain"] == "A"
    assert r[0]["distance_A"] <= r[1]["distance_A"]


def test_each_residue_appears_once_at_its_closest_atom():
    st = st_of([[1.0, 0, 0], [3.0, 0, 0]], ["A_145", "A_145"], ["TYR", "TYR"], ["A", "A"])
    r = SR.lining_residues(np.array([[0.0, 0, 0]]), st, radius=5.0)
    assert len(r) == 1 and r[0]["distance_A"] == pytest.approx(1.0)


# --- the whole site ------------------------------------------------------------------------------------------------

def test_a_site_carries_everything_a_user_acts_on():
    st = st_of(protein(6.0), ["A_1", "A_2", "A_3", "A_4"], ["TYR"] * 4, ["A"] * 4)
    s = SR.build_site(1, 1, 0.83, grid(4), [1.5, 1.5, 1.5], st,
                      hot_logits=np.zeros((64, 7)), prop_logits=np.zeros(14), caveats=["apo structure"])
    for k in ("id", "rank", "score", "score_type", "center", "docking_box", "volume_A3",
              "max_inscribed_radius_A", "residues", "pharmacophore", "hotspots", "ligand_type", "caveats"):
        assert k in s, k
    assert s["score_type"] == "calibrated_probability"
    assert s["caveats"] == ["apo structure"]


def test_the_score_is_passed_through_untransformed():
    # Any transform applied here would be invisible to whoever fitted the calibration.
    st = st_of(protein(), ["A_1"] * 4, ["TYR"] * 4, ["A"] * 4)
    assert SR.build_site(1, 1, 0.4567, grid(2), [0, 0, 0], st)["score"] == pytest.approx(0.457)


def test_a_report_names_its_schema():
    r = SR.build_report([], dict(pdb_id="1ABC"), dict(version="x"))
    assert r["schema"] == SR.SCHEMA and r["sites"] == []
