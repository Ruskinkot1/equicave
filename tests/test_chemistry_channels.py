"""Metals and the screened-Coulomb field: the two chemical inputs the network did not have.

Both are additions rather than changes, so what these tests pin is the part that would fail silently. A metal that
the parser never read costs nothing visible -- the training run succeeds and the channel is zero everywhere -- and
a field computed in the wrong frame costs equivariance, which no loss curve would show either. So: that the ions
are actually found and weighted by occupancy, that the charges sum to the integer charge of the group, that the
potential is invariant and the field rotates, and that enabling either flag really widens the probe features.
"""
import pathlib

import numpy as np
import pytest

from equicave import pocket_features as pf, structure


def pdb(lines):
    return "\n".join(lines) + "\nEND\n"


def metal_file(tmp_path, body):
    f = tmp_path / "m.pdb"; f.write_text(pdb(body)); return f


ATOM_CA = "ATOM      1  CA  ALA A   1      10.000  10.000  10.000  1.00 20.00           C"
ZN_FULL = "HETATM    2 ZN    ZN A 100       0.000   0.000   0.000  1.00 15.00          ZN"
ZN_PART = "HETATM    3 ZN    ZN A 101       4.000   0.000   0.000  0.40 15.00          ZN"
MG_FULL = "HETATM    4 MG    MG A 102       0.000   5.000   0.000  1.00 15.00          MG"
WATER = "HETATM    5  O   HOH A 200       1.000   1.000   1.000  1.00 15.00           O"
FE_HEM = "HETATM    6 FE   HEM A 300       0.000   0.000   6.000  1.00 15.00          FE"


def test_read_metals_finds_ions_and_cofactor_metals(tmp_path):
    m = structure.read_metals(metal_file(tmp_path, [ATOM_CA, ZN_FULL, ZN_PART, MG_FULL, WATER, FE_HEM]))
    assert sorted(m["element"].tolist()) == ["FE", "MG", "ZN", "ZN"]
    assert "HEM" in m["comp"].tolist()                      # a haem iron is chemistry to a cavity like any other
    assert pytest.approx(sorted(m["occupancy"].tolist())) == [0.4, 1.0, 1.0, 1.0]


def test_read_metals_skips_the_polymer_and_waters(tmp_path):
    m = structure.read_metals(metal_file(tmp_path, [ATOM_CA, WATER]))
    assert len(m["xyz"]) == 0 and m["xyz"].shape == (0, 3)


def test_read_metals_occupancy_floor(tmp_path):
    f = metal_file(tmp_path, [ZN_FULL, ZN_PART])
    assert len(structure.read_metals(f, min_occupancy=0.5)["xyz"]) == 1


def test_read_pdb_does_not_see_metals(tmp_path):
    """The reason the channel is worth adding at all: today every ion is dropped before featurisation."""
    st = structure.read_pdb(metal_file(tmp_path, [ATOM_CA, ZN_FULL, MG_FULL]))
    assert st["element"].tolist() == ["C"]


def test_point_metal_distance_and_occupancy_weight(tmp_path):
    m = structure.read_metals(metal_file(tmp_path, [ZN_FULL, ZN_PART, MG_FULL]))
    d, w = pf.point_metal(np.zeros((1, 3)), m, count_radius=8.0)
    names = [n for n, _ in pf.METAL_GROUPS]
    t, ae = names.index("transition"), names.index("alkaline_earth")
    assert d[0, t] == pytest.approx(0.0) and d[0, ae] == pytest.approx(5.0)
    assert w[0, t] == pytest.approx(1.4)                    # 1.0 + 0.4, not 2: occupancy is the confidence
    assert w[0, names.index("alkali")] == 0.0


def test_point_metal_without_metals_is_capped_not_zero(tmp_path):
    d, w = pf.point_metal(np.zeros((2, 3)), structure.read_metals(metal_file(tmp_path, [ATOM_CA])), count_radius=8.0)
    assert (d == 16.0).all() and (w == 0.0).all()           # "far away", not "at distance zero"


# --- charges and the field ---------------------------------------------------------------------------------------

def st_res(resname, atoms, base=(0.0, 0.0, 0.0)):
    n = len(atoms)
    xyz = np.array(base, float) + np.arange(n)[:, None] * np.array([1.0, 0.0, 0.0])
    return dict(xyz=xyz, element=np.array(["N"] * n), resname=np.array([resname] * n),
                resid=np.array([f"A_1"] * n), chain=np.array(["A"] * n), atom=np.array(atoms),
                backbone=np.zeros(n, bool), bfactor=np.zeros(n))


def test_formal_charges_sum_to_the_group_charge():
    arg = pf.formal_charges(st_res("ARG", ["NE", "NH1", "NH2"]))
    asp = pf.formal_charges(st_res("ASP", ["OD1", "OD2"]))
    assert arg.sum() == pytest.approx(1.0) and np.allclose(arg, 1 / 3)
    assert asp.sum() == pytest.approx(-1.0) and np.allclose(asp, -0.5)


def test_formal_charges_neutral_residue_is_zero():
    assert np.abs(pf.formal_charges(st_res("ALA", ["CB"]))).sum() == 0.0


def test_point_field_sign_follows_the_charge():
    p = np.array([[0.0, 4.0, 0.0]])
    pos, _ = pf.point_field(p, st_res("ARG", ["NE", "NH1", "NH2"]))
    neg, _ = pf.point_field(p, st_res("ASP", ["OD1", "OD2"]))
    assert pos[0] > 0 and neg[0] < 0


def test_point_field_screening_decays_faster_than_coulomb():
    st = st_res("ARG", ["NE", "NH1", "NH2"])
    near, _ = pf.point_field(np.array([[0.0, 3.0, 0.0]]), st)
    far, _ = pf.point_field(np.array([[0.0, 12.0, 0.0]]), st)
    assert far[0] < near[0] * (3.0 / 12.0)                  # 1/r alone would give exactly the ratio


def test_point_field_is_invariant_and_the_field_rotates():
    rng = np.random.default_rng(0)
    st = dict(st_res("ARG", ["NE", "NH1", "NH2"]))
    st["xyz"] = rng.normal(size=(3, 3)) * 3.0
    pts = rng.normal(size=(5, 3)) * 4.0
    R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    R = R * np.sign(np.linalg.det(R))                       # a rotation, not a reflection
    pot, fld = pf.point_field(pts, st)
    st_r = dict(st); st_r["xyz"] = st["xyz"] @ R.T
    pot_r, fld_r = pf.point_field(pts @ R.T, st_r)
    assert np.allclose(pot, pot_r, atol=1e-6)
    assert np.allclose(fld @ R.T, fld_r, atol=1e-6)
    assert not np.allclose(fld, fld_r)                      # the test would pass vacuously on a zero field


def test_point_field_with_no_charged_group_is_zero():
    pot, fld = pf.point_field(np.zeros((2, 3)), st_res("ALA", ["CB"]))
    assert (pot == 0).all() and (fld == 0).all()


# --- the featuriser ----------------------------------------------------------------------------------------------

def charged_pocket_pdb(tmp_path):
    """A pocketed ball whose residues carry real ionisable atom names, plus one zinc inside the cavity."""
    from tests.conftest import ball_with_pocket
    prot = ball_with_pocket(radius=11.0, pocket_center=(0, 0, 5.5), pocket_r=4.0, spacing=1.9)
    res = [("ALA", "CA", "C"), ("ARG", "NH1", "N"), ("ASP", "OD1", "O"), ("LEU", "CA", "C")]
    lines = []
    for i, (x, y, z) in enumerate(prot):
        rn, an, el = res[i % 4]
        lines.append(f"ATOM  {i + 1:5d}  {an:<3s} {rn} A{i + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 10.00           {el}")
    lines.append("HETATM 9001 ZN    ZN A 900       0.000   0.000   5.500  0.70 10.00          ZN")
    f = tmp_path / "charged.pdb"; f.write_text(pdb(lines)); return f


def test_flags_widen_the_probe_features_and_fill_a_vector_slot(tmp_path):
    pytest.importorskip("torch")
    from training.pockets import data as D
    f = charged_pocket_pdb(tmp_path)
    kw = dict(n_probe=48, n_surf=24)
    base = D.featurize(f, None, None, **kw)
    both = D.featurize(f, None, None, probe_metal=True, probe_electrostatic=True, **kw)
    assert base is not None and both is not None
    assert both["feat_probe"].shape[1] == base["feat_probe"].shape[1] + 2 * len(pf.METAL_GROUPS) + 2
    assert both["vec0"].shape == base["vec0"].shape          # the field fills a slot that was zeros
    n_res, n_probe = both["n_res"], both["n_probe"]
    fld = both["vec0"][n_res:n_res + n_probe, 2]
    assert np.isfinite(both["feat_probe"]).all() and np.isfinite(fld).all()
    norms = np.linalg.norm(fld, axis=1)
    assert (norms < 1.001).all() and (norms > 0.99).mean() > 0.9   # unit directions: a strong field cannot dominate
    md = both["feat_probe"][:, base["feat_probe"].shape[1]:base["feat_probe"].shape[1] + len(pf.METAL_GROUPS)]
    assert md.min() < 1.0                                    # the zinc in the cavity is actually seen
    assert np.allclose(base["vec0"][n_res:n_res + n_probe, 2], 0.0)


def test_cache_signature_pins_the_new_flags():
    """Both flags change the arrays, and the cache is keyed by PDB id, so a missing key means a run that silently
    reuses features from the arm it is being compared against."""
    import inspect
    from training.pockets import data as D
    src = inspect.getsource(D.build_cache)
    sig = src.split("sig = dict(")[1].split("sig_file")[0]
    assert "probe_metal=probe_metal" in sig and "probe_electrostatic=probe_electrostatic" in sig
    for name in ("probe_metal", "probe_electrostatic"):
        assert name in inspect.signature(D.build_cache).parameters
        assert name in inspect.signature(D.featurize).parameters


def test_the_arms_exist_and_are_additions():
    import pathlib, yaml
    ab = yaml.safe_load(pathlib.Path("training/configs/ablations.yaml").read_text())["ablations"]
    net = yaml.safe_load(pathlib.Path("training/configs/pockets_net.yaml").read_text())["data"]
    assert net["probe_metal"] is False and net["probe_electrostatic"] is False     # `full` is still the reference
    assert ab["probe_metal"] == {"data.probe_metal": True}
    assert ab["probe_electrostatic"] == {"data.probe_electrostatic": True}
    assert ab["probe_chemistry"] == {"data.probe_metal": True, "data.probe_electrostatic": True}


# --- conservation ------------------------------------------------------------------------------------------------

def test_point_conservation_weights_by_proximity_and_reports_coverage():
    xyz = np.array([[0.0, 0, 0], [0, 0, 2.0], [10.0, 0, 0]])
    res_of_atom = np.array([0, 1, 2])
    scores = np.array([1.0, 0.0, np.nan])                   # the third residue is a gap in the alignment
    mean, mx, cov = pf.point_conservation(np.array([[0.0, 0, 1.0], [10.0, 0, 1.0]]), xyz, res_of_atom, scores)
    assert mean[0] == pytest.approx(0.5) and mx[0] == pytest.approx(1.0)
    assert cov[0] == pytest.approx(1.0)
    assert cov[1] == pytest.approx(0.0) and mean[1] == 0.0  # lining present but unscored: coverage says so
    near, _, _ = pf.point_conservation(np.array([[0.0, 0, 0.5]]), xyz, res_of_atom, scores)
    assert near[0] > mean[0]                                # closer to the conserved residue, so weighted higher


def test_point_conservation_gap_is_dropped_not_imputed():
    xyz = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    mean, mx, cov = pf.point_conservation(np.zeros((1, 3)), xyz, np.array([0, 1]), np.array([0.8, np.nan]))
    assert mean[0] == pytest.approx(0.8) and cov[0] == pytest.approx(0.5)


def test_conservation_channel_and_its_loud_failure(tmp_path):
    pytest.importorskip("torch")
    from training.pockets import data as D
    from equicave import structure
    f = charged_pocket_pdb(tmp_path)
    rt = structure.residue_table(structure.read_pdb(f))
    d = tmp_path / "cons"; d.mkdir()
    (d / "charged.txt").write_text("".join(f"{i} {i % 5 / 4:.3f}\n" for i in range(len(rt["resid"]))))
    kw = dict(n_probe=48, n_surf=24)
    base = D.featurize(f, None, None, **kw)
    got = D.featurize(f, None, None, probe_conservation=True, conservation_dir=str(d), **kw)
    assert got["feat_probe"].shape[1] == base["feat_probe"].shape[1] + 3
    with pytest.raises(FileNotFoundError):
        D.featurize(f, None, None, probe_conservation=True, conservation_dir=str(tmp_path / "absent"), **kw)


def test_build_cache_refuses_a_partial_conservation_set(tmp_path):
    """The failure that matters: files for some structures, so the arm would train on a different protein set."""
    pytest.importorskip("torch")
    from training.pockets import data as D
    man = tmp_path / "m.csv"
    man.write_text("pdb,cluster30,fold,ligands\nA,c0,0,\"[[\"\"LIG\"\"]]\"\nB,c1,1,\"[[\"\"LIG\"\"]]\"\n")
    d = tmp_path / "cons"; d.mkdir(); (d / "A.txt").write_text("0 1.0\n")
    with pytest.raises(FileNotFoundError, match="1 of 2"):
        D.build_cache(man, tmp_path, tmp_path / "cache", None, probe_conservation=True, conservation_dir=str(d))


def test_protrusion_adds_three_long_range_counts(tmp_path):
    """The one addition with a measurement of ours behind it; the test pins that it is actually long-ranged."""
    pytest.importorskip("torch")
    from training.pockets import data as D
    f = charged_pocket_pdb(tmp_path)
    kw = dict(n_probe=48, n_surf=24)
    base = D.featurize(f, None, None, **kw)
    got = D.featurize(f, None, None, probe_protrusion=True, **kw)
    k = base["feat_probe"].shape[1]
    assert got["feat_probe"].shape[1] == k + 3
    long = got["feat_probe"][:, k:]
    assert np.isfinite(long).all() and (long > 0).all()
    assert (long[:, 0] <= long[:, 1] + 1e-6).all() and (long[:, 1] <= long[:, 2] + 1e-6).all()   # 10 < 12 < 15 A
    assert (long[:, 2] > long[:, 0]).any()                      # the radii are not all saturated on this structure


def test_protrusion_is_in_the_cache_signature():
    import inspect
    from training.pockets import data as D
    assert "probe_protrusion=probe_protrusion" in inspect.getsource(D.build_cache).split("sig_file")[0]
    assert "probe_protrusion" in inspect.signature(D.featurize).parameters


# --- calibration and stratified reporting --------------------------------------------------------------------------

def test_temperature_recovers_a_known_overconfidence():
    from equicave import calibration as CAL
    from equicave.metrics import ece
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2, 200_000)
    y = rng.random(200_000) < 1 / (1 + np.exp(-z))
    logit = z * 2.5                                          # the model is over-confident by a factor of 2.5
    t = CAL.temperature(logit, y)
    assert 2.3 < t < 2.7 and not CAL.at_bound(t)
    assert ece(y, CAL.apply(logit, t)) < 0.1 * ece(y, CAL.apply(logit, 1.0))


def test_temperature_is_one_when_there_is_nothing_to_fit():
    from equicave import calibration as CAL
    assert CAL.temperature(np.zeros(10), np.zeros(10)) == 1.0      # one class only
    assert CAL.temperature(np.zeros(0), np.zeros(0)) == 1.0


def test_at_bound_flags_a_boundary_value():
    from equicave import calibration as CAL
    assert CAL.at_bound(20.0) and CAL.at_bound(0.05) and not CAL.at_bound(1.4)


def test_split_ids_is_deterministic_and_disjoint():
    from equicave import calibration as CAL
    ids = [f"p{i}" for i in range(500)]
    a1, b1 = CAL.split_ids(ids); a2, b2 = CAL.split_ids(list(reversed(ids)))
    assert a1 == a2 and b1 == b2                              # not a function of file order
    assert not (a1 & b1) and len(a1 | b1) == len(ids)
    assert 0.4 < len(a1) / len(ids) < 0.6


def test_sumsq_ranking_is_not_invariant_under_temperature():
    """The premise of the whole calibration measurement: a monotone map changes a sum-of-squares ordering."""
    from equicave import calibration as CAL
    pos = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0],            # site A: two confident probes
                    [20.0, 0, 0], [21.0, 0, 0], [22.0, 0, 0], [23.0, 0, 0], [24.0, 0, 0]])  # site B: many lukewarm
    occ_logit = np.array([3.0, 3.0, 3.0, -0.5, -0.5, -0.5, -0.5, -0.5])
    conf = np.array([1.0, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1])
    center = pos.copy()
    sharp = CAL.sumsq_sites(CAL.apply(occ_logit, 0.2), pos, conf, center)[0][0]
    soft = CAL.sumsq_sites(CAL.apply(occ_logit, 8.0), pos, conf, center)[0][0]
    assert not np.allclose(sharp, soft)                       # the winning site changes with the temperature
    assert sharp[0] < 10.0 and soft[0] > 10.0                 # sharpening favours the few-confident site


def test_by_group_reports_counts_and_skips_small_groups():
    import pandas as pd
    from equicave import metrics as MT
    rows = []
    for i in range(30):
        for k in range(3):
            rows.append(dict(pdb=f"p{i}", cluster30=f"c{i}", n_sites=1, rank=k + 1,
                             score=1.0 - k * 0.1, label=int(k == 0 and i % 2 == 0)))
    rr = pd.DataFrame(rows)
    out = MT.by_group(rr, {"big": [f"p{i}" for i in range(30)], "small": ["p0", "p1"]})
    assert out["small"] == {"n": 2}                           # below min_n: count only, no claimable metric
    assert out["big"]["n"] == 30 and out["big"]["top1"] == pytest.approx(0.5)


def test_evaluate_reports_both_blocks(tmp_path):
    torch = pytest.importorskip("torch")
    import tests.test_training as T
    from training.pockets import net_task as NT
    files, cfg = T._tiny_cache(tmp_path, n=8)
    r = NT.train_one(cfg, files[:4], files[4:], torch.device("cpu"), tmp_path, log=lambda s: None)
    ev = NT.evaluate(r["model"], files, torch.device("cpu"), cfg, log=lambda s: None)
    assert "net_sites_by_class" in ev and "net_sites_dcc4_by_class" in ev
    c = ev["calibration"]
    assert c["n_fit"] + c["n_report"] == 8 and c["n_fit"] > 0 and c["n_report"] > 0
    assert "occ_ece_raw" in c and "occ_ece_calibrated" in c and "temperature_at_bound" in c
    assert len(c["reliability_calibrated"]) == 10
    cfg2 = dict(cfg); cfg2["eval"] = {"calibrate": False}
    assert "calibration" not in NT.evaluate(r["model"], files, torch.device("cpu"), cfg2, log=lambda s: None)


def test_composition_exposes_a_site_count_shift():
    """The confound this exists to surface: two subsets with the same accuracy but a different problem mix."""
    import pandas as pd
    from equicave import metrics as MT
    rows = [dict(pdb=f"a{i}", cluster30=f"c{i}", n_sites=1) for i in range(8)]
    rows += [dict(pdb=f"b{i}", cluster30=f"c{i}", n_sites=4) for i in range(2)]
    rows += [dict(pdb="a0", cluster30="c0", n_sites=1)]          # a second candidate row for one structure
    c = MT.composition(pd.DataFrame(rows))
    assert c["structures"] == 10 and c["clusters"] == 8          # de-duplicated by pdb, clusters shared on purpose
    assert c["single_site_fraction"] == pytest.approx(0.8) and c["mean_sites"] == pytest.approx(1.6)
    assert c["site_counts"] == {1: 8, 4: 2}
    assert MT.composition(pd.DataFrame([dict(pdb="x")]))["mean_sites"] is None   # no site column: not invented


def test_evaluate_declares_the_hard_novelty_subset():
    """A script, so what is pinned is that the flag and the subset exist and the corpus path is not hardcoded."""
    src = (pathlib.Path(__file__).resolve().parents[1] / "scripts/eval/evaluate.py").read_text()
    assert "--novel-corpus" in src and '"novel to all"' in src
    assert "their_similar" in src and "homology_corpus_clusters.json" in src
    assert "the 'novel to all' subset is skipped" in src        # missing corpus says so instead of reporting zero


# --- the featurisation reaches inference ---------------------------------------------------------------------------

def test_every_width_changing_flag_is_carried_to_inference():
    """A flag that changes `feat_probe` and is not in FEATURISE_KEYS is a model that cannot be used to predict.

    This is the guard for a bug that was live: the three inference paths passed n_probe, n_surf and k_scale only,
    so a checkpoint trained with any other featurisation was fed arrays of a different width -- or, for probe
    placement, of the same width and the wrong distribution, which is silent.
    """
    import inspect
    from training.pockets import data as D
    cache_only = {"manifest_csv", "pdb_dir", "out_dir", "esm_name", "limit", "device", "log", "n_probe", "n_surf",
                  "druglike_only", "probe_sampling", "point_model_tag"}
    expected = set(inspect.signature(D.build_cache).parameters) - cache_only
    assert expected <= set(D.FEATURISE_KEYS), sorted(expected - set(D.FEATURISE_KEYS))


def test_featurisation_kwargs_round_trips_a_config():
    from training.pockets import data as D
    dcfg = dict(n_probe=768, n_surf=512, k_scale=1.0, residue_chemistry=False, probe_potential=True,
                probe_metal=True, probe_electrostatic=False, probe_protrusion=True, probe_conservation=False,
                conservation_dir="x", probe_sampling="tiered", probe_ligandable_frac=0.5, esm=None)
    kw = D.featurisation_kwargs(dcfg)
    assert kw["residue_chemistry"] is False and kw["probe_metal"] is True and kw["probe_protrusion"] is True
    assert "n_probe" not in kw and "esm" not in kw              # passed positionally / separately
    assert "probe_ligandable_frac" not in kw                    # meaningless for tiered placement
    assert "probe_model" not in kw
    import inspect
    sig = inspect.signature(D.featurize).parameters
    assert set(kw) <= set(sig), sorted(set(kw) - set(sig))      # every key is a real featurize argument


def test_ligandable_placement_without_a_point_model_says_so():
    from training.pockets import data as D
    said = []
    kw = D.featurisation_kwargs(dict(probe_sampling="ligandable", probe_ligandable_frac=0.5), log=said.append)
    assert "probe_model" not in kw and said and "NOT what it was trained on" in said[0]
    kw2 = D.featurisation_kwargs(dict(probe_sampling="ligandable"), point_model="booster")
    assert kw2["probe_model"] == "booster"


def test_metal_channel_excludes_the_ligands_being_predicted(tmp_path):
    """The leakage path: on a benchmark where an ion is itself a site, an unfiltered metal channel is the label."""
    f = metal_file(tmp_path, [ATOM_CA, ZN_FULL, FE_HEM])
    assert len(structure.read_metals(f)["xyz"]) == 2
    assert structure.read_metals(f, exclude_comps={"ZN"})["comp"].tolist() == ["HEM"]
    assert len(structure.read_metals(f, exclude_comps={"ZN", "HEM"})["xyz"]) == 0
    import inspect
    from training.pockets import data as D
    src = inspect.getsource(D.featurize)
    assert "exclude_comps=lig_codes" in src


# --- the cache signature must not reject a cache it agrees with ----------------------------------------------------

def _write_sig(d, **over):
    import json
    base = dict(n_probe=8, n_surf=4, k_scale=1.0, druglike_only=False, require_interaction=True,
                residue_chemistry=True, probe_potential=True, probe_sampling="tiered", point_model_tag="",
                probe_ligandable_frac=None, probe_metal=False, probe_electrostatic=False,
                probe_conservation=False, probe_protrusion=False, conservation_dir=None, esm=None)
    from equicave import pocket_features as pf
    base["geometry"] = pf.geometry()
    base.update(over)
    (d / ".featurisation.json").write_text(json.dumps(base, indent=1, sort_keys=True))
    return base


def test_a_signature_predating_a_new_flag_is_still_reused(tmp_path):
    """The regression this guards: adding an off-by-default flag killed every arm of the grid at startup.

    A cache written before `probe_metal` existed holds exactly the arrays a cache written with
    `probe_metal=False` would hold, so the absent key must compare equal to False rather than to nothing.
    """
    import json
    from training.pockets import data as D
    cache = tmp_path / "cache"; cache.mkdir()
    sig = _write_sig(cache)
    for k in ("probe_metal", "probe_electrostatic", "probe_conservation", "probe_protrusion", "conservation_dir"):
        sig.pop(k, None)
    (cache / ".featurisation.json").write_text(json.dumps(sig, indent=1, sort_keys=True))
    man = tmp_path / "m.csv"; man.write_text("pdb,cluster30,fold,ligands\n")      # no rows: only the guard runs
    said = []
    D.build_cache(man, tmp_path, cache, None, n_probe=8, n_surf=4, log=said.append)
    assert any("predates" in s for s in said)
    now = json.loads((cache / ".featurisation.json").read_text())
    assert now["probe_metal"] is False and "probe_protrusion" in now          # upgraded, so the next check is exact


def test_a_signature_predating_a_flag_still_refuses_when_the_flag_is_on(tmp_path):
    import json
    from training.pockets import data as D
    cache = tmp_path / "cache"; cache.mkdir()
    sig = _write_sig(cache); sig.pop("probe_protrusion")
    (cache / ".featurisation.json").write_text(json.dumps(sig, indent=1, sort_keys=True))
    man = tmp_path / "m.csv"; man.write_text("pdb,cluster30,fold,ligands\n")
    with pytest.raises(RuntimeError, match="probe_protrusion"):
        D.build_cache(man, tmp_path, cache, None, n_probe=8, n_surf=4, probe_protrusion=True, log=lambda s: None)


def test_a_real_featurisation_difference_is_still_refused(tmp_path):
    from training.pockets import data as D
    cache = tmp_path / "cache"; cache.mkdir()
    _write_sig(cache, residue_chemistry=True)
    man = tmp_path / "m.csv"; man.write_text("pdb,cluster30,fold,ligands\n")
    with pytest.raises(RuntimeError, match="residue_chemistry"):
        D.build_cache(man, tmp_path, cache, None, n_probe=8, n_surf=4, residue_chemistry=False, log=lambda s: None)


def test_every_new_signature_key_has_a_neutral_default():
    """A new off-by-default flag that is not in SIG_DEFAULTS repeats the outage, so the list is checked here."""
    import inspect, json
    from training.pockets import data as D
    src = inspect.getsource(D.build_cache)
    body = src.split("sig = dict(")[1].split("sig_file")[0]
    keys = {k.strip() for k in __import__("re").findall(r"(\w+)=", body)}
    for name in ("probe_metal", "probe_electrostatic", "probe_conservation", "probe_protrusion", "conservation_dir"):
        assert name in keys and name in D.SIG_DEFAULTS
