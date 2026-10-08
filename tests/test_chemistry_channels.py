"""Metals and the screened-Coulomb field: the two chemical inputs the network did not have.

Both are additions rather than changes, so what these tests pin is the part that would fail silently. A metal that
the parser never read costs nothing visible -- the training run succeeds and the channel is zero everywhere -- and
a field computed in the wrong frame costs equivariance, which no loss curve would show either. So: that the ions
are actually found and weighted by occupancy, that the charges sum to the integer charge of the group, that the
potential is invariant and the field rotates, and that enabling either flag really widens the probe features.
"""
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
