import numpy as np
from scipy.spatial import cKDTree

from equicave import pockets as pk, structure
from tests.conftest import shell


def test_buried_counts_rays():
    xyz = shell()
    tp = cKDTree(xyz)
    inside = pk._buried(np.array([[0, 0, 0.0]]), tp)[0]
    outside = pk._buried(np.array([[40, 40, 40.0]]), tp)[0]
    assert inside > outside and outside == 0 and inside <= pk.N_RAYS


def test_cavity_box_follows_walls():
    xyz = shell()
    ctr, box, pts = pk.cavity_box(np.array([0, 0, 0.0]), xyz, radius=20.0)
    assert max(box) < 30 and len(pts) > 10
    assert all(b >= pk.MIN_SIDE for b in box)


def test_shape_axes_sorted():
    pts = np.random.default_rng(0).normal(size=(100, 3)) * np.array([5, 2, 1])
    ax = pk.shape_axes(pts)
    assert ax[0] >= ax[1] >= ax[2] > 0


def test_sas_points_sit_outside_atoms():
    xyz = shell(n_ring=12, zs=(0,))
    pts, owner = pk.sas_points(xyz, n_sphere=40)
    d = cKDTree(xyz).query(pts)[0]
    assert len(pts) and (d >= 1.7 + 1.4 - 1e-6).all() and owner.max() < len(xyz)


def test_dca_dcc():
    lig = np.array([[0, 0, 0], [10, 0, 0.0]])
    assert pk.dca([9, 0, 0], lig) == 1.0 and pk.dcc([5, 0, 0], lig) == 0.0


def test_pdb_roundtrip(tmp_path):
    xyz = shell(n_ring=8, zs=(0, 4))
    p = structure.write_pdb(tmp_path / "x.pdb", xyz, element="C", resname="ALA", record="ATOM")
    st = structure.read_pdb(p)
    assert np.allclose(st["xyz"], xyz, atol=1e-3) and len(structure.residue_table(st)["resid"]) == len(xyz)
    structure.write_pdb(tmp_path / "l.pdb", np.random.default_rng(0).normal(size=(10, 3)), resname="LIG", resseq=1)
    text = (tmp_path / "x.pdb").read_text().replace("END\n", "") + (tmp_path / "l.pdb").read_text()
    (tmp_path / "c.pdb").write_text(text)
    ligs = structure.read_ligands(tmp_path / "c.pdb")
    assert len(ligs) == 1 and ligs[0]["comp"] == "LIG" and len(ligs[0]["xyz"]) == 10
