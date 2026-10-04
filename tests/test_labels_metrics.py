import numpy as np
import pandas as pd

from equicave import labels, metrics as M


def _lig(xyz, el=None, comp="LIG"):
    xyz = np.asarray(xyz, float)
    return dict(comp=comp, chain="A", resseq="1", xyz=xyz, element=np.array(el or ["C"] * len(xyz)), atom=np.array([f"C{i}" for i in range(len(xyz))]))


def test_group_sites_and_centres():
    ligs = [_lig([[0, 0, 0], [1, 0, 0]]), _lig([[2, 0, 0], [3, 0, 0]]), _lig([[30, 0, 0], [31, 0, 0]])]
    sites = labels.group_sites(ligs)
    assert sites == [[0, 1], [2]]
    c = labels.site_centres(ligs, sites)
    assert c.shape == (2, 3) and abs(c[0, 0] - 1.5) < 1e-9


def test_point_labels_fallback_without_ccd():
    ligs = [_lig([[0, 0, 0], [3, 0, 0]], el=["C", "O"])]
    pts = np.array([[0, 0, 0.5], [3, 0, 0.5], [10, 0, 0]])
    occ, hot, prox = labels.point_labels(pts, ligs, {}, require_interaction=False)
    assert occ.tolist() == [True, True, False]
    assert hot[0, 0] and hot[1, 3] and not hot[2].any()
    assert (hot == prox).all()            # without a receptor the two targets coincide
    occ, hot, prox = labels.point_labels(pts, [], {})
    assert not occ.any() and not hot.any() and hot.shape == (3, len(labels.HOTSPOT_CLASSES))


def _receptor(atoms):
    """A tiny receptor: list of (atom name, residue name, xyz)."""
    import numpy as _np
    return dict(xyz=_np.array([a[2] for a in atoms], float), atom=_np.array([a[0] for a in atoms]),
                resname=_np.array([a[1] for a in atoms]), element=_np.array([a[0][0] for a in atoms]),
                backbone=_np.array([a[0] in ("N", "CA", "C", "O") for a in atoms]),
                resid=_np.array([f"A_{i}" for i in range(len(atoms))]), chain=_np.array(["A"] * len(atoms)),
                bfactor=_np.zeros(len(atoms)))


def test_protein_atom_typing():
    st = _receptor([("NZ", "LYS", (0, 0, 0)), ("OD1", "ASP", (5, 0, 0)), ("CZ", "PHE", (10, 0, 0)),
                    ("CB", "LEU", (15, 0, 0)), ("N", "ALA", (20, 0, 0)), ("O", "ALA", (25, 0, 0))])
    t = labels.protein_atom_types(st)
    assert t["cation"][0] and t["donor"][0] and t["anion"][1] and t["acceptor"][1]
    assert t["aromatic"][2] and t["hydrophobic"][3] and t["donor"][4] and t["acceptor"][5]


def test_interaction_validation_drops_unpartnered_atoms():
    st = _receptor([("OD1", "ASP", (0, 0, 3.0)), ("CB", "LEU", (6, 0, 3.0))])
    # atom 0: a donor 3 A from the Asp acceptor -> validated. atom 1: a carbon 3 A from a Leu CB -> validated.
    # atom 2: a donor far from everything -> dropped.
    lig = _lig([[0, 0, 0], [6, 0, 0], [40, 0, 0]], el=["N", "C", "N"])
    cls = labels.ligand_atom_classes(lig, None)
    inter = labels.interaction_classes(lig, cls, st)
    hbd = labels.HOTSPOT_CLASSES.index("hbd"); hydro = labels.HOTSPOT_CLASSES.index("hydrophobic_c")
    assert cls[0, hbd] and inter[0, hbd]
    assert cls[1, hydro] and inter[1, hydro]
    assert cls[2, hbd] and not inter[2, hbd]
    assert inter.sum() < cls.sum()


def test_require_interaction_changes_the_field():
    st = _receptor([("OD1", "ASP", (0, 0, 3.0))])
    ligs = [_lig([[0, 0, 0], [40, 0, 0]], el=["N", "N"])]
    pts = np.array([[0, 0, 0.5], [40, 0, 0.5]])
    _, hot, prox = labels.point_labels(pts, ligs, {}, st=st, require_interaction=True)
    hbd = labels.HOTSPOT_CLASSES.index("hbd")
    assert prox[0, hbd] and prox[1, hbd]           # both atoms are donors by chemistry
    assert hot[0, hbd] and not hot[1, hbd]         # only one of them actually donates to the receptor


def test_druglike_filter():
    from equicave import ccd
    atp = ccd.load("ATP"); sti = ccd.load("STI")
    if atp is None or sti is None:
        import pytest as _p; _p.skip("CCD not cached")
    big = _lig(np.zeros((31, 3)))
    assert not labels.druglike_ligand(big, atp)                 # nucleotide class is excluded
    assert labels.druglike_ligand(_lig(np.zeros((37, 3))), sti)  # imatinib-like: rings, right size
    assert not labels.druglike_ligand(_lig(np.zeros((4, 3))), None)  # too small whatever it is


def test_site_properties_shapes():
    prot = np.random.default_rng(0).normal(size=(300, 3)) * 8
    ligs = [_lig(np.random.default_rng(1).normal(size=(40, 3)), el=["C"] * 30 + ["O"] * 10)]
    v = labels.site_properties(ligs, {}, prot)
    assert v.shape == (len(labels.PROPERTY_CLASSES),) and v[labels.PROPERTY_CLASSES.index("size_large")]


def test_ranking_metrics_and_bootstrap():
    df = pd.DataFrame(dict(pdb=["a"] * 3 + ["b"] * 3, cluster30=["x"] * 3 + ["y"] * 3, n_sites=[1] * 6,
                           label=[0, 1, 0, 0, 0, 1], s=[3, 2, 1, 3, 2, 1]))
    per = M.per_structure(df, "s")
    assert per.top1.tolist() == [False, False] and per.top3.tolist() == [True, True]
    assert per.topN2.tolist() == [True, True] and per.first.tolist() == [2, 3]
    m, lo, hi = M.boot_ci(per.top3.to_numpy(float), per.cluster30.to_numpy())
    assert m == 1.0 and lo <= m <= hi
    pb = M.paired_boot(per.top3.to_numpy(float), per.top1.to_numpy(float), per.cluster30.to_numpy())
    assert pb["diff"] == 1.0 and pb["p_le_0"] == 0.0


def test_classification_metrics():
    y = np.array([1, 0, 1, 0, 1, 0]); s = np.array([.9, .1, .8, .3, .6, .4])
    assert M.auroc(y, s) == 1.0 and M.average_precision(y, s) == 1.0
    assert 0 <= M.ece(y, s) <= 1
    assert M.enrichment_at(y, s, 0.5) == 2.0
    obs, p = M.permutation_control(y, s, M.auroc, n=50)
    assert obs == 1.0 and p < 0.2


def test_redundancy_statistic():
    # structure a: three predictions, the first two both hit site 0, the third hits site 1 -> one redundant of three
    df = pd.DataFrame(dict(pdb=["a"] * 3, cluster30=["x"] * 3, n_sites=[2] * 3,
                           label=[1, 1, 1], site_idx=[0, 0, 1], s=[3.0, 2.0, 1.0]))
    r = M.redundancy(df, "s")
    assert r["n_hitting_predictions"] == 3 and r["n_redundant"] == 1
    assert abs(r["fraction"] - 1 / 3) < 1e-9
    # no hits at all: the fraction is undefined, not zero
    import numpy as np
    empty = df.assign(label=0)
    assert np.isnan(M.redundancy(empty, "s")["fraction"])
