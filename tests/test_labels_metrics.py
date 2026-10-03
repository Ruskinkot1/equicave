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
    occ, hot = labels.point_labels(pts, ligs, {})
    assert occ.tolist() == [True, True, False]
    assert hot[0, 0] and hot[1, 3] and not hot[2].any()


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
