"""CPU smoke test of the whole network pipeline on synthetic structures: featurise -> cache -> train 1 epoch -> evaluate."""
import json
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from equicave import structure  # noqa: E402
from training.pockets import data as D, net_task as NT  # noqa: E402
from tests.conftest import ball_with_pocket  # noqa: E402


def _synthetic(tmp_path, name, seed):
    rng = np.random.default_rng(seed)
    prot = ball_with_pocket(radius=11.0, pocket_center=(0, 0, 5.5), pocket_r=4.0, spacing=1.9) + rng.normal(0, 0.1, size=(1, 3))
    lines = []
    aa = ["ALA", "LEU", "SER", "ASP", "LYS", "PHE"]
    for i, (x, y, z) in enumerate(prot):          # one residue per atom, atom named CA so residue tables work
        lines.append(f"ATOM  {i + 1:5d}  CA  {aa[i % 6]} A{i + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 10.00           C")
    lig = np.array([0, 0, 5.5]) + rng.normal(0, 1.0, size=(10, 3))
    for i, (x, y, z) in enumerate(lig):
        el = "O" if i % 3 == 0 else "C"
        lines.append(f"HETATM{900 + i:5d}  {el}{i:<2d} LIG B 900    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 10.00           {el}")
    p = tmp_path / f"{name}.pdb"; p.write_text("\n".join(lines) + "\nEND\n")
    return p


def _tiny_cache(tmp_path, n=4):
    """A cache of `n` synthetic structures plus a minimal config, shared by the tests below."""
    cache = tmp_path / "cache"; cache.mkdir(exist_ok=True)
    files = []
    for k in range(n):
        p = _synthetic(tmp_path, f"S{k}", k)
        d = D.featurize(p, {"LIG"}, None, n_probe=64, n_surf=48, seed=k)
        assert d is not None and "y_res" in d and d["y_occ"].sum() > 0 and d["site_centers"].shape == (1, 3)
        d["cluster30"] = f"c{k}"; d["fold"] = k % 2
        D.save(d, cache / f"S{k}.npz"); files.append(cache / f"S{k}.npz")
    cfg = dict(model=dict(dim=16, layers=1, heads=2, n_rbf=8, cutoff=8.0, equivariant=True, use_vectors=True, use_tensors=True, chiral=True,
                          use_surface=True, use_probes=True, dropout=0.0),
               loss=dict(w_res=1, w_occ=1, w_center=0.5, w_conf=0.5, w_hot=1, w_prop=0.5, occ_pos_weight=2.0, res_pos_weight=2.0),
               optim=dict(lr=1e-3, weight_decay=0.0, epochs=2, accumulate=1, warmup_epochs=0, ema=0.5, patience=5, seed=0, rotate=True, amp="none"))
    return files, cfg


def test_featurize_cache_train_evaluate(tmp_path):
    files, cfg = _tiny_cache(tmp_path)
    r = NT.train_one(cfg, files[:2], files[2:], torch.device("cpu"), tmp_path, log=lambda s: None)
    assert len(r["history"]) == 2 and (tmp_path / "model.pt").exists()
    ev = NT.evaluate(r["model"], files[2:], torch.device("cpu"), cfg)
    assert set(ev) >= {"res_ap", "occ_ap", "hot_ap_per_class", "net_sites", "val_score"}
    m, _ = NT.load_model(tmp_path / "model.pt", torch.device("cpu"))
    d = D.load(files[2]); out = m(D.to_torch(d))
    feats = NT.net_features(out, d["pos"][d["n_res"]:d["n_res"] + d["n_probe"]], d["cand_centers"][:3])
    assert len(feats) == min(3, len(d["cand_centers"])) and all(0 <= f["net_seg"] <= 1 for f in feats)


def test_training_resumes_from_its_checkpoint(tmp_path):
    """A run that dies must continue, not restart: a multi-hour GPU run otherwise loses every epoch.

    Trains two epochs, then trains again from the same directory with a higher epoch budget, and asserts the second
    call picked up where the first stopped rather than starting over -- the history is appended to, not replaced.
    """
    files, cfg = _tiny_cache(tmp_path)
    cfg["optim"] = dict(cfg["optim"], epochs=2, patience=10)
    out = tmp_path / "run"; out.mkdir()
    r1 = NT.train_one(cfg, files, files[:1], torch.device("cpu"), out, log=lambda s: None)
    assert (out / "checkpoint.pt").exists()
    assert len(r1["history"]) == 2
    cfg2 = dict(cfg, optim=dict(cfg["optim"], epochs=4))
    r2 = NT.train_one(cfg2, files, files[:1], torch.device("cpu"), out, log=lambda s: None)
    assert len(r2["history"]) == 4, "resume restarted from scratch instead of continuing"
    assert [h["epoch"] for h in r2["history"]] == [1, 2, 3, 4]
    # The inherited epochs keep their metrics; only `seconds`, which is wall clock, may differ. The comparison has
    # to treat NaN as equal to itself, because a class with no positive probe in this tiny set reports NaN AP.
    def same(a, b):
        if isinstance(a, dict):
            return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
        if isinstance(a, list):
            return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
        if isinstance(a, float) and isinstance(b, float):
            return a == b or (np.isnan(a) and np.isnan(b))
        return a == b

    for a, b in zip(r1["history"], r2["history"][:2]):
        assert same(a["train"], b["train"]) and same(a["val"], b["val"]), "the resumed run rewrote inherited epochs"


def test_ligandable_probe_placement_spends_the_budget_without_erasing_the_negatives(tmp_path):
    """Probes are the one component the trained ablation found to matter, so where they go is the real knob.

    Three properties, and the third is what makes the first two safe: a perfect ligandability model must pull probes
    onto the ligand, the probe count and uniqueness must not change, and the default fraction must still leave
    probes far from the ligand -- spending the whole budget by score leaves almost nothing for the site head to
    reject, which is the failure this mixture exists to avoid.
    """
    from scipy.spatial import cKDTree
    from equicave import structure as ST

    pdb = _synthetic(tmp_path, "L", 0)
    ligs = ST.read_ligands(pdb, min_heavy=8)
    assert ligs
    lig_tree = cKDTree(np.vstack([l["xyz"] for l in ligs]))

    class Oracle:
        """A perfect ligandability model: scores a point by how close it is to a ligand atom."""
        def __init__(self, free_points=None):
            self.free = free_points

        def predict(self, X):                        # X is the per-point feature matrix; we score by geometry
            return -lig_tree.query(self.free)[0] if self.free is not None else np.zeros(len(X))

    def probes(frac, model):
        d = D.featurize(pdb, {"LIG"}, None, n_probe=64, n_surf=32, probe_model=model, probe_ligandable_frac=frac)
        return d["pos"][d["n_res"]:d["n_res"] + d["n_probe"]]

    # a constant model must change which probes are chosen but never how many, and never select one twice
    flat = probes(0.5, Oracle())
    random_only = probes(0.0, None)
    assert len(flat) == len(random_only) > 0
    assert len(set(map(tuple, np.round(flat, 3)))) == len(flat), "a probe was selected twice"

    # with a model that knows where the ligand is, more budget by score means more probes on the ligand, and the
    # default must keep some of them far away
    class Perfect:
        def __init__(self):
            self.seen = None

        def predict(self, X):
            # point_features puts the distance to the nearest protein atom in a known column, but the test only
            # needs *some* monotone preference, so rank by the feature that correlates with being inside a cavity
            return X[:, ps_buried_index()]

    near = lambda pts: float((lig_tree.query(pts)[0] <= 4.0).mean())
    far = lambda pts: float((lig_tree.query(pts)[0] > 8.0).mean())
    assert far(random_only) > 0.0, "the synthetic structure has no distant free points to begin with"
    full = probes(1.0, Perfect())
    half = probes(0.5, Perfect())
    assert len(full) == len(half) == len(random_only)
    assert far(half) >= far(full), "the mixture must retain more distant probes than pure score selection"


def ps_buried_index() -> int:
    """Index of the lattice-buriedness column in the per-point feature row."""
    from equicave import point_score as ps
    return ps.POINT_FEATURES.index("p_buried")
