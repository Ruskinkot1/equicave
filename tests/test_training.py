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


def test_featurize_cache_train_evaluate(tmp_path):
    cache = tmp_path / "cache"; cache.mkdir()
    files = []
    for k in range(4):
        p = _synthetic(tmp_path, f"S{k}", k)
        d = D.featurize(p, {"LIG"}, None, n_probe=64, n_surf=48, seed=k)
        assert d is not None and "y_res" in d and d["y_occ"].sum() > 0 and d["site_centers"].shape == (1, 3)
        d["cluster30"] = f"c{k}"; d["fold"] = k % 2
        D.save(d, cache / f"S{k}.npz"); files.append(cache / f"S{k}.npz")
    cfg = dict(model=dict(dim=16, layers=1, heads=2, n_rbf=8, cutoff=8.0, equivariant=True, use_vectors=True, use_tensors=True, chiral=True,
                          use_surface=True, use_probes=True, dropout=0.0),
               loss=dict(w_res=1, w_occ=1, w_center=0.5, w_conf=0.5, w_hot=1, w_prop=0.5, occ_pos_weight=2.0, res_pos_weight=2.0),
               optim=dict(lr=1e-3, weight_decay=0.0, epochs=2, accumulate=1, warmup_epochs=0, ema=0.5, patience=5, seed=0, rotate=True, amp="none"))
    r = NT.train_one(cfg, files[:2], files[2:], torch.device("cpu"), tmp_path, log=lambda s: None)
    assert len(r["history"]) == 2 and (tmp_path / "model.pt").exists()
    ev = NT.evaluate(r["model"], files[2:], torch.device("cpu"), cfg)
    assert set(ev) >= {"res_ap", "occ_ap", "hot_ap_per_class", "net_sites", "val_score"}
    m, _ = NT.load_model(tmp_path / "model.pt", torch.device("cpu"))
    d = D.load(files[2]); out = m(D.to_torch(d))
    feats = NT.net_features(out, d["pos"][d["n_res"]:d["n_res"] + d["n_probe"]], d["cand_centers"][:3])
    assert len(feats) == min(3, len(d["cand_centers"])) and all(0 <= f["net_seg"] <= 1 for f in feats)
