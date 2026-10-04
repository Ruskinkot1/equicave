"""Featurisation of one structure for EquiCave-Net and the on-disk cache used by training.

Node types and inputs
  res   one per residue at CA: one-hot residue (21) + 14 amino-acid property scalars (`equicave.residues`) +
        2 pocket-facing cosines + ESM-2 embedding (optional) + [b-factor z, local density]
        initial vectors: N->CA, C->CA, CA->CB(virtual for GLY), CA->side-chain centroid, CA->functional-group centroid
        The last two say which way the residue's chemistry faces, which is what distinguishes an Asp whose
        carboxylate lines the cavity from one pointing at the solvent.
  probe free cavity lattice points (buriedness >= FILL_MIN_BURIED, deep points first), <= n_probe
        scalars: buriedness/26, distance to protein/8, deep flag, counts of atoms within 4/6/8 A (scaled)
        initial vectors: unit vector to nearest atom, mean direction to atoms within 6 A, zero
  surf  SAS points (Shrake-Rupley), <= n_surf: one-hot owner element (5) + one-hot owner residue (21) + buriedness/26
        initial vectors: outward normal, owner CA->atom, zero
Edges: k nearest within a radius per ordered type pair; edge_type = 3 * src_type + dst_type.
Labels: residue segmentation, probe occupancy, probe hotspot classes, site centres, site property vectors,
site-probe soft membership (probes within 8 A of the site's ligand atoms).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from equicave import ccd, detect, labels as LB, pockets as pk, residues as resprops, structure

RES = list("ACDEFGHIKLMNPQRSTVWY") + ["X"]
ELEM = ["C", "N", "O", "S", "X"]
K_EDGES = {(0, 0): 16, (0, 1): 8, (0, 2): 8, (1, 0): 24, (1, 1): 8, (1, 2): 8, (2, 0): 6, (2, 1): 8, (2, 2): 8}
R_EDGES = {(0, 0): 12.0, (0, 1): 8.0, (0, 2): 8.0, (1, 0): 10.0, (1, 1): 4.0, (1, 2): 6.0, (2, 0): 6.0, (2, 1): 6.0, (2, 2): 4.0}


def onehot(idx: np.ndarray, n: int) -> np.ndarray:
    o = np.zeros((len(idx), n), np.float32); o[np.arange(len(idx)), idx] = 1; return o


def backbone_vectors(st: dict, rt: dict) -> np.ndarray:
    """[L, 3, 3]: N->CA, C->CA, CA->CB (ideal virtual CB from the backbone when the real one is missing)."""
    pos = {}
    for i, (rid, name) in enumerate(zip(st["resid"], st["atom"])):
        if name in ("N", "C", "CB"):
            pos.setdefault(rid, {})[name] = st["xyz"][i]
    out = np.zeros((len(rt["resid"]), 3, 3), np.float32)
    for k, (rid, ca) in enumerate(zip(rt["resid"], rt["ca"])):
        p = pos.get(rid, {})
        n = p.get("N", ca + np.array([1.0, 0, 0])); c = p.get("C", ca + np.array([0, 1.0, 0]))
        b = n - ca; cc = c - ca
        cb = p.get("CB", ca - 0.58273431 * np.cross(b, cc) + 0.56802827 * b - 0.54067466 * cc)   # ideal virtual CB
        out[k, 0] = b; out[k, 1] = cc; out[k, 2] = cb - ca
    norms = np.linalg.norm(out, axis=-1, keepdims=True)
    return out / np.maximum(norms, 1e-6)


SEQ_BUCKETS = (1, 2, 3, 4, 8, 16, 32, 64)     # |i - j| along the chain, bucketed; the last bucket is "far or other chain"


def edge_scalars(ei: np.ndarray, types: np.ndarray, res_index: np.ndarray, chain_index: np.ndarray) -> np.ndarray:
    """Per-edge scalars that distances cannot express: sequence separation bucket and same-chain flag.

    Two residues 4 apart along the chain are a helix turn; two residues from different chains at the same distance are
    an interface. Without this the network cannot tell those cases apart, since both look identical in geometry.
    """
    n_b = len(SEQ_BUCKETS) + 1
    out = np.zeros((ei.shape[1], n_b + 2), np.float32)
    src, dst = ei
    both_res = (types[src] == 0) & (types[dst] == 0)
    same_chain = both_res & (chain_index[src] == chain_index[dst])
    sep = np.abs(res_index[src] - res_index[dst])
    bucket = np.full(ei.shape[1], n_b - 1, int)
    for b, lim in enumerate(SEQ_BUCKETS):
        bucket = np.where(same_chain & (sep <= lim) & (bucket == n_b - 1), b, bucket)
    out[np.arange(ei.shape[1]), np.where(both_res, bucket, n_b - 1)] = 1.0
    out[:, n_b] = same_chain.astype(np.float32)
    out[:, n_b + 1] = both_res.astype(np.float32)
    return out


def knn_edges(pos: np.ndarray, types: np.ndarray, k_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Edges per ordered node-type pair. `k_scale` < 1 thins the graph (CPU pilots); 1.0 is the training default."""
    src, dst = [], []
    trees = {t: (cKDTree(pos[types == t]), np.where(types == t)[0]) for t in np.unique(types)}
    for (ts, td), k0 in K_EDGES.items():
        k = max(2, int(round(k0 * k_scale)))
        if ts not in trees or td not in trees:
            continue
        tree_s, idx_s = trees[ts]; idx_d = trees[td][1]
        kk = min(k, len(idx_s) - (1 if ts == td else 0))
        if kk <= 0:
            continue
        d, j = tree_s.query(pos[idx_d], k=kk + (1 if ts == td else 0), distance_upper_bound=R_EDGES[(ts, td)])
        d, j = np.atleast_2d(d), np.atleast_2d(j)
        for col in range(j.shape[1]):
            ok = np.isfinite(d[:, col])
            s = idx_s[np.minimum(j[ok, col], len(idx_s) - 1)]; t = idx_d[ok]
            keep = s != t
            src.append(s[keep]); dst.append(t[keep])
    if not src:
        return np.zeros((2, 0), np.int64), np.zeros(0, np.int64)
    ei = np.stack([np.concatenate(src), np.concatenate(dst)]).astype(np.int64)
    ei = np.unique(ei, axis=1)
    return ei, (types[ei[0]] * 3 + types[ei[1]]).astype(np.int64)


def featurize(pdb_path, lig_codes: set[str] | None, esm: np.ndarray | None, n_probe: int = 768, n_surf: int = 512,
              seed: int = 0, entries: dict | None = None, k_scale: float = 1.0, druglike_only: bool = False,
              require_interaction: bool = True, residue_chemistry: bool = True) -> dict | None:
    """All arrays of one structure (inputs + labels when ligands are given). None if the structure is unusable."""
    rng = np.random.default_rng(seed)
    st = structure.read_pdb(pdb_path)
    if len(st["xyz"]) < 50:
        return None
    rt = structure.residue_table(st)
    xyz = st["xyz"]; tree = cKDTree(xyz)
    # residues
    ridx = np.array([RES.index(a) if a in RES else 20 for a in rt["seq"]])
    bz = (st["bfactor"] - st["bfactor"].mean()) / (st["bfactor"].std() + 1e-6)
    res_b = np.array([bz[st["resid"] == r].mean() for r in rt["resid"]], np.float32)
    nb8 = np.array([len(tree.query_ball_point(c, 8.0)) for c in rt["ca"]], np.float32)   # local density ~ 1 - rSASA
    parts = [onehot(ridx, 21)]
    if residue_chemistry:
        parts.append(resprops.properties(rt["resname"]).astype(np.float32))
    feat_res = np.concatenate(parts + [res_b[:, None], (nb8[:, None] / 60.0)], 1).astype(np.float32)
    if esm is not None:
        feat_res = np.concatenate([feat_res, esm.astype(np.float32)], 1)
    vec_res = backbone_vectors(st, rt)
    if residue_chemistry:
        vec_res = np.concatenate([vec_res, resprops.side_chain_vectors(st, rt)[:, 1:]], 1)
    vec_res = vec_res.astype(np.float32)
    # probes: cavity lattice points
    cands, field = detect.detect_sites(xyz, return_field=True)
    lo, bur, free, dist = field["origin"], field["buried"], field["free"], field["dist"]
    mask = free & (bur >= detect.FILL_MIN_BURIED)
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return None
    b_all = bur[tuple(idx.T)]
    deep = b_all >= detect.DETECT_MIN_BURIED
    order = np.concatenate([rng.permutation(np.where(deep)[0]), rng.permutation(np.where(~deep)[0])])[:n_probe]
    idx = idx[order]
    ppos = (lo + idx * pk.GRID).astype(np.float32)
    pb = bur[tuple(idx.T)].astype(np.float32); pd_ = dist[tuple(idx.T)].astype(np.float32)
    cnt = np.stack([np.array([len(x) for x in tree.query_ball_point(ppos, r)]) for r in (4.0, 6.0, 8.0)], 1).astype(np.float32)
    feat_probe = np.concatenate([pb[:, None] / 26, pd_[:, None] / 8, (pb >= detect.DETECT_MIN_BURIED)[:, None].astype(np.float32),
                                 cnt / np.array([20, 60, 140], np.float32)], 1)
    dn, jn = tree.query(ppos)
    v1 = (xyz[jn] - ppos) / np.maximum(dn[:, None], 1e-6)
    v2 = np.zeros_like(ppos)
    for i, nb in enumerate(tree.query_ball_point(ppos, 6.0)):
        if nb:
            d = xyz[nb] - ppos[i]; v2[i] = d.mean(0)
    v2 /= np.maximum(np.linalg.norm(v2, axis=1, keepdims=True), 1e-6)
    vec_probe = np.stack([v1, v2] + [np.zeros_like(v1)] * (vec_res.shape[1] - 2), 1).astype(np.float32)
    # surface points
    spts, owner = pk.sas_points(xyz, pk.vdw_radii(st["element"]), n_sphere=30)
    if len(spts) > n_surf:
        sel = rng.choice(len(spts), n_surf, replace=False); spts, owner = spts[sel], owner[sel]
    sel_e = np.array([ELEM.index(e) if e in ELEM else 4 for e in st["element"][owner]])
    rmap = {r: i for i, r in enumerate(rt["resid"])}
    sres = np.array([rmap[r] for r in st["resid"][owner]])
    sb = pk._buried(spts, tree).astype(np.float32) if len(spts) else np.zeros(0, np.float32)
    feat_surf = np.concatenate([onehot(sel_e, 5), onehot(ridx[sres], 21), sb[:, None] / 26], 1).astype(np.float32)
    normal = spts - xyz[owner]; normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-6)
    ca_atom = xyz[owner] - rt["ca"][sres]; ca_atom /= np.maximum(np.linalg.norm(ca_atom, axis=1, keepdims=True), 1e-6)
    vec_surf = np.stack([normal, ca_atom] + [np.zeros_like(normal)] * (vec_res.shape[1] - 2), 1).astype(np.float32)
    if residue_chemistry:                                                # invariant: does the chemistry face the cavity
        feat_res = np.concatenate([feat_res, resprops.pocket_facing(st, rt, ppos).astype(np.float32)], 1)
    # graph
    pos = np.concatenate([rt["ca"], ppos, spts]).astype(np.float32)
    types = np.concatenate([np.zeros(len(rt["ca"])), np.ones(len(ppos)), np.full(len(spts), 2)]).astype(np.int64)
    ei, et = knn_edges(pos, types, k_scale)
    n_res_nodes = len(rt["ca"])
    res_index = np.zeros(len(pos), int); chain_index = np.zeros(len(pos), int)
    res_index[:n_res_nodes] = np.arange(n_res_nodes)
    chains = {c: i for i, c in enumerate(dict.fromkeys(rt["chain"].tolist()))}
    chain_index[:n_res_nodes] = [chains[c] for c in rt["chain"]]
    res_index[n_res_nodes:] = -10_000; chain_index[n_res_nodes:] = -1
    out = dict(pdb=Path(pdb_path).stem, pos=pos, node_type=types, feat_res=feat_res, feat_probe=feat_probe, feat_surf=feat_surf,
               vec0=np.concatenate([vec_res, vec_probe, vec_surf]).astype(np.float32), edge_index=ei, edge_type=et,
               edge_scalar=edge_scalars(ei, types, res_index, chain_index),
               res_type=ridx.astype(np.int64), n_res=n_res_nodes, n_probe=len(ppos), n_surf=len(spts), resid=rt["resid"],
               cand_centers=np.array([c["center"] for c in cands], np.float32).reshape(-1, 3))
    # labels
    if lig_codes is not None:
        ligs = [l for l in structure.read_ligands(pdb_path, min_heavy=8) if l["comp"] in lig_codes]
        entries = entries or {}
        if druglike_only:
            ligs = [l for l in ligs if LB.druglike_ligand(l, entries.get(l["comp"]))]
        if ligs:
            L = np.vstack([l["xyz"] for l in ligs])
            sites = LB.group_sites(ligs)
            occ, hot, prox = LB.point_labels(ppos, ligs, entries, st=st, require_interaction=require_interaction)
            out.update(y_res=LB.residue_labels(st, L).astype(np.float32), y_occ=occ.astype(np.float32), y_hot=hot.astype(np.float32),
                       y_hot_proximity=prox.astype(np.float32),
                       site_centers=LB.site_centres(ligs, sites).astype(np.float32),
                       y_prop=np.stack([LB.site_properties([ligs[i] for i in s], entries, xyz) for s in sites]).astype(np.float32),
                       site_n_atoms=np.array([sum(len(ligs[i]["xyz"]) for i in s) for s in sites], np.float32),
                       site_probe_mask=np.stack([(cKDTree(np.vstack([ligs[i]["xyz"] for i in s])).query(ppos)[0] <= 8.0) for s in sites]).astype(np.float32),
                       lig_xyz=L.astype(np.float32), lig_cls=np.vstack([LB.ligand_atom_classes(l, entries.get(l["comp"])) for l in ligs]).astype(np.float32))
    return out


def save(d: dict, path: Path) -> None:
    np.savez_compressed(path, **{k: (np.array(v) if not isinstance(v, np.ndarray) else v) for k, v in d.items()})


def load(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}


def to_torch(d: dict, device="cpu"):
    import torch
    n_res, n_probe = int(d["n_res"]), int(d["n_probe"])
    keys = ["pos", "node_type", "feat_res", "feat_probe", "feat_surf", "vec0", "edge_index", "edge_type"]
    b = {k: torch.as_tensor(np.asarray(d[k]), device=device) for k in keys}
    for k in ("edge_scalar", "res_type", "site_n_atoms"):
        if k in d:
            b[k] = torch.as_tensor(np.asarray(d[k]), device=device)
    b["slices"] = dict(res=torch.arange(n_res, device=device), probe=torch.arange(n_res, n_res + n_probe, device=device),
                       surf=torch.arange(n_res + n_probe, len(d["pos"]), device=device))
    for k in ("y_res", "y_occ", "y_hot", "y_hot_proximity", "site_centers", "y_prop", "site_probe_mask", "site_n_atoms"):
        if k in d:
            b[k] = torch.as_tensor(np.asarray(d[k]), device=device, dtype=torch.float32)
    if "res_type" in b:
        b["res_type"] = b["res_type"].long()
    b["pdb"] = d.get("pdb", "")
    return b


def mask_residues(b: dict, frac: float = 0.15, rng=None):
    """Replace the residue one-hot of a random subset by a mask token (all zeros) and return the mask.

    The masked residues are the targets of the self-supervised `masked_residue_loss`. Only the 21 one-hot columns are
    cleared; the b-factor and density columns and the ESM-2 block, when present, are cleared too, otherwise the task
    would be trivially solvable from the embedding.
    """
    import torch
    if frac <= 0 or "res_type" not in b:
        return b, torch.zeros(len(b["feat_res"]), dtype=torch.bool, device=b["feat_res"].device)
    g = torch.Generator(device="cpu")
    if rng is not None:
        g.manual_seed(int(rng.integers(0, 2 ** 31)))
    m = torch.rand(len(b["feat_res"]), generator=g).to(b["feat_res"].device) < frac
    b = dict(b)
    f = b["feat_res"].clone()
    f[m] = 0.0
    b["feat_res"] = f
    return b, m


def random_rotation(b: dict, rng: np.random.Generator):
    """Rotate positions and initial vectors (augmentation; the network is equivariant, so this is a check more than a need)."""
    import torch
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    R = torch.tensor(q, dtype=torch.float32, device=b["pos"].device)
    c = b["pos"].mean(0)
    b = dict(b)
    b["pos"] = (b["pos"] - c) @ R.T
    b["vec0"] = torch.einsum("nkc,dc->nkd", b["vec0"], R)
    if "site_centers" in b:
        b["site_centers"] = (b["site_centers"] - c) @ R.T
    return b


def build_cache(manifest_csv: Path, pdb_dir: Path, out_dir: Path, esm_name: str | None, limit: int = 0, n_probe: int = 768,
                n_surf: int = 512, device: str = "cpu", log=print, k_scale: float = 1.0, druglike_only: bool = False,
                require_interaction: bool = True, residue_chemistry: bool = True) -> list[str]:
    """Featurise every manifest structure once; returns the list of cached ids. Idempotent."""
    import csv
    from training.pockets.esm_embed import Embedder, cached
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(manifest_csv)))
    if limit:
        rows = rows[:limit]
    emb = Embedder(esm_name, device) if esm_name else None
    comps = sorted({l[0] for r in rows for l in json.loads(r["ligands"])})
    entries = {c: ccd.load(c) for c in comps}
    done = []
    for i, r in enumerate(rows, 1):
        f = out_dir / f"{r['pdb']}.npz"; p = pdb_dir / f"{r['pdb']}.pdb"
        if f.exists():
            done.append(r["pdb"]); continue
        if not p.exists():
            continue
        try:
            st = structure.read_pdb(p); rt = structure.residue_table(st)
            e = cached(emb, r["pdb"], rt["seq"], rt["chain"], out_dir.parent / "esm") if esm_name else None
            d = featurize(p, {l[0] for l in json.loads(r["ligands"])}, e, n_probe, n_surf, entries=entries, k_scale=k_scale,
                          druglike_only=druglike_only, require_interaction=require_interaction,
                          residue_chemistry=residue_chemistry)
            if d is not None and "y_res" in d:
                d["cluster30"] = r["cluster30"]; d["fold"] = int(r["fold"])
                save(d, f); done.append(r["pdb"])
        except Exception as ex:  # noqa: BLE001
            log(f"  {r['pdb']}: {type(ex).__name__}: {ex}")
        if i % 50 == 0:
            log(f"  cached {len(done)}/{i}")
    return done
