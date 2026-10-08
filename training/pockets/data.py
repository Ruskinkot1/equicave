"""Featurisation of one structure for EquiCave-Net and the on-disk cache used by training.

Node types and inputs
  res   one per residue at CA: one-hot residue (21) + 14 amino-acid property scalars (`equicave.residues`) +
        2 pocket-facing cosines + ESM-2 embedding (optional) + [b-factor z, local density]
        initial vectors: N->CA, C->CA, CA->CB(virtual for GLY), CA->side-chain centroid, CA->functional-group centroid
        The last two say which way the residue's chemistry faces, which is what distinguishes an Asp whose
        carboxylate lines the cavity from one pointing at the solvent.
  probe free cavity lattice points (buriedness >= FILL_MIN_BURIED, deep points first), <= n_probe
        scalars: buriedness/26, distance to protein/8, deep flag, counts of atoms within 4/6/8 A (scaled),
        and the interaction potential of the point itself (21 numbers): a strict availability flag, the distance to
        the nearest partner and the number of partners within 6 A, for each of the seven interaction types. A plain
        "within the cutoff" flag is useless here, because inside a protein every probe satisfies it; the distances
        and counts are where the signal is. Without them the trunk has to infer "this probe faces an Asp
        carboxylate" from residue one-hots through several rounds of message passing, when it is one tree query; the
        same information is what the ranker's chemistry group is made of, and that group is worth the most to it.
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

from equicave import (ccd, detect, labels as LB, pocket_features as pf, pockets as pk, progress,
                      residues as resprops, structure)

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
              require_interaction: bool = True, residue_chemistry: bool = True,
              probe_potential: bool = True, probe_model=None,
              probe_ligandable_frac: float = 0.5, probe_metal: bool = False,
              probe_electrostatic: bool = False, probe_conservation: bool = False,
              conservation_dir: str | None = None, probe_protrusion: bool = False) -> dict | None:
    """All arrays of one structure (inputs + labels when ligands are given). None if the structure is unusable.

    `probe_model`: an optional per-point ligandability booster. Probes are the one component the ablation shows to
    matter -- removing them costs 0.263 site top-1, where every other component is within two seed standard
    deviations of zero -- so which free points become probes is the highest-leverage choice in the architecture.
    Without a model the points are ordered at random inside each buriedness tier; with one they are ordered by
    predicted ligandability inside each tier, which spends the probe budget on the points a ligand atom is likely to
    occupy while the tiers still guarantee coverage of every cavity.
    """
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
    if probe_model is not None:
        # Part of the budget goes to the points the per-point model likes, the rest keeps the original tiered random
        # order. Spending all of it on the best-scoring points is measurably wrong rather than merely aggressive:
        # on one structure it put 0.895 of the probes within 4 A of a ligand against 0.277 at random, which sounds
        # like a win until you remember the site head's job is to *reject* the other cavities. With almost no probes
        # left outside the true site there are almost no negatives to learn from. The fraction is the knob, and
        # `probe_tiered` (fraction 0) is the comparison arm.
        from equicave import point_score as ps
        all_pts = (lo + idx * pk.GRID).astype(np.float32)
        types_s = LB.protein_atom_types(st)
        trees_s = {k: (cKDTree(xyz[m]) if m.sum() else None) for k, m in types_s.items()}
        sc = probe_model.predict(ps.point_features(all_pts, st, tree, trees_s))
        n_lig = int(round(n_probe * float(probe_ligandable_frac)))
        by_score = np.argsort(-sc)[:n_lig]
        rest = np.concatenate([rng.permutation(np.where(deep)[0]), rng.permutation(np.where(~deep)[0])])
        rest = rest[~np.isin(rest, by_score)]
        order = np.concatenate([by_score, rest])[:n_probe]
    else:
        order = np.concatenate([rng.permutation(np.where(deep)[0]), rng.permutation(np.where(~deep)[0])])[:n_probe]
    idx = idx[order]
    ppos = (lo + idx * pk.GRID).astype(np.float32)
    pb = bur[tuple(idx.T)].astype(np.float32); pd_ = dist[tuple(idx.T)].astype(np.float32)
    cnt = np.stack([np.array([len(x) for x in tree.query_ball_point(ppos, r)]) for r in (4.0, 6.0, 8.0)], 1).astype(np.float32)
    feat_probe = np.concatenate([pb[:, None] / 26, pd_[:, None] / 8, (pb >= detect.DETECT_MIN_BURIED)[:, None].astype(np.float32),
                                 cnt / np.array([20, 60, 140], np.float32)], 1)
    types_e = LB.protein_atom_types(st) if (probe_potential or probe_electrostatic) else None
    if probe_potential:
        types_p = types_e
        trees_p = {k: (cKDTree(xyz[m]) if m.sum() else None) for k, m in types_p.items()}
        cnt_p, dmin, avail = pf.point_potential(ppos, trees_p)
        feat_probe = np.concatenate([feat_probe, avail, dmin / 6.0, np.log1p(cnt_p) / 3.0], 1).astype(np.float32)
    if probe_protrusion:
        # P2Rank's protrusion -- the number of *all* protein atoms within 10 A of the point -- is the single most
        # important of its 35 features (RF importance 0.0845 against 0.0139 for the runner-up); protrusion alone
        # reaches COACH420 Top-n 64.2 against 71.4 for the full set, and removing it costs 10.9 Top-n points on
        # COACH420 against 1.9 on HOLO4K. Our probe channels stop at 8 A, and the long radius is not redundant
        # with the closure field: measured on 268 structures and 8039 candidates of our own detector, the
        # correct-against-decoy AUC of a single count is 0.650 at 8 A and 0.837 at 12 A against 0.755 for the
        # 26-direction closure, and a logistic model on our existing four geometric inputs goes from 0.7999 to
        # 0.8469 (5-fold CV) when the three long counts are added. log1p rather than a divisor because the counts
        # run to the high hundreds and their useful variation is at the low end.
        cnt_l = np.stack([np.array([len(x) for x in tree.query_ball_point(ppos, r)]) for r in (10.0, 12.0, 15.0)], 1)
        feat_probe = np.concatenate([feat_probe, np.log1p(cnt_l).astype(np.float32) / 6.0], 1).astype(np.float32)
    if probe_metal:
        # Metals reach the network from `read_metals`, not from `read_pdb`: the parser keeps the polymer only, so
        # until now every ion in every training structure was discarded before featurisation. Ions are about 40 %
        # of ligand sites in LIGYSIS and no published site predictor ablates its metal channel, so this is new
        # information rather than a re-encoding of something the model already has -- which is the property the
        # ESM-2, surface and degree-2 arms all failed to have.
        md, mw = pf.point_metal(ppos, structure.read_metals(pdb_path))
        feat_probe = np.concatenate([feat_probe, md / 16.0, np.log1p(mw) / 2.0], 1).astype(np.float32)
    if probe_conservation:
        # The one external signal in this literature with a measured effect on train-dissimilar structures
        # (P2Rank_CONS: top-(N+2) 53.9 % against 51.9 % on LIGYSIS, +346 true positives at a 100-false-positive
        # budget, on a benchmark with 0.5-9.7 % training overlap) and the only one that cannot be leakage, since it
        # is computed per target at inference from an alignment. The scores are read, not computed: an alignment
        # against a UniRef-scale database costs more than the whole rest of this pipeline, and the file is the
        # interface. Missing file is an error rather than zeros, or the arm would train on an empty channel and
        # report it as a null.
        from equicave import conservation as CONS
        f_c = Path(conservation_dir or "") / f"{Path(pdb_path).stem}.txt"
        sc = CONS.from_file(f_c, len(rt["resid"]))
        if sc is None:
            raise FileNotFoundError(f"probe_conservation needs per-residue scores at {f_c} "
                                    f"(two columns, `residue_index score`, as P2Rank consumes them)")
        rmap_c = {r: i for i, r in enumerate(rt["resid"])}
        ra_c = np.array([rmap_c[r] for r in st["resid"]], int)
        c_mean, c_max, c_cov = pf.point_conservation(ppos, xyz, ra_c, sc)
        feat_probe = np.concatenate([feat_probe, np.stack([c_mean, c_max, c_cov], 1)], 1).astype(np.float32)
    e_fld = None
    if probe_electrostatic:
        e_pot, e_fld = pf.point_field(ppos, st, types_e)
        feat_probe = np.concatenate([feat_probe, e_pot[:, None] * 2.0,
                                     np.linalg.norm(e_fld, axis=1, keepdims=True) * 10.0], 1).astype(np.float32)
    dn, jn = tree.query(ppos)
    v1 = (xyz[jn] - ppos) / np.maximum(dn[:, None], 1e-6)
    v2 = np.zeros_like(ppos)
    for i, nb in enumerate(tree.query_ball_point(ppos, 6.0)):
        if nb:
            d = xyz[nb] - ppos[i]; v2[i] = d.mean(0)
    v2 /= np.maximum(np.linalg.norm(v2, axis=1, keepdims=True), 1e-6)
    pvec = [v1, v2]
    if e_fld is not None:                       # degree 1, so it rotates with the structure: see pf.point_field
        pvec.append(e_fld / np.maximum(np.linalg.norm(e_fld, axis=1, keepdims=True), 1e-6))
    vec_probe = np.stack(pvec + [np.zeros_like(v1)] * (vec_res.shape[1] - len(pvec)), 1).astype(np.float32)
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
               probe_potential=(avail if probe_potential else np.zeros((len(ppos), 7), np.float32)),
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
                       **_probe_directions(ppos, ligs, sites),
                       lig_xyz=L.astype(np.float32), lig_cls=np.vstack([LB.ligand_atom_classes(l, entries.get(l["comp"])) for l in ligs]).astype(np.float32))
    return out


def _probe_directions(ppos: np.ndarray, ligs: list, sites: list, reach: float = 8.0) -> dict:
    """Unit vector from each probe to the nearest ligand heavy atom of its site, and the mask of usable probes.

    A probe is usable when exactly one site is within `reach`: with two sites in range the target direction is
    ambiguous and the probe is dropped rather than supervised towards an arbitrary one. This is the dense geometric
    target of P12 -- the centre set loss reaches one proposal per site, this reaches every probe near a site.
    """
    per_site = [cKDTree(np.vstack([ligs[i]["xyz"] for i in s])) for s in sites]
    d = np.stack([t.query(ppos)[0] for t in per_site])                       # [S, P]
    near = d <= reach
    usable = near.sum(0) == 1
    which = np.argmax(near, 0)
    direction = np.zeros((len(ppos), 3), np.float32)
    for si, tree in enumerate(per_site):
        rows = np.flatnonzero(usable & (which == si))
        if not len(rows):
            continue
        target = tree.data[tree.query(ppos[rows])[1]]
        v = target - ppos[rows]
        n = np.linalg.norm(v, axis=1, keepdims=True)
        ok = n[:, 0] > 1e-6
        direction[rows[ok]] = (v[ok] / n[ok]).astype(np.float32)
        usable[rows[~ok]] = False                       # a probe sitting on a ligand atom has no direction
    return dict(probe_dir=direction, probe_dir_mask=usable.astype(np.float32))


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
    for k in ("edge_scalar", "res_type", "site_n_atoms", "probe_potential"):
        if k in d:
            b[k] = torch.as_tensor(np.asarray(d[k]), device=device)
    b["slices"] = dict(res=torch.arange(n_res, device=device), probe=torch.arange(n_res, n_res + n_probe, device=device),
                       surf=torch.arange(n_res + n_probe, len(d["pos"]), device=device))
    for k in ("y_res", "y_occ", "y_hot", "y_hot_proximity", "site_centers", "y_prop", "site_probe_mask", "site_n_atoms",
              "probe_dir", "probe_dir_mask"):
        if k in d:
            b[k] = torch.as_tensor(np.asarray(d[k]), device=device, dtype=torch.float32)
    if "res_type" in b:
        b["res_type"] = b["res_type"].long()
    b["pdb"] = d.get("pdb", "")
    return b


def mask_probe_potential(b: dict, frac: float = 0.2, rng=None):
    """Hide the potential channels of a random subset of probes and return them as a target.

    Feeding the potential and predicting it would be an identity map; masking makes it supervision. A probe whose own
    channels are hidden has to infer what it could bind from its neighbourhood, which is the reasoning the site heads
    need. The hidden rows are zeroed in `feat_probe` (both the availability flags and the distances).
    """
    import torch
    if frac <= 0 or "probe_potential" not in b:
        return b, torch.zeros(0, dtype=torch.bool, device=b["feat_probe"].device), None
    n = len(b["feat_probe"])
    g = torch.Generator(device="cpu")
    if rng is not None:
        g.manual_seed(int(rng.integers(0, 2 ** 31)))
    m = (torch.rand(n, generator=g) < frac).to(b["feat_probe"].device)
    b = dict(b)
    f = b["feat_probe"].clone()
    f[m, -21:] = 0.0
    b["feat_probe"] = f
    return b, m, b["probe_potential"]


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
    if "probe_dir" in b:            # a direction, so it rotates but is not translated; forgetting this silently
        b["probe_dir"] = b["probe_dir"] @ R.T          # destroys the direction loss, hence test_probe_directions_rotate
    return b


def jitter(b: dict, rng: np.random.Generator, pos: float = 0.0, feat: float = 0.0, drop: float = 0.0):
    """Coordinate, feature and node-dropout noise, the three published augmentations for this task.

    Shiota et al. (PLOS ONE 2024, 10.1371/journal.pone.0308425) is the one clean data-versus-augmentation
    experiment in binding-site prediction: the same model, the same test set and the same identity filter, with
    noise augmentation worth +0.0976 combined PR-AUC against +0.0256 for 2.6x the training structures and +0.0012
    for class balancing. Augmentation is about four times the value of more data at this scale, and the sigmas
    here are theirs rather than ours -- 0.5 A on positions, 0.3 on features, 0.03 node drop.

    Why it should help a model that is already equivariant: rotation augmentation is a no-op for us, so the only
    invariance we currently train is one we already had by construction. These three are different -- coordinate
    noise asks the model to tolerate the resolution and conformational slack that separates one crystal form from
    another, which is exactly what fails when the test structure is a different deposition of the same fold.

    Nodes are dropped from the graph, so the edges touching them go too; a structure never drops below one node.
    """
    import torch
    if not (pos or feat or drop):
        return b
    b = dict(b)
    dev = b["pos"].device
    if pos:
        b["pos"] = b["pos"] + torch.tensor(rng.normal(0, pos, b["pos"].shape), dtype=b["pos"].dtype, device=dev)
    if feat:
        for k in ("feat_res", "feat_probe", "feat_surf"):
            if k in b and b[k].numel():
                b[k] = b[k] + torch.tensor(rng.normal(0, feat, tuple(b[k].shape)), dtype=b[k].dtype, device=dev)
    if drop:
        n = b["pos"].shape[0]
        keep = torch.tensor(rng.random(n) >= drop, device=dev)
        if not bool(keep.any()):
            return b
        b = drop_nodes(b, keep)
    return b


def drop_nodes(b: dict, keep):
    """Keep the nodes `keep` marks, renumber the edges, and shrink every per-node and per-slice tensor with them."""
    import torch
    dev = b["pos"].device
    n = b["pos"].shape[0]
    idx = torch.full((n,), -1, dtype=torch.long, device=dev)
    idx[keep] = torch.arange(int(keep.sum()), device=dev)
    out = dict(b)
    for k, v in b.items():
        if torch.is_tensor(v) and v.shape[:1] == (n,) and k not in ("edge_index", "edge_type"):
            out[k] = v[keep]
    ei = b["edge_index"]
    live = keep[ei[0]] & keep[ei[1]]
    out["edge_index"] = idx[ei[:, live]]
    if "edge_type" in b:
        out["edge_type"] = b["edge_type"][live]
    if "edge_scalar" in b and torch.is_tensor(b["edge_scalar"]) and len(b["edge_scalar"]) == ei.shape[1]:
        out["edge_scalar"] = b["edge_scalar"][live]
    if "slices" in b:
        # Slices index nodes by type; each becomes the surviving members under the new numbering.
        out["slices"] = {k: idx[v[keep[v]]] for k, v in b["slices"].items()}
    return out


def build_cache(manifest_csv: Path, pdb_dir: Path, out_dir: Path, esm_name: str | None, limit: int = 0, n_probe: int = 768,
                n_surf: int = 512, device: str = "cpu", log=print, k_scale: float = 1.0, druglike_only: bool = False,
                require_interaction: bool = True, residue_chemistry: bool = True,
                probe_potential: bool = True, probe_sampling: str = "tiered",
                point_model_tag: str = "", probe_ligandable_frac: float = 0.5,
                probe_metal: bool = False, probe_electrostatic: bool = False,
                probe_conservation: bool = False, conservation_dir: str | None = None,
                probe_protrusion: bool = False) -> list[str]:
    """Featurise every manifest structure once; returns the list of cached ids. Idempotent.

    `probe_sampling="ligandable"` places the probes by the per-point model instead of at random within each tier.
    Leakage is avoided by construction: a structure in fold k is scored by the point model trained **without** fold
    k (`models/point_<tag>_fold<k>.txt`), the same out-of-fold discipline the ranker's aggregates use, so the
    network never sees probe positions chosen with knowledge of its own structure's ligands.
    """
    import csv
    from training.pockets.esm_embed import Embedder, cached
    out_dir.mkdir(parents=True, exist_ok=True)

    # The cache is keyed by PDB id alone, so every setting that changes the arrays has to be pinned beside it or a
    # changed featurisation silently reuses the old files and the run measures the previous architecture while the
    # log claims the new one. This bit us in waiting: `probe_sampling` moved from random to learned placement and
    # nothing in the pipeline would have noticed.
    from equicave import pocket_features as pf
    sig = dict(n_probe=n_probe, n_surf=n_surf, k_scale=k_scale, druglike_only=druglike_only,
               require_interaction=require_interaction, residue_chemistry=residue_chemistry,
               probe_potential=probe_potential, probe_sampling=probe_sampling,
               point_model_tag=point_model_tag if probe_sampling == "ligandable" else "",
               probe_ligandable_frac=probe_ligandable_frac if probe_sampling == "ligandable" else None,
               probe_metal=probe_metal, probe_electrostatic=probe_electrostatic,
               probe_conservation=probe_conservation, probe_protrusion=probe_protrusion,
               conservation_dir=conservation_dir if probe_conservation else None,
               esm=esm_name, geometry=pf.geometry())
    sig_file = out_dir / ".featurisation.json"
    if sig_file.exists():
        old_sig = json.loads(sig_file.read_text())
        if old_sig != sig:
            differs = sorted(k for k in set(old_sig) | set(sig) if old_sig.get(k) != sig.get(k))
            raise RuntimeError(
                f"the cache at {out_dir} was built with a different featurisation and would be reused as is, so "
                f"the run would measure the old one. Differs in: {differs}. Delete the directory to rebuild it, or "
                f"point data.cache_dir at a new one (the arms that change featurisation each want their own).")
    else:
        sig_file.write_text(json.dumps(sig, indent=1, sort_keys=True))

    rows = list(csv.DictReader(open(manifest_csv)))
    if limit:
        rows = rows[:limit]
    if probe_conservation:
        # Checked here and not per structure: a missing file raises inside `featurize`, where build_cache catches
        # it and drops that structure, so an absent conservation directory would quietly shrink the dataset and
        # the arm would be compared against the baseline on a different set of proteins.
        d_c = Path(conservation_dir or "")
        have = [r["pdb"] for r in rows if (d_c / f"{r['pdb']}.txt").exists()]
        if len(have) < len(rows):
            raise FileNotFoundError(
                f"probe_conservation: {len(rows) - len(have)} of {len(rows)} manifest structures have no "
                f"conservation file in {d_c or '<unset data.conservation_dir>'}. Expected one `<pdb>.txt` per "
                f"structure, two columns `residue_index score`, in the residue order of the PDB file. This arm "
                f"needs an alignment per target (UniRef-scale search); nothing in this repository produces one.")
    emb = Embedder(esm_name, device) if esm_name else None
    comps = sorted({l[0] for r in rows for l in json.loads(r["ligands"])})
    # Each component is one small download the first time it is seen; at full scale there are thousands of them.
    entries = {c: ccd.load(c) for c in progress.track(comps, "chemical component definitions", unit="comp")}

    fold_models = {}
    if probe_sampling == "ligandable":
        import lightgbm as lgb
        root = Path(__file__).resolve().parents[2] / "models"
        tag = point_model_tag or "native"
        for k in sorted({int(r["fold"]) for r in rows}):
            f_k = root / f"point_{tag}_fold{k}.txt"
            if not f_k.exists():
                raise FileNotFoundError(
                    f"probe_sampling=ligandable needs {f_k}: train the per-point model first "
                    f"(scripts/train/build_points.py then train_point_model.py --tag {tag}). The per-fold models "
                    f"are required rather than the full-data one, so a structure is never scored by a model that "
                    f"saw its own ligands.")
            fold_models[k] = lgb.Booster(model_file=str(f_k))
        log(f"probe placement by ligandability, out of fold, from models/point_{tag}_fold*.txt")

    done = []
    with progress.Bar("featurising the network cache", len(rows), unit="pdb") as bar:
        for r in rows:
            f = out_dir / f"{r['pdb']}.npz"; p = pdb_dir / f"{r['pdb']}.pdb"
            if f.exists():
                done.append(r["pdb"]); bar.update(1, postfix=f"{len(done)} cached"); continue
            if not p.exists():
                bar.update(1, postfix=f"{len(done)} cached, {r['pdb']} has no pdb file"); continue
            try:
                st = structure.read_pdb(p); rt = structure.residue_table(st)
                e = cached(emb, r["pdb"], rt["seq"], rt["chain"], out_dir.parent / "esm") if esm_name else None
                d = featurize(p, {l[0] for l in json.loads(r["ligands"])}, e, n_probe, n_surf, entries=entries, k_scale=k_scale,
                              druglike_only=druglike_only, require_interaction=require_interaction,
                              residue_chemistry=residue_chemistry, probe_potential=probe_potential,
                              probe_model=fold_models.get(int(r["fold"])),
                              probe_ligandable_frac=probe_ligandable_frac,
                              probe_metal=probe_metal, probe_electrostatic=probe_electrostatic,
                              probe_conservation=probe_conservation, conservation_dir=conservation_dir,
                              probe_protrusion=probe_protrusion)
                if d is not None and "y_res" in d:
                    d["cluster30"] = r["cluster30"]; d["fold"] = int(r["fold"])
                    save(d, f); done.append(r["pdb"])
            except Exception as ex:  # noqa: BLE001
                log(f"  {r['pdb']}: {type(ex).__name__}: {ex}")
            bar.update(1, postfix=f"{len(done)} cached")
    return done
