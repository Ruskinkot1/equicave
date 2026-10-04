#!/usr/bin/env python
"""Protein-language-model features per candidate, for the ranker, without the network.

Why this first. The one quantified comparison of feature sources in the pocket literature attributes the largest
single learned-feature gain to ESM-2 embeddings (GDEGAN reports +15.6 % DCC and +9.3 % DCA averaged over three sets;
VN-EGNN's ablation puts them at +0.06 to +0.078 DCC), while the choice of equivariant layer is worth less. Those
features are available to a gradient-boosted ranker directly, with no network and no GPU: embed the sequence once per
structure, aggregate over the residues lining a candidate, and reduce the dimension so a tabular model can use it.

Per candidate: the mean and the max of the ESM-2 residue embeddings over the residues within `--radius` of its
centre, each projected to `--dims` components by a PCA fitted on the training structures only (the fold assignment is
respected: the PCA sees no structure of the fold it will score), plus three invariants that need no projection --
the cosine between the candidate's mean embedding and the protein's mean embedding (how unusual its lining is), the
mean embedding norm, and the number of lining residues.

Usage: python scripts/train/build_esm_features.py --tag native2 [--model facebook/esm2_t12_35M_UR50D] [--dims 16]
Output: data/processed/esm_features_<tag>.csv.gz, joined to the candidate table by (pdb, center).
"""
import argparse, json, pathlib, sys, time

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO))
from equicave import structure, tables  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="native2"); ap.add_argument("--ds", default=str(REPO / "data/processed"))
    ap.add_argument("--pdb-dir", default=str(REPO / "data/pockets_ds/pdb"))
    ap.add_argument("--model", default="facebook/esm2_t12_35M_UR50D")
    ap.add_argument("--radius", type=float, default=8.0); ap.add_argument("--dims", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    ds = pathlib.Path(a.ds)
    cand = tables.read_table(ds, f"candidates_{a.tag}")
    pdbs = sorted(cand["pdb"].unique())
    if a.limit:
        pdbs = pdbs[:a.limit]
    from training.pockets.esm_embed import Embedder, cached
    emb = Embedder(a.model, a.device)
    cache = ds.parent / "cache/esm"
    from scipy.spatial import cKDTree
    rows, t0 = [], time.time()
    for i, pdb in enumerate(pdbs, 1):
        p = pathlib.Path(a.pdb_dir) / f"{pdb}.pdb"
        if not p.exists():
            continue
        st = structure.read_pdb(p); rt = structure.residue_table(st)
        e = cached(emb, pdb, rt["seq"], rt["chain"], cache)
        if e is None or len(e) != len(rt["ca"]):
            continue
        prot_mean = e.mean(0)
        tree = cKDTree(rt["ca"])
        sub = cand[cand["pdb"] == pdb]
        for _, r in sub.iterrows():
            c = np.array([float(x) for x in str(r["center"]).split(";")])
            idx = tree.query_ball_point(c, a.radius + 4.0)        # Ca within reach of the pocket
            if not idx:
                idx = [int(tree.query(c)[1])]
            E = e[idx]
            m, mx = E.mean(0), E.max(0)
            cos = float(m @ prot_mean / (np.linalg.norm(m) * np.linalg.norm(prot_mean) + 1e-9))
            rows.append(dict(pdb=pdb, center=r["center"], esm_n_res=float(len(idx)), esm_norm=float(np.linalg.norm(m)),
                             esm_cos_protein=cos, _mean=m, _max=mx))
        if i % 50 == 0:
            print(f"  {i}/{len(pdbs)} structures, {len(rows)} candidates, {time.time() - t0:.0f}s", flush=True)
    if not rows:
        sys.exit("no embeddings produced")
    df = pd.DataFrame(rows)
    folds = cand.groupby("pdb")["fold"].first()
    df["fold"] = df["pdb"].map(folds).fillna(0).astype(int)
    M = np.vstack(df["_mean"].to_numpy()); X = np.vstack(df["_max"].to_numpy())
    from sklearn.decomposition import PCA
    for name, A in (("mean", M), ("max", X)):
        out = np.zeros((len(df), a.dims), np.float32)
        for k in sorted(df["fold"].unique()):            # fit on the other folds only: no structure sees its own PCA
            tr = (df["fold"] != k).to_numpy()
            if tr.sum() < a.dims + 5:
                tr = np.ones(len(df), bool)
            pca = PCA(n_components=a.dims, random_state=0).fit(A[tr])
            out[~tr] = pca.transform(A[~tr])
        for j in range(a.dims):
            df[f"esm_{name}{j}"] = out[:, j]
    df = df.drop(columns=["_mean", "_max"])
    path = tables.write_table(df, ds, f"esm_features_{a.tag}")
    feats = [c for c in df.columns if c.startswith("esm_")]
    # a projection fitted on everything, saved so inference produces the same columns for a new structure
    proj = {}
    for name, A in (("mean", M), ("max", X)):
        pca = PCA(n_components=a.dims, random_state=0).fit(A)
        proj[f"{name}_components"] = pca.components_.astype(np.float32)
        proj[f"{name}_mean"] = pca.mean_.astype(np.float32)
    np.savez_compressed(ds / f"esm_projection_{a.tag}.npz", **proj)
    (ds / f"esm_features_{a.tag}.json").write_text(json.dumps(dict(model=a.model, radius=a.radius, dims=a.dims,
                                                                   features=feats, n=len(df),
                                                                   projection=f"esm_projection_{a.tag}.npz"), indent=1))
    print(f"{len(df)} candidates, {len(feats)} features -> {path.name}; projection saved for inference")


if __name__ == "__main__":
    main()
