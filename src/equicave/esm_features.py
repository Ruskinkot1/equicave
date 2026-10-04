"""Language-model features for a candidate at inference time, reproducing the training columns exactly.

`scripts/train/build_esm_features.py` reduces the pooled ESM-2 embeddings of a candidate's lining residues with a
PCA fitted out of fold for the cross-validated numbers, and also saves a projection fitted on everything
(`esm_projection_<tag>.npz`) for serving. This module applies that saved projection to a new structure, so a model
trained with `esm_*` columns can score a protein it has never seen with the same feature definition. Without the
projection file the features are unavailable and the caller is told, rather than being handed zeros that would look
like a valid prediction.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import structure

REPO = Path(__file__).resolve().parents[2]


class EsmFeatures:
    def __init__(self, tag: str = "native2", ds=None, device: str = "cpu"):
        ds = Path(ds or REPO / "data/processed")
        spec_file = ds / f"esm_features_{tag}.json"
        if not spec_file.exists():
            raise FileNotFoundError(f"{spec_file.name} missing: run scripts/train/build_esm_features.py --tag {tag}")
        self.spec = json.loads(spec_file.read_text())
        proj_file = ds / self.spec.get("projection", f"esm_projection_{tag}.npz")
        if not proj_file.exists():
            raise FileNotFoundError(f"{proj_file.name} missing: rebuild the features so the projection is saved")
        self.proj = dict(np.load(proj_file))
        self.columns = self.spec["features"]
        self.radius = float(self.spec["radius"])
        self.dims = int(self.spec["dims"])
        from training.pockets.esm_embed import Embedder
        self._embedder = Embedder(self.spec["model"], device)
        self._cache = REPO / "data/cache/esm"

    def _project(self, name: str, v: np.ndarray) -> np.ndarray:
        return (v - self.proj[f"{name}_mean"]) @ self.proj[f"{name}_components"].T

    def __call__(self, pdb_path, centers: np.ndarray) -> list[dict]:
        """One dict per candidate centre with exactly the training column names."""
        from scipy.spatial import cKDTree
        from training.pockets.esm_embed import cached
        st = structure.read_pdb(pdb_path)
        rt = structure.residue_table(st)
        e = cached(self._embedder, Path(pdb_path).stem, rt["seq"], rt["chain"], self._cache)
        centers = np.atleast_2d(np.asarray(centers, float))
        if e is None or len(e) != len(rt["ca"]):
            return [dict.fromkeys(self.columns, 0.0) for _ in centers]
        prot_mean = e.mean(0)
        tree = cKDTree(rt["ca"])
        rows = []
        for c in centers:
            idx = tree.query_ball_point(c, self.radius + 4.0) or [int(tree.query(c)[1])]
            E = e[idx]
            m, mx = E.mean(0), E.max(0)
            row = dict(esm_n_res=float(len(idx)), esm_norm=float(np.linalg.norm(m)),
                       esm_cos_protein=float(m @ prot_mean / (np.linalg.norm(m) * np.linalg.norm(prot_mean) + 1e-9)))
            for name, v in (("mean", m), ("max", mx)):
                for j, x in enumerate(self._project(name, v)):
                    row[f"esm_{name}{j}"] = float(x)
            rows.append({k: row.get(k, 0.0) for k in self.columns})
        return rows
