"""Frozen ESM-2 residue embeddings (MIT licence, facebook/esm2_*), cached per structure as float16 npz.

Default model esm2_t12_35M_UR50D (480-d) runs on CPU; on a GPU use esm2_t33_650M_UR50D. Chains are embedded one at
a time (<= 1022 residues per window) and concatenated in the residue order of `structure.residue_table`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT = "facebook/esm2_t12_35M_UR50D"


class Embedder:
    def __init__(self, name: str = DEFAULT, device: str = "cpu"):
        import torch
        from transformers import AutoTokenizer, EsmModel
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = EsmModel.from_pretrained(name).eval().to(device)
        self.device = device; self.name = name
        self.dim = self.model.config.hidden_size
        self.torch = torch

    def __call__(self, seq: str, chains: np.ndarray) -> np.ndarray:
        """seq: 1-letter residues; chains: chain id per residue. Returns [L, dim] float32."""
        out = np.zeros((len(seq), self.dim), np.float32)
        start = 0
        for c in dict.fromkeys(chains.tolist()):
            m = chains == c; n = int(m.sum()); s = seq[start:start + n]
            vecs = []
            for i in range(0, n, 1000):
                w = s[i:i + 1000]
                with self.torch.no_grad():
                    h = self.model(**self.tok(w, return_tensors="pt").to(self.device)).last_hidden_state[0, 1:1 + len(w)]
                vecs.append(h.float().cpu().numpy())
            out[start:start + n] = np.concatenate(vecs) if vecs else 0
            start += n
        return out


def cached(embedder: Embedder | None, pdb: str, seq: str, chains: np.ndarray, cache: Path) -> np.ndarray | None:
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"{pdb}.npz"
    if f.exists():
        e = np.load(f)["esm"]
        if len(e) == len(seq):
            return e.astype(np.float32)
    if embedder is None:
        return None
    e = embedder(seq, chains)
    np.savez_compressed(f, esm=e.astype(np.float16))
    return e
