"""A permutation-equivariant set transformer that ranks the candidates of one structure jointly.

Why not gradient boosting. LightGBM scores each candidate on its own; everything it knows about the competition has
to be hand-built into the features (our within-structure z-scores and margins exist for exactly that reason). But the
task is *choose the best of this set*, the set is small (at most 30), and the useful comparisons are learnable: "this
cavity is deeper than every other one here", "two candidates share a cavity, so one of them is redundant". A set
transformer does that natively.

Design:
- **Input** one structure at a time: `[n, F]` candidate features, standardised by statistics fitted on the training
  folds only.
- **Trunk** `L` pre-norm transformer blocks of multi-head self-attention over the candidate axis **with no positional
  encoding**, so the model is exactly permutation-equivariant: feeding the candidates in a different order permutes
  the scores the same way and changes nothing else. `tests/test_set_ranker.py` asserts this numerically.
- **Head** one scalar per candidate.
- **Loss** listwise: softmax cross-entropy of the predicted scores against a target distribution built from graded
  relevance (2 within 2 A of a ligand atom, 1 within 4 A, 0 otherwise), which optimises the top of the list directly,
  plus an auxiliary per-candidate BCE that keeps the scores calibratable. Structures with no positive candidate
  contribute the BCE term only.

The model is small (about 0.4 M parameters at `dim=128, layers=3`), trains on a CPU in minutes for a few hundred
structures, and its score is designed to be stacked with the LambdaRank score rather than to replace it.
"""
from __future__ import annotations

import numpy as np


def _torch():
    import torch
    return torch


class SetRanker:
    """Fit / predict over structures, each a list of candidate feature rows."""

    def __init__(self, n_features: int, dim: int = 128, layers: int = 3, heads: int = 8, dropout: float = 0.1,
                 lr: float = 3e-4, weight_decay: float = 1e-2, epochs: int = 60, batch: int = 8, seed: int = 0,
                 aux_weight: float = 0.3, device: str = "cpu"):
        torch = _torch()
        import torch.nn as nn
        self.p = dict(n_features=n_features, dim=dim, layers=layers, heads=heads, dropout=dropout, lr=lr,
                      weight_decay=weight_decay, epochs=epochs, batch=batch, seed=seed, aux_weight=aux_weight)
        torch.manual_seed(seed)
        self.device = torch.device(device)

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
                self.att = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
                self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * dim, dim))

            def forward(self, x, mask):
                h = self.n1(x)
                a, _ = self.att(h, h, h, key_padding_mask=mask, need_weights=False)
                x = x + a
                return x + self.ff(self.n2(x))

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.inp = nn.Sequential(nn.Linear(n_features, dim), nn.GELU(), nn.Linear(dim, dim))
                self.blocks = nn.ModuleList([Block() for _ in range(layers)])
                self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

            def forward(self, x, mask):
                h = self.inp(x)
                for b in self.blocks:
                    h = b(h, mask)
                return self.out(h).squeeze(-1)

        self.net = Net().to(self.device)
        self.mean_ = None
        self.std_ = None

    # ---- data plumbing -------------------------------------------------------------------------------------------
    def _standardise(self, X: np.ndarray, fit: bool) -> np.ndarray:
        if fit:
            self.mean_ = np.nanmean(X, 0)
            self.std_ = np.nanstd(X, 0) + 1e-6
        X = np.where(np.isfinite(X), X, 0.0)
        return np.clip((X - self.mean_) / self.std_, -8, 8)

    @staticmethod
    def _pad(groups, dim):
        torch = _torch()
        n = max(len(g) for g in groups)
        X = torch.zeros(len(groups), n, dim)
        mask = torch.ones(len(groups), n, dtype=torch.bool)
        for i, g in enumerate(groups):
            X[i, :len(g)] = torch.as_tensor(g, dtype=torch.float32)
            mask[i, :len(g)] = False
        return X, mask

    def fit(self, X: np.ndarray, groups: np.ndarray, relevance: np.ndarray, labels: np.ndarray,
            log=lambda s: None) -> "SetRanker":
        """`groups`: structure id per row. `relevance`: graded (0/1/2). `labels`: binary hit."""
        torch = _torch()
        import torch.nn.functional as F
        Xs = self._standardise(np.asarray(X, float), fit=True)
        order = np.argsort(groups, kind="stable")
        gs, idx = np.unique(groups[order], return_index=True)
        chunks = np.split(order, idx[1:])
        rng = np.random.default_rng(self.p["seed"])
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.p["lr"], weight_decay=self.p["weight_decay"])
        steps = self.p["epochs"] * max(1, len(chunks) // self.p["batch"])
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.p["lr"], total_steps=max(1, steps))
        self.net.train()
        for ep in range(self.p["epochs"]):
            perm = rng.permutation(len(chunks))
            tot = 0.0
            for b0 in range(0, len(perm), self.p["batch"]):
                sel = [chunks[i] for i in perm[b0:b0 + self.p["batch"]]]
                Xb, mask = self._pad([Xs[c] for c in sel], Xs.shape[1])
                rel, lab = self._pad([relevance[c][:, None] for c in sel], 1), self._pad([labels[c][:, None] for c in sel], 1)[0]
                Xb, mask, rel, lab = Xb.to(self.device), mask.to(self.device), rel[0].squeeze(-1).to(self.device), lab.squeeze(-1).to(self.device)
                logits = self.net(Xb, mask)
                logits = logits.masked_fill(mask, -1e9)
                target = rel.masked_fill(mask, 0.0)
                has_pos = target.sum(1) > 0
                loss = self.p["aux_weight"] * F.binary_cross_entropy_with_logits(
                    logits[~mask], lab[~mask], pos_weight=torch.tensor(3.0, device=self.device))
                if has_pos.any():                           # listwise softmax cross-entropy on the graded targets
                    t = target[has_pos]
                    t = t / t.sum(1, keepdim=True)
                    loss = loss + -(t * torch.log_softmax(logits[has_pos], dim=1)).sum(1).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                if sched.last_epoch < steps - 1:
                    sched.step()
                tot += float(loss)
            if (ep + 1) % 20 == 0:
                log(f"    set ranker epoch {ep + 1}/{self.p['epochs']} loss {tot / max(1, len(perm) / self.p['batch']):.4f}")
        return self

    def predict(self, X: np.ndarray, groups: np.ndarray) -> np.ndarray:
        torch = _torch()
        Xs = self._standardise(np.asarray(X, float), fit=False)
        out = np.zeros(len(Xs))
        self.net.eval()
        with torch.no_grad():
            for g in np.unique(groups):
                m = groups == g
                Xb, mask = self._pad([Xs[m]], Xs.shape[1])
                s = self.net(Xb.to(self.device), mask.to(self.device))[0, :int(m.sum())]
                out[m] = s.cpu().numpy()
        return out

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters())
