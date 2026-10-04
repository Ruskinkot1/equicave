"""e3nn backbone for EquiCave-Net: the same network with spherical-harmonic tensor products instead of our
hand-written Cartesian couplings.

Why both exist. The Cartesian layer in `model.py` is written out by hand and limited to degree 2; it is auditable and
has no dependency. This module expresses the same idea with e3nn (MIT): irreducible representations
`0e + 1o + 2e (+ 3o)`, edge features from real spherical harmonics of the edge direction, and a fully connected
tensor product whose weights come from an MLP over invariants. Having both lets us
  1. cross-check the hand-written layer against a reference implementation (`tests/test_model_e3nn.py` asserts both
     are equivariant to the same tolerance and that their invariant outputs agree in distribution),
  2. go above degree 2, which the Cartesian form does not reach, and ablate `lmax`,
  3. report whether the extra machinery pays for itself, which is a question the pocket literature has not answered.

Selected by `model.backbone: e3nn` with `model.lmax`. The node types, edge types, heads and losses are identical, so
the two backbones are interchangeable inside `net_task.py`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import Gate

from .model import EYE, dice_loss, focal_bce, seg_loss, seg_softmax, sym_traceless  # noqa: F401  (shared losses)


def irreps_for(dim: int, lmax: int) -> o3.Irreps:
    """`dim` channels of every degree up to `lmax`, parities as for vectors/tensors of position space."""
    parts = []
    for l in range(lmax + 1):
        p = "e" if l % 2 == 0 else "o"
        parts.append(f"{dim}x{l}{p}")
    return o3.Irreps("+".join(parts))


class E3Layer(nn.Module):
    """One e3nn message-passing layer: spherical-harmonic edge features, weighted tensor product, gated non-linearity."""

    def __init__(self, irreps: o3.Irreps, lmax: int = 2, n_rbf: int = 32, cutoff: float = 10.0, n_edge_types: int = 9,
                 hidden: int = 64, dropout: float = 0.0):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax)
        self.cutoff, self.n_rbf = cutoff, n_rbf
        self.etype = nn.Embedding(n_edge_types, 8)
        self.tp = o3.FullyConnectedTensorProduct(self.irreps, self.sh_irreps, self.irreps, shared_weights=False)
        n_inv = 2 * self.irreps.count(o3.Irrep("0e")) + n_rbf + 8
        self.weight_mlp = nn.Sequential(nn.Linear(n_inv, hidden), nn.SiLU(), nn.Linear(hidden, self.tp.weight_numel))
        self.att_mlp = nn.Sequential(nn.Linear(n_inv, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        scalars = self.irreps.count(o3.Irrep("0e"))
        gated = o3.Irreps([(mul, ir) for mul, ir in self.irreps if ir.l > 0])
        self.gate = Gate(f"{scalars}x0e", [torch.nn.functional.silu], f"{gated.num_irreps}x0e", [torch.sigmoid], gated)
        self.pre_gate = o3.Linear(self.irreps, self.gate.irreps_in)
        self.post_gate = o3.Linear(self.gate.irreps_out, self.irreps)
        self.self_int = o3.Linear(self.irreps, self.irreps)
        self.drop = nn.Dropout(dropout)

    def scalars_of(self, x: torch.Tensor) -> torch.Tensor:
        n = self.irreps.count(o3.Irrep("0e"))
        return x[:, :n]

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor):
        src, dst = edge_index
        r = pos[dst] - pos[src]
        d = r.norm(dim=1).clamp(min=1e-6)
        sh = o3.spherical_harmonics(self.sh_irreps, r / d[:, None], normalize=True, normalization="component")
        rbf = soft_one_hot_linspace(d, 0.0, self.cutoff, self.n_rbf, basis="smooth_finite", cutoff=True)
        inv = torch.cat([self.scalars_of(x)[dst], self.scalars_of(x)[src], rbf, self.etype(edge_type)], -1)
        w = self.weight_mlp(inv)
        msg = self.tp(x[src], sh, w)
        a = seg_softmax(self.att_mlp(inv), dst, x.shape[0])
        out = torch.zeros_like(x).index_add_(0, dst, a * msg)
        x = x + self.drop(out)
        x = x + self.post_gate(self.gate(self.pre_gate(x)))
        return x + self.self_int(x)


class EquiCaveNetE3(nn.Module):
    """EquiCave-Net with the e3nn backbone. Same inputs, same heads, same outputs as `model.EquiCaveNet`."""

    def __init__(self, in_dims: dict, dim: int = 64, layers: int = 4, lmax: int = 2, n_rbf: int = 32, cutoff: float = 10.0,
                 n_hot: int = 7, n_props: int = 14, dropout: float = 0.0, n_init_vec: int = 3, **_ignored):
        super().__init__()
        self.irreps = irreps_for(dim, lmax)
        self.dim, self.lmax = dim, lmax
        self.embed = nn.ModuleDict({k: nn.Sequential(nn.Linear(v, dim), nn.SiLU(), nn.Linear(dim, dim)) for k, v in in_dims.items()})
        self.type_emb = nn.Embedding(3, dim)
        # initial vectors become the l=1 part; scalars the l=0 part; higher degrees start at zero
        self.vec_in = nn.Linear(n_init_vec, dim, bias=False)
        self.layers = nn.ModuleList([E3Layer(self.irreps, lmax, n_rbf, cutoff, 9, 2 * dim, dropout) for _ in range(layers)])
        self.head_res = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))
        self.head_occ = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))
        self.head_conf = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))
        self.head_hot = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, n_hot))
        self.head_prop = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, n_props))
        self.head_aux = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 2))   # auxiliary: buriedness, depth
        self.off_vec = o3.Linear(self.irreps, o3.Irreps("1x1o"))                            # equivariant centre offset

    def _assemble(self, b: dict) -> torch.Tensor:
        n = len(b["node_type"])
        x0 = torch.zeros(n, self.dim, device=b["pos"].device)
        for k, idx in b["slices"].items():
            if len(idx):
                x0[idx] = self.embed[k](b[f"feat_{k}"])
        x0 = x0 + self.type_emb(b["node_type"])
        V = torch.einsum("nkc,kf->nfc", b["vec0"], self.vec_in.weight.t())                  # [N, dim, 3]
        parts = [x0, V.reshape(n, -1)]
        for l in range(2, self.lmax + 1):
            parts.append(torch.zeros(n, self.dim * (2 * l + 1), device=x0.device))
        return torch.cat(parts, -1)

    def _scalars(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :self.dim]

    def _vectors(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.dim:self.dim * 4].reshape(len(x), self.dim, 3)

    def forward(self, b: dict) -> dict:
        x = self._assemble(b)
        for layer in self.layers:
            x = layer(x, b["pos"], b["edge_index"], b["edge_type"])
        res, probe = b["slices"]["res"], b["slices"]["probe"]
        s = self._scalars(x)
        out = dict(res_logit=self.head_res(s[res]).squeeze(-1), occ_logit=self.head_occ(s[probe]).squeeze(-1),
                   conf_logit=self.head_conf(s[probe]).squeeze(-1), hot_logit=self.head_hot(s[probe]),
                   aux=self.head_aux(s), x=x)
        out["offset"] = self.off_vec(x[probe])
        out["center"] = b["pos"][probe] + out["offset"]
        if "site_probe_mask" in b and len(b["site_probe_mask"]):
            w = b["site_probe_mask"]
            vn = self._vectors(x)[probe].norm(dim=-1)
            pooled = torch.cat([w @ s[probe], w @ vn], -1) / w.sum(1, keepdim=True).clamp(min=1e-6)
            out["prop_logit"] = self.head_prop(pooled)
        return out
