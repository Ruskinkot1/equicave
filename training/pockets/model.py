"""EquiCave-Net: geometric tensor attention over protein residues, cavity probes and surface points.

Written from scratch in plain PyTorch (no e3nn, no torch_geometric). Every node carries three kinds of channels:
    x : [N, F]        scalars         (degree 0)
    V : [N, F, 3]     vectors         (degree 1, rotate as r)
    T : [N, F, 3, 3]  symmetric traceless tensors (degree 2, rotate as R T R^T)
Messages along an edge j -> i with relative vector r = p_i - p_j, u = r / |r| are built only from equivariant
primitives: vectors V_j, T_j u, u; tensors T_j, (u u^T - I/3), sym-traceless(V_j u^T); scalars from invariant
contractions <V_j, u>, u^T T_j u, |V_j|, |T_j|. Attention weights and all gates are functions of invariants, so the
network is SO(3)-equivariant by construction. Pseudo-scalars (triple products of vector channels) are added to the
invariant stream when `chiral=True`, so mirror images are distinguished (SO(3), not O(3)).

Ablations are switches: `use_vectors`, `use_tensors`, `equivariant` (False = distances-only invariant GNN with the
same depth and width), `chiral`. Node types: 0 residue (CA), 1 cavity probe, 2 surface point; edge types are the
9 ordered type pairs. Heads: residue segmentation, probe occupancy, centre offset (an equivariant vector read out
from V) + confidence, hotspot classes per probe, multi-label site properties from pooled probes.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn

EYE = torch.eye(3)


def sym_traceless(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Symmetric traceless part of the outer product a b^T. a, b: [..., 3] -> [..., 3, 3]."""
    m = 0.5 * (a.unsqueeze(-1) * b.unsqueeze(-2) + b.unsqueeze(-1) * a.unsqueeze(-2))
    tr = (a * b).sum(-1) / 3.0
    return m - tr[..., None, None] * EYE.to(a)


def seg_softmax(logits: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """Softmax of `logits` [E, H] over groups given by `index` [E] (destination nodes)."""
    H = logits.shape[1]
    idx = index[:, None].expand(-1, H)
    mx = torch.full((n, H), -1e30, device=logits.device, dtype=logits.dtype).scatter_reduce(0, idx, logits, "amax", include_self=True)
    ex = torch.exp(logits - mx[index])
    den = torch.zeros((n, H), device=logits.device, dtype=logits.dtype).index_add_(0, index, ex)
    return ex / (den[index] + 1e-9)


class RBF(nn.Module):
    def __init__(self, n: int = 32, cutoff: float = 10.0):
        super().__init__()
        self.register_buffer("mu", torch.linspace(0, cutoff, n))
        self.gamma = (n / cutoff) ** 2 * 0.5
        self.cutoff = cutoff

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        env = 0.5 * (torch.cos(math.pi * d.clamp(max=self.cutoff) / self.cutoff) + 1.0)
        return torch.exp(-self.gamma * (d[:, None] - self.mu[None]) ** 2) * env[:, None]


class EqNorm(nn.Module):
    """LayerNorm on scalars; RMS normalisation of vector / tensor channels (per node, over channels)."""
    def __init__(self, dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.gv = nn.Parameter(torch.ones(dim)); self.gt = nn.Parameter(torch.ones(dim))

    def forward(self, x, V, T):
        x = self.ln(x)
        if V is not None:
            V = V / torch.sqrt((V ** 2).sum((1, 2), keepdim=True) / V.shape[1] + 1e-6) * self.gv[None, :, None]
        if T is not None:
            T = T / torch.sqrt((T ** 2).sum((1, 2, 3), keepdim=True) / T.shape[1] + 1e-6) * self.gt[None, :, None, None]
        return x, V, T


class GeoTensorAttention(nn.Module):
    """One message-passing layer with attention over invariants and gated degree-0/1/2 messages."""
    def __init__(self, dim: int, heads: int = 4, n_rbf: int = 32, cutoff: float = 10.0, n_edge_types: int = 9,
                 use_vectors: bool = True, use_tensors: bool = True, chiral: bool = True, dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0
        self.dim, self.heads, self.use_vectors, self.use_tensors, self.chiral = dim, heads, use_vectors, use_tensors and use_vectors, chiral
        self.rbf = RBF(n_rbf, cutoff)
        self.etype = nn.Embedding(n_edge_types, 16)
        n_inv = 2 * dim + n_rbf + 16 + (2 * dim if use_vectors else 0) + (2 * dim if self.use_tensors else 0)
        self.n_gates = 3 + (3 if use_vectors else 0) + (3 if self.use_tensors else 0)
        self.edge_mlp = nn.Sequential(nn.Linear(n_inv, 2 * dim), nn.SiLU(), nn.Linear(2 * dim, heads + self.n_gates * dim))
        self.wx = nn.Linear(dim, dim, bias=False)
        if use_vectors:
            self.wv = nn.Linear(dim, dim, bias=False)
        if self.use_tensors:
            self.wt = nn.Linear(dim, dim, bias=False)
        n_node_inv = dim + (dim if use_vectors else 0) + (dim if self.use_tensors else 0) + (4 if (chiral and use_vectors) else 0)
        self.node_mlp = nn.Sequential(nn.Linear(n_node_inv, 2 * dim), nn.SiLU(), nn.Linear(2 * dim, 3 * dim))
        self.norm = EqNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.register_buffer("tri", torch.tensor([[0, 1, 2], [3, 4, 5], [1, 2, 6], [0, 4, 7]]))

    def forward(self, x, V, T, pos, edge_index, edge_type):
        src, dst = edge_index                  # message j=src -> i=dst
        n, F, H = x.shape[0], self.dim, self.heads
        r = pos[dst] - pos[src]
        d = r.norm(dim=1).clamp(min=1e-6)
        u = r / d[:, None]
        inv = [x[dst], x[src], self.rbf(d), self.etype(edge_type)]
        if self.use_vectors:
            Vj = V[src]
            vu = (Vj * u[:, None, :]).sum(-1)                              # <V_j, u>       [E, F]
            inv += [vu, Vj.norm(dim=-1)]
        if self.use_tensors:
            Tj = T[src]
            Tu = torch.einsum("efij,ej->efi", Tj, u)                       # T_j u          [E, F, 3]
            utu = (Tu * u[:, None, :]).sum(-1)                             # u^T T_j u      [E, F]
            inv += [utu, Tj.flatten(2).norm(dim=-1)]
        out = self.edge_mlp(torch.cat(inv, -1))
        att = seg_softmax(out[:, :H], dst, n)                              # [E, H]
        a = att.repeat_interleave(F // H, dim=1)                           # [E, F]
        gates = out[:, H:].view(-1, self.n_gates, F)
        g = iter(gates.unbind(1))
        m_x = next(g) * self.wx(x[src])
        if self.use_vectors:
            m_x = m_x + next(g) * vu
        else:
            next(g)
        if self.use_tensors:
            m_x = m_x + next(g) * utu
        else:
            next(g)
        dx = torch.zeros_like(x).index_add_(0, dst, a * m_x)
        dV = dT = None
        if self.use_vectors:
            Wv = torch.einsum("efc,fg->egc", Vj, self.wv.weight.t())
            m_v = next(g)[..., None] * Wv + next(g)[..., None] * u[:, None, :]
            if self.use_tensors:
                m_v = m_v + next(g)[..., None] * Tu
            else:
                next(g)
            dV = torch.zeros_like(V).index_add_(0, dst, a[..., None] * m_v)
        if self.use_tensors:
            Wt = torch.einsum("efij,fg->egij", Tj, self.wt.weight.t())
            uu = (u[:, :, None] * u[:, None, :] - EYE.to(u) / 3.0)[:, None]  # [E, 1, 3, 3]
            m_t = next(g)[..., None, None] * Wt + next(g)[..., None, None] * uu + next(g)[..., None, None] * sym_traceless(Vj, u[:, None, :].expand_as(Vj))
            dT = torch.zeros_like(T).index_add_(0, dst, a[..., None, None] * m_t)
        x = x + self.drop(dx)
        if dV is not None:
            V = V + dV
        if dT is not None:
            T = T + dT
        # node-wise tensor refinement: invariants (and pseudo-scalars) gate equivariant self-interactions
        ninv = [x]
        if self.use_vectors:
            ninv.append(V.norm(dim=-1))
            if self.chiral:
                a_, b_, c_ = V[:, self.tri[:, 0]], V[:, self.tri[:, 1]], V[:, self.tri[:, 2]]
                ninv.append((torch.cross(a_, b_, dim=-1) * c_).sum(-1))     # triple products: flip under reflection
        if self.use_tensors:
            ninv.append(T.flatten(2).norm(dim=-1))
        h, g1, g2 = self.node_mlp(torch.cat(ninv, -1)).chunk(3, -1)
        x = x + h
        if self.use_tensors:
            V = V + g1[..., None] * torch.einsum("nfij,nfj->nfi", T, V)
            T = T + g2[..., None, None] * sym_traceless(V, V)
        return self.norm(x, V, T)


class EquiCaveNet(nn.Module):
    def __init__(self, in_dims: dict, dim: int = 96, layers: int = 4, heads: int = 4, n_rbf: int = 32, cutoff: float = 10.0,
                 n_hot: int = 7, n_props: int = 14, equivariant: bool = True, use_vectors: bool = True, use_tensors: bool = True,
                 chiral: bool = True, use_surface: bool = True, use_probes: bool = True, dropout: float = 0.0, n_init_vec: int = 3):
        super().__init__()
        self.dim, self.equivariant = dim, equivariant
        self.use_vectors = equivariant and use_vectors
        self.use_tensors = self.use_vectors and use_tensors
        self.use_surface, self.use_probes = use_surface, use_probes
        self.embed = nn.ModuleDict({k: nn.Sequential(nn.Linear(v, dim), nn.SiLU(), nn.Linear(dim, dim)) for k, v in in_dims.items()})
        self.type_emb = nn.Embedding(3, dim)
        self.vec_in = nn.Linear(n_init_vec, dim, bias=False)
        self.layers = nn.ModuleList([GeoTensorAttention(dim, heads, n_rbf, cutoff, 9, self.use_vectors, self.use_tensors, chiral, dropout)
                                     for _ in range(layers)])
        self.head_res = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))
        self.head_occ = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))
        self.head_conf = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))
        self.head_hot = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, n_hot))
        self.head_prop = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, n_props))
        self.off_vec = nn.Linear(dim, 1, bias=False)                       # equivariant read-out from vector channels
        self.off_inv = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 3))   # invariant model: plain regression (not equivariant)

    def forward(self, b: dict) -> dict:
        """b: node_type [N], pos [N,3], feat_{res,probe,surf} per type, node_slices, vec0 [N,n_init_vec,3],
        edge_index [2,E], edge_type [E]. Returns logits per node type and the offset vectors of probes."""
        x = torch.zeros(len(b["node_type"]), self.dim, device=b["pos"].device)
        for k, idx in b["slices"].items():
            if len(idx):
                x[idx] = self.embed[k](b[f"feat_{k}"])
        x = x + self.type_emb(b["node_type"])
        V = torch.einsum("nkc,kf->nfc", b["vec0"], self.vec_in.weight.t()) if self.use_vectors else None
        T = sym_traceless(V, V) if self.use_tensors else None
        ei, et = b["edge_index"], b["edge_type"]
        if not self.use_surface or not self.use_probes:                     # ablations: drop node types from the graph
            keep = torch.ones(len(x), dtype=torch.bool, device=x.device)
            if not self.use_surface:
                keep &= b["node_type"] != 2
            if not self.use_probes:
                keep &= b["node_type"] != 1
            m = keep[ei[0]] & keep[ei[1]]
            ei, et = ei[:, m], et[m]
        for layer in self.layers:
            x, V, T = layer(x, V, T, b["pos"], ei, et)
        res, probe = b["slices"]["res"], b["slices"]["probe"]
        out = dict(res_logit=self.head_res(x[res]).squeeze(-1), occ_logit=self.head_occ(x[probe]).squeeze(-1),
                   conf_logit=self.head_conf(x[probe]).squeeze(-1), hot_logit=self.head_hot(x[probe]), x=x)
        if self.use_vectors:
            out["offset"] = torch.einsum("nfc,f->nc", V[probe], self.off_vec.weight[0])
        else:
            out["offset"] = self.off_inv(x[probe])
        out["center"] = b["pos"][probe] + out["offset"]
        if "site_probe_mask" in b and len(b["site_probe_mask"]):          # [S, n_probe] soft membership per labelled site
            w = b["site_probe_mask"]
            xp = x[probe]
            vn = V[probe].norm(dim=-1) if self.use_vectors else torch.zeros_like(xp)
            pooled = torch.cat([w @ xp, w @ vn], -1) / w.sum(1, keepdim=True).clamp(min=1e-6)
            out["prop_logit"] = self.head_prop(pooled)
        return out


def dice_loss(logit: torch.Tensor, y: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    p = torch.sigmoid(logit)
    return 1 - (2 * (p * y).sum() + eps) / (p.sum() + y.sum() + eps)


def seg_loss(logit, y, pos_weight: float = 1.0):
    return dice_loss(logit, y) + Fn.binary_cross_entropy_with_logits(logit, y, pos_weight=torch.tensor(pos_weight, device=logit.device))


def center_set_loss(center: torch.Tensor, conf_logit: torch.Tensor, probe_pos: torch.Tensor, sites: torch.Tensor,
                    radius: float = 8.0, hit: float = 4.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Set loss: every true site is approached by its best probe within `radius`; confidence BCE on (dist < hit)."""
    if len(sites) == 0:
        return center.sum() * 0, Fn.binary_cross_entropy_with_logits(conf_logit, torch.zeros_like(conf_logit))
    d_pred = torch.cdist(center, sites)                                    # [P, S]
    d_probe = torch.cdist(probe_pos, sites)
    reg = []
    for s in range(sites.shape[0]):
        near = d_probe[:, s] < radius
        if near.any():
            reg.append(d_pred[near, s].min())
    reg = torch.stack(reg).mean() if reg else center.sum() * 0
    target = (d_pred.min(1).values < hit).float().detach()
    conf = Fn.binary_cross_entropy_with_logits(conf_logit, target)
    return reg, conf


def focal_bce(logit, y, gamma: float = 2.0, alpha: float = 0.75):
    p = torch.sigmoid(logit)
    ce = Fn.binary_cross_entropy_with_logits(logit, y, reduction="none")
    pt = p * y + (1 - p) * (1 - y)
    w = alpha * y + (1 - alpha) * (1 - y)
    return (w * (1 - pt) ** gamma * ce).mean()
