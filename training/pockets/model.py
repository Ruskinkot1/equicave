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

Ablations are switches: `use_vectors`, `use_tensors`, `chiral`, and `equivariant`.

**The invariant arm is a fair comparison, not a blinded one.** Setting `equivariant=False` must change exactly one
thing: whether geometry is carried by steerable channels or by invariants. It must not also take the geometry away.
The earlier version dropped `vec0` entirely, which removed the backbone directions, the side-chain chemistry vectors
and the surface normals, so the arm measured "coordinates versus no coordinates" — the confound the pocket literature
has been criticised for. With `invariant_mode="frames"` (the default when `equivariant=False`) every node's initial
vectors are **scalarised in its own local frame**: a frame is built from two of the node's own vectors by
Gram-Schmidt, and each vector is replaced by its three coordinates in that frame, which are rotation-invariant by
construction. Degenerate frames (a probe with no atom within 6 A has a zero second vector) fall back to the axis-aligned
frame, which is a documented approximation rather than a silent one. The invariant arm then sees the same geometric
information, at the same depth and width, and only the mechanism differs. `invariant_mode="distances"` keeps the old
blinded behaviour, available on purpose so the difference between the two can be reported. Node types: 0 residue (CA), 1 cavity probe, 2 surface point; edge types are the
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


def add_into(dest: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """`dest.index_add_(0, index, src)` with the source brought to the destination's dtype.

    Mixed precision has broken this model four times, each time in a different accumulation, and each fix cast one
    more intermediate by hand. The cause is structural: autocast keeps a list of operations it forces to float32,
    the list differs between its CPU and CUDA implementations, and a float32 value multiplied into a bf16 stream
    promotes the product back to float32 -- which `index_add_` refuses. Casting at the accumulation is the only
    place where every such path has to pass, so it is done here instead of at each producer.
    """
    return dest.index_add_(0, index, src.to(dest.dtype))


def seg_mean(v: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """Mean of `v` [E, F] over the edges sharing a destination, in float32 for the reason `seg_softmax` gives."""
    val = v.float()
    tot = torch.zeros((n, val.shape[1]), device=val.device, dtype=val.dtype).index_add_(0, index, val)
    cnt = torch.zeros((n, 1), device=val.device, dtype=val.dtype).index_add_(
        0, index, torch.ones((len(index), 1), device=val.device, dtype=val.dtype))
    return tot / cnt.clamp(min=1.0)


def seg_softmax(logits: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """Softmax of `logits` [E, H] over groups given by `index` [E] (destination nodes).

    Computed in float32 and returned in the input dtype, which is both the numerically right thing for a
    normalisation and the only form that survives mixed precision. Autocast keeps a list of operations it forces
    to float32, `exp` among them on CUDA, so a buffer allocated as `logits.dtype` (bf16) and an `exp` result
    (float32) met in `index_add_`, which raises outright. The list differs between the CPU and CUDA autocast
    implementations -- on CPU `exp` is not promoted -- so a CPU bf16 test passes while the CUDA run dies on its
    first batch, which is exactly what happened. Doing the whole reduction in float32 cannot be caught out by
    either policy.
    """
    dt = logits.dtype
    lg = logits.float()
    H = lg.shape[1]
    idx = index[:, None].expand(-1, H)
    mx = torch.full((n, H), -1e30, device=lg.device, dtype=lg.dtype).scatter_reduce(0, idx, lg, "amax", include_self=True)
    ex = torch.exp(lg - mx[index])
    den = torch.zeros((n, H), device=lg.device, dtype=lg.dtype).index_add_(0, index, ex)
    return (ex / (den[index] + 1e-9)).to(dt)


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
                 use_vectors: bool = True, use_tensors: bool = True, chiral: bool = True, dropout: float = 0.0,
                 n_edge_scalar: int = 0, attn_kernel: str = "dot"):
        super().__init__()
        assert dim % heads == 0
        assert attn_kernel in ("dot", "gaussian")
        self.dim, self.heads, self.use_vectors, self.use_tensors, self.chiral = dim, heads, use_vectors, use_tensors and use_vectors, chiral
        self.attn_kernel = attn_kernel
        if attn_kernel == "gaussian":
            # One temperature per head, passed through softplus so it stays positive. Initialised at
            # softplus(0.5413) = 1, which makes the kernel's starting width the identity scale of the normalised
            # differences; heads then specialise on their own.
            self.xi = nn.Parameter(torch.full((heads,), 0.5413))
        self.n_edge_scalar = n_edge_scalar
        self.rbf = RBF(n_rbf, cutoff)
        self.etype = nn.Embedding(n_edge_types, 16)
        n_inv = 2 * dim + n_rbf + 16 + n_edge_scalar + (2 * dim if use_vectors else 0) + (2 * dim if self.use_tensors else 0)
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

    def attn_logits(self, learned, x, src, dst, n):
        """Attention logits: the edge MLP's own, or a Gaussian kernel on variance-normalised feature differences.

        The Gaussian form follows Gaussian Dynamic Attention (Wang et al. 2026, arXiv 2603.19817), recorded in
        docs/PROVENANCE.md; the implementation here is ours. The score is

            alpha_ij  ~  exp( -|| (x_i - x_j) / sd_{N(i)} ||^2 / (2 xi_h) )

        with the standard deviation taken per destination node over its own incoming edges, so the width adapts to
        how varied a neighbourhood is rather than being a single learned projection. Two properties matter here.
        It reads only the invariant stream `x`, so the layer stays exactly as equivariant as it was -- the kernel
        cannot see a rotation. And it is a per-head scalar rather than a key-query matrix, so it adds `heads`
        parameters instead of O(dim^2); on this model that is 8 against the ~33 k a projection pair would cost.

        The statistics are per destination node and therefore never cross a structure boundary, since no edge does.
        """
        if self.attn_kernel == "dot":
            return learned
        H = self.heads
        df = (x[dst] - x[src]).float()
        mu = seg_mean(df, dst, n)
        var = seg_mean((df - mu[dst]) ** 2, dst, n)
        ds = df / (var[dst] + 1e-6).sqrt()
        q = ds.view(-1, H, self.dim // H).pow(2).mean(-1)                  # [E, H], mean so width is dim-free
        return (-q / (2.0 * (nn.functional.softplus(self.xi) + 1e-3))).to(learned.dtype)

    def forward(self, x, V, T, pos, edge_index, edge_type, edge_scalar=None):
        src, dst = edge_index                  # message j=src -> i=dst
        n, F, H = x.shape[0], self.dim, self.heads
        # Mixed precision: the residual stream `x` carries the working dtype. Under autocast a Linear returns bf16
        # while a softmax runs in float32 and a coordinate difference stays float32, so a product of the two
        # promotes back to float32 and `index_add_` into a bf16 buffer raises. Edge geometry is brought to the
        # stream's dtype once, here, rather than patched at each accumulation.
        dt = x.dtype
        V = V.to(dt) if V is not None else None
        T = T.to(dt) if T is not None else None
        r = (pos[dst] - pos[src]).to(dt)
        d = r.norm(dim=1).clamp(min=1e-6)
        u = r / d[:, None]
        inv = [x[dst], x[src], self.rbf(d), self.etype(edge_type).to(dt)]
        if self.n_edge_scalar:
            inv.append(edge_scalar if edge_scalar is not None else torch.zeros(len(src), self.n_edge_scalar, device=x.device, dtype=x.dtype))
        if self.use_vectors:
            Vj = V[src]
            vu = (Vj * u[:, None, :]).sum(-1)                              # <V_j, u>       [E, F]
            inv += [vu, Vj.norm(dim=-1)]
        if self.use_tensors:
            Tj = T[src]
            Tu = torch.einsum("efij,ej->efi", Tj, u)                       # T_j u          [E, F, 3]
            utu = (Tu * u[:, None, :]).sum(-1)                             # u^T T_j u      [E, F]
            inv += [utu, Tj.flatten(2).norm(dim=-1)]
        # `.to(dt)`: the MLP ends in a normalisation, which autocast runs in float32, so its output -- and the
        # gates and attention logits taken from it -- come back float32 even when the stream is bf16.
        out = self.edge_mlp(torch.cat(inv, -1)).to(dt)
        att = seg_softmax(self.attn_logits(out[:, :H], x, src, dst, n), dst, n)   # [E, H]
        a = att.repeat_interleave(F // H, dim=1).to(dt)                    # [E, F]; softmax runs in fp32
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
        dx = add_into(torch.zeros_like(x), dst, a * m_x)
        dV = dT = None
        if self.use_vectors:
            Wv = torch.einsum("efc,fg->egc", Vj, self.wv.weight.t())
            m_v = next(g)[..., None] * Wv + next(g)[..., None] * u[:, None, :]
            if self.use_tensors:
                m_v = m_v + next(g)[..., None] * Tu
            else:
                next(g)
            dV = add_into(torch.zeros_like(V), dst, a[..., None] * m_v)
        if self.use_tensors:
            Wt = torch.einsum("efij,fg->egij", Tj, self.wt.weight.t())
            uu = (u[:, :, None] * u[:, None, :] - EYE.to(u) / 3.0)[:, None]  # [E, 1, 3, 3]
            m_t = next(g)[..., None, None] * Wt + next(g)[..., None, None] * uu + next(g)[..., None, None] * sym_traceless(Vj, u[:, None, :].expand_as(Vj))
            dT = add_into(torch.zeros_like(T), dst, a[..., None, None] * m_t)
        x = x + self.drop(dx).to(x.dtype)
        if dV is not None:
            V = V + dV.to(V.dtype)
        if dT is not None:
            T = T + dT.to(T.dtype)
        # node-wise tensor refinement: invariants (and pseudo-scalars) gate equivariant self-interactions
        ninv = [x]
        if self.use_vectors:
            ninv.append(V.norm(dim=-1))
            if self.chiral:
                a_, b_, c_ = V[:, self.tri[:, 0]], V[:, self.tri[:, 1]], V[:, self.tri[:, 2]]
                ninv.append((torch.cross(a_, b_, dim=-1) * c_).sum(-1))     # triple products: flip under reflection
        if self.use_tensors:
            ninv.append(T.flatten(2).norm(dim=-1))
        h, g1, g2 = self.node_mlp(torch.cat(ninv, -1)).to(dt).chunk(3, -1)
        x = x + h
        if self.use_tensors:
            V = V + g1[..., None] * torch.einsum("nfij,nfj->nfi", T, V)
            T = T + g2[..., None, None] * sym_traceless(V, V)
        # EqNorm normalises, and autocast runs normalisation in float32, so without this the next layer receives
        # float32 channels while its residual buffers are bf16 -- the same mismatch one layer later.
        xo, Vo, To = self.norm(x, V, T)
        return xo.to(dt), None if Vo is None else Vo.to(dt), None if To is None else To.to(dt)


N_SITE_SUMMARY = 7          # invariant per-site scalars, see SiteDecoder.summaries


class SiteDecoder(nn.Module):
    """Explicit site tokens that compete with one another, instead of probes scored one at a time.

    Why this exists. Every other head in this network is per node: a probe's confidence is computed from its own
    neighbourhood and nothing tells it that another pocket in the same protein is the better answer. The measured
    failure is exactly that -- on the COACH420 structures our first prediction gets wrong, the correct candidate is
    ranked *second* in 24 of 53 cases and within the first five in 41 of 53, while 34 of 53 first predictions are
    more than 8 A from the ligand, so they are a different pocket rather than a near miss. A model whose scores are
    computed independently cannot represent "this pocket, not that one"; this module can, because the sites are
    tokens in one attention stack and each score is a function of the whole list.

    It also makes the model propose the objects the metric scores. The sites are formed by the same rule the
    evaluation uses -- confidence order, greedy non-maximum suppression at `nms` angstroms over predicted centres --
    so the list the decoder ranks is the list top-1 is computed from, and the ranking loss is applied to it.

    What it deliberately does not do is move the centres. Six definitions of a candidate's centre were measured and
    all gave DCC within noise of each other, so centre refinement is a closed avenue; this module only reranks, and
    a site's centre stays the confidence-weighted mean of its member probes' predicted centres.

    Everything the tokens see is invariant under a global rotation (pooled node features, vector norms, the six
    scalars of `summaries`, and pairwise distances as an attention bias), so the site scores are invariant while the
    centres stay equivariant by construction, being convex combinations of positions.
    """

    AGGREGATORS = ("sum_sq", "mean", "max", "sum")

    def __init__(self, dim: int, heads: int = 4, layers: int = 2, n_sites: int = 32, nms: float = 6.0,
                 membership: float = 8.0, n_rbf: int = 16, cutoff: float = 30.0, dropout: float = 0.0,
                 agg: str = "sum_sq"):
        super().__init__()
        assert agg in SiteDecoder.AGGREGATORS, f"agg must be one of {SiteDecoder.AGGREGATORS}"
        self.dim, self.heads, self.n_sites, self.nms, self.membership = dim, heads, n_sites, nms, membership
        self.agg = agg
        self.hd = dim // heads
        self.token = nn.Sequential(nn.Linear(2 * dim + N_SITE_SUMMARY, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.rbf = RBF(n_rbf, cutoff)
        self.qkv = nn.ModuleList([nn.Linear(dim, 3 * dim) for _ in range(layers)])
        self.bias = nn.ModuleList([nn.Sequential(nn.Linear(n_rbf, heads)) for _ in range(layers)])
        self.proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layers)])
        self.ff = nn.ModuleList([nn.Sequential(nn.Linear(dim, 2 * dim), nn.SiLU(), nn.Linear(2 * dim, dim))
                                 for _ in range(layers)])
        self.n1 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        self.n2 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))

    @torch.no_grad()
    def seeds(self, conf: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        """Indices of the probes that seed the sites: confidence order under greedy NMS, the evaluation's own rule."""
        order = torch.argsort(conf, descending=True)
        keep: list[int] = []
        for i in order.tolist():
            if not keep:
                keep.append(i)
            elif (center[i] - center[keep]).norm(dim=-1).min() > self.nms:
                keep.append(i)
            if len(keep) >= self.n_sites:
                break
        return torch.tensor(keep, dtype=torch.long, device=conf.device)

    def aggregate(self, w: torch.Tensor, occ: torch.Tensor) -> torch.Tensor:
        """Collapse a site's member probes into one occupancy score, by the rule `agg` names.

        The three accurate methods in this field use three different rules and nobody has compared them. P2Rank
        sums the *squares* of its per-point ligandability over a cluster and reports that PRANK tested the mean
        and rejected it; GrASP and VN-EGNN both average; YuelPocket takes the peak. The choice is a hyperparameter
        in every one of them, never an ablation, so this is the comparison the field is missing rather than a
        tuning knob.

        Sum-of-squares rewards a site that has several confident probes over one that has many lukewarm ones,
        which is why it is the default here: it is the rule the strongest ranker uses. The membership weights
        multiply the term rather than normalising it, except for `mean`, where normalising is the point.
        """
        o = occ[:, None]
        if self.agg == "mean":
            return w @ o / w.sum(1, keepdim=True).clamp(min=1e-6)
        if self.agg == "max":
            return (w * occ[None, :]).max(1, keepdim=True).values
        if self.agg == "sum":
            return torch.log1p(w @ o)
        return torch.log1p(w @ (o ** 2))                      # sum_sq, P2Rank's rule and the default

    def summaries(self, w: torch.Tensor, occ: torch.Tensor, conf: torch.Tensor, center: torch.Tensor,
                  site_center: torch.Tensor, probe_pos: torch.Tensor) -> torch.Tensor:
        """[S, N_SITE_SUMMARY] invariant scalars per site: how much pocket there is and how sure of it the probes are.

        Six, in order: the membership mass (how many probes the site gathers), the mean and maximum occupancy
        probability of its members, their mean confidence, the spread of their predicted centres, and the distance
        from the site to the centroid of all probes -- the centrality that the gradient-boosted ranker found to be
        among its most useful columns, which no per-probe head can see.
        """
        mass = w.sum(1, keepdim=True)
        agg_occ = self.aggregate(w, occ)
        mean_occ = w @ occ[:, None] / mass.clamp(min=1e-6)
        max_occ = torch.stack([occ[w[s] > 1e-3].max() if (w[s] > 1e-3).any() else occ.new_zeros(())
                               for s in range(len(w))])[:, None]
        mean_conf = w @ conf[:, None] / mass.clamp(min=1e-6)
        spread = ((w @ (center ** 2) / mass.clamp(min=1e-6)) - (w @ center / mass.clamp(min=1e-6)) ** 2)
        spread = spread.clamp(min=0).sum(-1, keepdim=True).sqrt()
        centrality = (site_center - probe_pos.mean(0, keepdim=True)).norm(dim=-1, keepdim=True)
        return torch.cat([torch.log1p(mass), agg_occ, mean_occ, max_occ, mean_conf, spread, centrality / 10.0], -1)

    def forward(self, xp: torch.Tensor, vnorm: torch.Tensor, center: torch.Tensor, probe_pos: torch.Tensor,
                conf_logit: torch.Tensor, occ_logit: torch.Tensor) -> dict:
        if not len(conf_logit):
            z = xp.new_zeros(0)
            return dict(site_logit=z, site_center=xp.new_zeros((0, 3)), site_member=xp.new_zeros((0, 0)))
        # The geometry arrives in float32 (positions are never cast) and the features in the stream's dtype, so
        # membership and centres are computed in float32 -- they are coordinates -- and brought to the stream only
        # where they meet the weights.
        dt = xp.dtype
        conf, occ = torch.sigmoid(conf_logit.float()), torch.sigmoid(occ_logit.float())
        center = center.float()
        idx = self.seeds(conf_logit.detach().float(), center.detach())
        seed_c = center[idx]                                             # [S, 3]
        d = torch.cdist(seed_c, center)                                  # [S, P]
        # Soft membership, so the gradient reaches every probe that supports a site rather than only its seed.
        w = torch.exp(-(d ** 2) / (2 * self.membership ** 2)) * (d < 2 * self.membership)
        w = w * conf[None, :].clamp(min=1e-3)                            # a probe speaks for a site in proportion to its own confidence
        mass = w.sum(1, keepdim=True).clamp(min=1e-6)
        site_center = w @ center / mass                                  # convex combination of positions: equivariant
        wd, massd = w.to(dt), mass.to(dt)
        feats = torch.cat([wd @ xp / massd, wd @ vnorm.to(dt) / massd,
                           self.summaries(w, occ, conf, center, site_center, probe_pos.float()).to(dt)], -1)
        h = self.token(feats)
        bias_in = self.rbf(torch.cdist(site_center, site_center).reshape(-1)).reshape(len(h), len(h), -1).to(dt)
        for qkv, bias, proj, ff, n1, n2 in zip(self.qkv, self.bias, self.proj, self.ff, self.n1, self.n2):
            q, k, v = qkv(n1(h)).chunk(3, dim=-1)
            q = q.reshape(-1, self.heads, self.hd); k = k.reshape(-1, self.heads, self.hd)
            v = v.reshape(-1, self.heads, self.hd)
            logits = torch.einsum("qhd,khd->qkh", q, k) / (self.hd ** 0.5) + bias(bias_in)
            a = self.drop(torch.softmax(logits, dim=1))
            h = h + proj(torch.einsum("qkh,khd->qhd", a, v).reshape(len(h), -1))
            h = h + ff(n2(h))
        return dict(site_logit=self.head(h).squeeze(-1), site_center=site_center, site_member=w, site_seed=idx)


class EquiCaveNet(nn.Module):
    def __init__(self, in_dims: dict, dim: int = 96, layers: int = 4, heads: int = 4, n_rbf: int = 32, cutoff: float = 10.0,
                 n_hot: int = 7, n_props: int = 14, equivariant: bool = True, use_vectors: bool = True, use_tensors: bool = True,
                 chiral: bool = True, use_surface: bool = True, use_probes: bool = True, dropout: float = 0.0, n_init_vec: int = 3,
                 recycles: int = 0, n_edge_scalar: int = 0, n_res_types: int = 21,
                 invariant_mode: str = "frames", site_decoder: bool = True, site_layers: int = 2,
                 n_site_tokens: int = 32, site_nms: float = 6.0, site_membership: float = 8.0,
                 probe_update: str = "center", flow_step: float = 2.0, attn_kernel: str = "dot",
                 site_agg: str = "sum_sq", **_ignored):
        super().__init__()
        self.dim, self.equivariant = dim, equivariant
        self.recycles, self.n_edge_scalar = recycles, n_edge_scalar
        self.invariant_mode = invariant_mode
        self.n_init_vec = n_init_vec
        self.use_vectors = equivariant and use_vectors
        self.use_tensors = self.use_vectors and use_tensors
        self.use_surface, self.use_probes = use_surface, use_probes
        # the invariant arm receives the same geometry as scalars: 3 coordinates per initial vector in a local frame
        self.n_scalarised = 3 * n_init_vec if (not equivariant and invariant_mode == "frames") else 0
        self.embed = nn.ModuleDict({k: nn.Sequential(nn.Linear(v + self.n_scalarised, dim), nn.SiLU(), nn.Linear(dim, dim))
                                    for k, v in in_dims.items()})
        self.type_emb = nn.Embedding(3, dim)
        self.vec_in = nn.Linear(n_init_vec, dim, bias=False)
        self.layers = nn.ModuleList([GeoTensorAttention(dim, heads, n_rbf, cutoff, 9, self.use_vectors, self.use_tensors, chiral,
                                                        dropout, n_edge_scalar, attn_kernel) for _ in range(layers)])
        self.head_res = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))
        self.head_occ = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))
        self.head_conf = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))
        self.head_hot = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, n_hot))
        self.head_prop = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, n_props))
        self.head_seq = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, n_res_types))   # masked-residue task
        self.head_pot = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 7))              # masked-potential task
        self.head_size = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, 1))         # ligand heavy atoms of the site
        self.off_vec = nn.Linear(dim, 1, bias=False)                       # equivariant read-out from vector channels
        # A second equivariant vector per probe, supervised by a cosine loss towards the nearest ligand heavy atom
        # (P12). Its purpose is supervision density: the centre set loss matches one proposal per true site, so the
        # geometric gradient reaches at most ~3 of 768 probes per structure, while this reaches every probe that sits
        # within reach of exactly one site -- one to two orders of magnitude more geometric signal at 128 parameters.
        # Two independent pocket ablations (EquiPocket ICML 2024, GDEGAN 2026) report a gain for this mechanism, both
        # concentrated in DCC, which is the axis where our gradient-boosted ranker structurally cannot help.
        self.dir_vec = nn.Linear(dim, 1, bias=False)
        self.dir_inv = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 3))   # invariant arms: not equivariant
        self.off_inv = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 3))   # invariant model: plain regression (not equivariant)
        # The sites as entities that compete, which no per-probe head can represent. See SiteDecoder.
        self.site_decoder = SiteDecoder(dim, heads, site_layers, n_site_tokens, site_nms, site_membership,
                                        dropout=dropout, agg=site_agg) if site_decoder else None
        # How a probe moves between passes. "center" jumps it to its own predicted centre, which is one large
        # uncorrected step; "flow" walks it along the direction head's prediction in `flow_step`-angstrom steps, so
        # the only two mechanisms the ablation found to matter -- the probes and the dense direction signal -- drive
        # the refinement together, and every intermediate position is supervised.
        self.probe_update, self.flow_step = probe_update, flow_step

    def local_frame_scalars(self, vec0: torch.Tensor) -> torch.Tensor:
        """[N, 3 * K] rotation-invariant coordinates of a node's own vectors in a frame built from two of them.

        e1 from the first non-degenerate vector, e2 the Gram-Schmidt complement of the second, e3 their cross
        product; where that is degenerate the axis-aligned frame is used, which is an approximation only for nodes
        whose own geometry is missing (a probe with no atom within 6 A). Each vector's three coordinates in the frame
        are invariant under a global rotation, so the invariant arm sees the same geometry the equivariant one does.
        """
        v1, v2 = vec0[:, 0], vec0[:, 1] if vec0.shape[1] > 1 else vec0[:, 0]
        e1 = v1 / v1.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        bad1 = v1.norm(dim=-1) < 1e-6
        e1 = torch.where(bad1[:, None], torch.tensor([1.0, 0.0, 0.0], device=vec0.device).expand_as(e1), e1)
        proj = v2 - (v2 * e1).sum(-1, keepdim=True) * e1
        bad2 = proj.norm(dim=-1) < 1e-6
        fallback = torch.tensor([0.0, 1.0, 0.0], device=vec0.device).expand_as(proj)
        proj = torch.where(bad2[:, None], fallback - (fallback * e1).sum(-1, keepdim=True) * e1, proj)
        e2 = proj / proj.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        e3 = torch.cross(e1, e2, dim=-1)
        R = torch.stack([e1, e2, e3], dim=1)                       # [N, 3, 3] rows are the frame axes
        return torch.einsum("nij,nkj->nki", R, vec0).reshape(len(vec0), -1)

    def trunk(self, b: dict, pos: torch.Tensor, ei: torch.Tensor, et: torch.Tensor, es):
        """One pass of the message-passing trunk at the given probe positions. Returns (x, V, T).

        Mixed precision. Under `torch.autocast` a Linear returns the autocast dtype (bf16) while a buffer allocated
        here and an nn.Embedding lookup stay float32, and writing one into the other raises outright -- which is how
        this surfaced, on the first bf16 step of a GPU run. The node features are embedded first, the working dtype
        is taken from what those layers actually produced, and everything entering the layer stack is brought to it,
        so the pass runs in one dtype whether autocast is on or off instead of silently promoting back to float32
        at the first addition (which would keep bf16 enabled and deliver none of its speed).
        """
        inv_scalars = self.local_frame_scalars(b["vec0"]) if self.n_scalarised else None
        parts = {}
        for k, idx in b["slices"].items():
            if len(idx):
                f = b[f"feat_{k}"]
                if inv_scalars is not None:
                    f = torch.cat([f, inv_scalars[idx]], -1)
                parts[k] = self.embed[k](f)
        dt = next(iter(parts.values())).dtype if parts else pos.dtype
        x = torch.zeros(len(b["node_type"]), self.dim, device=pos.device, dtype=dt)
        for k, v in parts.items():
            x[b["slices"][k]] = v
        x = x + self.type_emb(b["node_type"]).to(dt)
        V = torch.einsum("nkc,kf->nfc", b["vec0"].to(dt), self.vec_in.weight.t().to(dt)) if self.use_vectors else None
        T = sym_traceless(V, V) if self.use_tensors else None
        for layer in self.layers:
            x, V, T = layer(x, V, T, pos, ei, et, es)
        return x, V, T

    def forward(self, b: dict) -> dict:
        """b: node_type [N], pos [N,3], feat_{res,probe,surf} per type, node_slices, vec0 [N,n_init_vec,3],
        edge_index [2,E], edge_type [E], optional edge_scalar [E,n_edge_scalar]. Returns logits per node type and the
        offset vectors of probes. With `recycles > 0` the probes are moved to their predicted centres and the pass is
        repeated with their edges rebuilt; every pass is returned in `passes` for deep supervision."""
        ei, et = b["edge_index"], b["edge_type"]
        es = b.get("edge_scalar") if self.n_edge_scalar else None
        if not self.use_surface or not self.use_probes:                     # ablations: drop node types from the graph
            keep = torch.ones(len(b["node_type"]), dtype=torch.bool, device=b["pos"].device)
            if not self.use_surface:
                keep &= b["node_type"] != 2
            if not self.use_probes:
                keep &= b["node_type"] != 1
            m = keep[ei[0]] & keep[ei[1]]
            ei, et = ei[:, m], et[m]
            es = es[m] if es is not None else None
        res, probe = b["slices"]["res"], b["slices"]["probe"]
        pos = b["pos"]
        passes = []
        for cycle in range(self.recycles + 1):
            if cycle:                                      # move the probes and relink them
                prev = passes[-1]
                if self.probe_update == "flow" and "dir" in prev:
                    step = Fn.normalize(prev["dir"].detach(), dim=-1) * self.flow_step
                    moved = prev["probe_pos"].detach() + step
                else:
                    moved = prev["center"].detach()
                pos, ei, et, es = self.relink(b, moved, ei, et, es)
            x, V, T = self.trunk(b, pos, ei, et, es)
            o = dict(res_logit=self.head_res(x[res]).squeeze(-1), occ_logit=self.head_occ(x[probe]).squeeze(-1),
                     conf_logit=self.head_conf(x[probe]).squeeze(-1), hot_logit=self.head_hot(x[probe]),
                     seq_logit=self.head_seq(x[res]), pot_logit=self.head_pot(x[probe]), x=x, probe_pos=pos[probe])
            if self.use_vectors:
                o["offset"] = torch.einsum("nfc,f->nc", V[probe], self.off_vec.weight[0])
                o["dir"] = torch.einsum("nfc,f->nc", V[probe], self.dir_vec.weight[0])
            else:
                o["offset"] = self.off_inv(x[probe])
                o["dir"] = self.dir_inv(x[probe])
            o["center"] = pos[probe] + o["offset"]
            if "site_probe_mask" in b and len(b["site_probe_mask"]):      # [S, n_probe] soft membership per labelled site
                w = b["site_probe_mask"]
                xp = x[probe]
                vn = V[probe].norm(dim=-1) if self.use_vectors else torch.zeros_like(xp)
                pooled = torch.cat([w @ xp, w @ vn], -1) / w.sum(1, keepdim=True).clamp(min=1e-6)
                o["prop_logit"] = self.head_prop(pooled)
                o["size_pred"] = self.head_size(pooled).squeeze(-1)
            passes.append(o)
        last = passes[-1]
        if self.site_decoder is not None:
            # Only on the final pass: the decoder ranks, and ranking an unconverged probe cloud buys nothing while
            # costing a full attention stack per cycle.
            vnorm = (V[probe].norm(dim=-1) if self.use_vectors else torch.zeros_like(x[probe]))
            last.update(self.site_decoder(x[probe], vnorm, last["center"], last["probe_pos"],
                                          last["conf_logit"], last["occ_logit"]))
        out = dict(last); out["passes"] = passes
        return out

    def relink(self, b: dict, new_probe_pos: torch.Tensor, ei, et, es):
        """Probe positions replaced by the predicted centres; probe edges rebuilt by distance, other edges kept."""
        probe = b["slices"]["probe"]
        pos = b["pos"].clone()
        pos[probe] = new_probe_pos
        keep = (b["node_type"][ei[0]] != 1) & (b["node_type"][ei[1]] != 1)
        ei_keep, et_keep = ei[:, keep], et[keep]
        es_keep = es[keep] if es is not None else None
        others = torch.cat([b["slices"]["res"], b["slices"]["surf"]])
        if len(others) == 0 or len(probe) == 0:
            return pos, ei_keep, et_keep, es_keep
        d = torch.cdist(pos[probe], pos[others])
        k = min(16, len(others))
        idx = d.topk(k, largest=False).indices                      # probe -> nearest residues/surface points
        src = others[idx.reshape(-1)]
        dst = probe.repeat_interleave(k)
        dd = torch.cdist(pos[probe], pos[probe])
        kp = min(8, len(probe))
        pidx = dd.topk(kp, largest=False).indices
        psrc = probe[pidx.reshape(-1)]; pdst = probe.repeat_interleave(kp)
        new_src = torch.cat([src, dst, psrc]); new_dst = torch.cat([dst, src, pdst])
        m = new_src != new_dst
        new_src, new_dst = new_src[m], new_dst[m]
        new_et = b["node_type"][new_src] * 3 + b["node_type"][new_dst]
        ei2 = torch.cat([ei_keep, torch.stack([new_src, new_dst])], 1)
        et2 = torch.cat([et_keep, new_et])
        es2 = None
        if es is not None:
            es2 = torch.cat([es_keep, torch.zeros(len(new_et), es.shape[1], device=es.device, dtype=es.dtype)])
        return pos, ei2, et2, es2


def dice_loss(logit: torch.Tensor, y: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    p = torch.sigmoid(logit)
    return 1 - (2 * (p * y).sum() + eps) / (p.sum() + y.sum() + eps)


def seg_loss(logit, y, pos_weight: float = 1.0):
    return dice_loss(logit, y) + Fn.binary_cross_entropy_with_logits(logit, y, pos_weight=torch.tensor(pos_weight, device=logit.device))


def center_set_loss(center: torch.Tensor, conf_logit: torch.Tensor, probe_pos: torch.Tensor, sites: torch.Tensor,
                    radius: float = 8.0, hit: float = 4.0, hungarian: bool = True,
                    n_proposals: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Set loss over sites and confidence BCE.

    `hungarian=True` assigns each true site to a *different* proposal by optimal one-to-one matching
    (`scipy.optimize.linear_sum_assignment` over the most confident `n_proposals` probes), which is what a detector
    should be trained on: one prediction per site, no single probe claiming every site at once. With `hungarian=False`
    the loss is the older "each site is approached by its nearest probe within `radius`" form, kept for the ablation.
    """
    if len(sites) == 0:
        return center.sum() * 0, Fn.binary_cross_entropy_with_logits(conf_logit, torch.zeros_like(conf_logit))
    d_pred = torch.cdist(center, sites)                                    # [P, S]
    if hungarian and len(center) and len(sites):
        from scipy.optimize import linear_sum_assignment
        k = min(n_proposals, len(center))
        top = torch.topk(conf_logit, k).indices
        sub = d_pred[top]                                                  # [k, S]
        rows, cols = linear_sum_assignment(sub.detach().cpu().numpy())
        reg = sub[rows, cols].mean()
    else:
        d_probe = torch.cdist(probe_pos, sites)
        terms = [d_pred[d_probe[:, s] < radius, s].min() for s in range(sites.shape[0]) if (d_probe[:, s] < radius).any()]
        reg = torch.stack(terms).mean() if terms else center.sum() * 0
    target = (d_pred.min(1).values < hit).float().detach()
    conf = Fn.binary_cross_entropy_with_logits(conf_logit, target)
    return reg, conf


def masked_residue_loss(seq_logit: torch.Tensor, true_types: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Self-supervised auxiliary task: recover the identity of residues whose input one-hot was replaced by a mask.

    The target is not derivable from the remaining inputs, so unlike predicting buriedness (which is an input
    feature) this is real extra supervision: the trunk has to use structural context to name a residue.
    """
    if mask.sum() == 0:
        return seq_logit.sum() * 0
    return Fn.cross_entropy(seq_logit[mask], true_types[mask])


def potential_loss(pot_logit: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Recover the hidden interaction potential of masked probes: dense, free supervision on exactly the
    chemistry the ranker's most valuable feature group is built from."""
    if mask is None or mask.sum() == 0:
        return pot_logit.sum() * 0
    return Fn.binary_cross_entropy_with_logits(pot_logit[mask], target[mask])


def size_loss(size_pred: torch.Tensor, n_atoms: torch.Tensor) -> torch.Tensor:
    """Auxiliary regression: how many ligand heavy atoms the site holds (log scale), a proxy for pocket capacity."""
    if len(n_atoms) == 0:
        return size_pred.sum() * 0
    return Fn.smooth_l1_loss(size_pred, torch.log1p(n_atoms.float()))


def direction_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cosine loss from each usable probe's predicted vector to the direction of the nearest ligand heavy atom (P12).

    Only the direction is supervised, not the length, so the head is free to use its magnitude for anything else; an
    unnormalisable prediction (a probe whose vector channels collapsed to zero) contributes the maximum loss of 1
    rather than a NaN. Returns 0 when no probe is usable, so a structure with overlapping sites costs nothing.
    """
    m = mask > 0.5
    if not m.any():
        return pred.sum() * 0.0
    cos = torch.nn.functional.cosine_similarity(pred[m], target[m], dim=-1, eps=1e-6)
    return (1.0 - cos).mean()


def listwise_site_loss(conf_logit: torch.Tensor, center: torch.Tensor, site_centers: torch.Tensor,
                       probe_pos: torch.Tensor, hit: float = 4.0, top: int = 64) -> torch.Tensor:
    """Softmax cross-entropy over the probes of one structure, with the probes near a true site as the targets (P2).

    The network has no ordering objective at all: every other head is a per-node or per-site term, so nothing in the
    loss says that *this* probe should outrank *that* one in the same protein, which is exactly the decision the
    benchmark's top-1 measures. This is a listwise term over the structure's own probes -- the same shape as the
    ranking problem -- and it is normalised per structure, so a protein with many probes does not dominate.

    The positive set is every probe whose predicted centre lands within `hit` of a true site centre; the loss is the
    negative log of the total probability mass those probes receive. `top` caps the list at the most confident
    probes so the gradient concentrates where the ordering is actually decided.
    """
    if not len(site_centers) or not len(conf_logit):
        return conf_logit.sum() * 0.0
    k = min(top, len(conf_logit))
    idx = conf_logit.topk(k).indices
    d = torch.cdist(center[idx], site_centers).min(1).values
    pos = d <= hit
    if not pos.any() or pos.all():                     # no decision to make: nothing to order, or nothing to reject
        return conf_logit.sum() * 0.0
    logp = torch.log_softmax(conf_logit[idx], 0)
    return -torch.logsumexp(logp[pos], 0)


def focal_bce(logit, y, gamma: float = 2.0, alpha: float = 0.75):
    p = torch.sigmoid(logit)
    ce = Fn.binary_cross_entropy_with_logits(logit, y, reduction="none")
    pt = p * y + (1 - p) * (1 - y)
    w = alpha * y + (1 - alpha) * (1 - y)
    return (w * (1 - pt) ** gamma * ce).mean()


def site_rank_loss(site_logit: torch.Tensor, site_center: torch.Tensor, true_centers: torch.Tensor,
                   hit: float = 4.0) -> torch.Tensor:
    """Listwise cross-entropy over the decoder's site list, with the sites that hit a true site as the targets.

    This is the ranking objective applied to the objects the metric scores: the list is the same one evaluation
    builds (confidence order, NMS at the evaluation's radius) and a site counts as correct under the same rule the
    benchmark uses. `listwise_site_loss` does the analogous thing over raw probes, where one true site is
    represented by dozens of near-duplicate positives and the decision the loss poses is therefore much easier than
    the decision top-1 poses; here each pocket appears once.
    """
    if not len(true_centers) or not len(site_logit):
        return site_logit.sum() * 0.0
    d = torch.cdist(site_center, true_centers).min(1).values
    pos = d <= hit
    if not pos.any() or pos.all():                     # nothing to order, or nothing to reject
        return site_logit.sum() * 0.0
    return -torch.logsumexp(torch.log_softmax(site_logit, 0)[pos], 0)


def site_margin_loss(site_logit: torch.Tensor, site_center: torch.Tensor, true_centers: torch.Tensor,
                     hit: float = 4.0, margin: float = 1.0) -> torch.Tensor:
    """Hinge on the hardest mistake: the best wrong site must score below the best right one by `margin`.

    The listwise term above spreads its gradient over the whole list, and a list where one wrong pocket is almost as
    good as the right one can still have a respectable loss. Top-1 does not care about the whole list: it is decided
    by a single comparison, between the best correct site and the best incorrect one. This term is that comparison
    and nothing else, which is the measured failure mode written as a loss -- the right pocket ranked second.
    """
    if not len(true_centers) or not len(site_logit):
        return site_logit.sum() * 0.0
    d = torch.cdist(site_center, true_centers).min(1).values
    pos = d <= hit
    if not pos.any() or pos.all():
        return site_logit.sum() * 0.0
    return Fn.relu(margin + site_logit[~pos].max() - site_logit[pos].max())
