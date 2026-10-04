import numpy as np
import pytest

torch = pytest.importorskip("torch")
from training.pockets import model as M  # noqa: E402


def _rot(seed=0):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return torch.tensor(q, dtype=torch.float32)


def _batch(n_res=12, n_probe=10, n_surf=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = n_res + n_probe + n_surf
    pos = torch.randn(n, 3, generator=g) * 4
    node_type = torch.tensor([0] * n_res + [1] * n_probe + [2] * n_surf)
    src, dst = [], []
    for i in range(n):
        d = (pos - pos[i]).norm(dim=1); d[i] = 1e9
        for j in d.topk(6, largest=False).indices.tolist():
            src.append(j); dst.append(i)
    ei = torch.tensor([src, dst]); et = node_type[ei[0]] * 3 + node_type[ei[1]]
    slices = dict(res=torch.arange(n_res), probe=torch.arange(n_res, n_res + n_probe), surf=torch.arange(n_res + n_probe, n))
    b = dict(pos=pos, node_type=node_type, edge_index=ei, edge_type=et, slices=slices,
             feat_res=torch.randn(n_res, 5, generator=g), feat_probe=torch.randn(n_probe, 4, generator=g), feat_surf=torch.randn(n_surf, 6, generator=g),
             vec0=torch.randn(n, 3, 3, generator=g), site_probe_mask=torch.rand(2, n_probe, generator=g))
    return b


def _apply(b, R, t):
    c = dict(b); c["pos"] = b["pos"] @ R.T + t; c["vec0"] = torch.einsum("nkc,dc->nkd", b["vec0"], R)
    return c


@pytest.mark.parametrize("use_tensors,chiral", [(True, True), (False, False), (True, False)])
def test_so3_equivariance(use_tensors, chiral):
    torch.manual_seed(0)
    net = M.EquiCaveNet(dict(res=5, probe=4, surf=6), dim=32, layers=2, heads=4, use_tensors=use_tensors, chiral=chiral).eval()
    b = _batch(); R = _rot(); t = torch.tensor([3.0, -2.0, 1.0])
    with torch.no_grad():
        o1 = net(b); o2 = net(_apply(b, R, t))
    for k in ("res_logit", "occ_logit", "conf_logit", "hot_logit", "prop_logit"):
        assert torch.allclose(o1[k], o2[k], atol=1e-4), k
    assert torch.allclose(o1["offset"] @ R.T, o2["offset"], atol=1e-4)
    assert torch.allclose(o1["center"] @ R.T + t, o2["center"], atol=1e-4)


def test_chiral_model_is_not_reflection_invariant_but_achiral_is():
    torch.manual_seed(0)
    b = _batch()
    P = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))      # a mirror
    for chiral, expect_same in ((False, True), (True, False)):
        net = M.EquiCaveNet(dict(res=5, probe=4, surf=6), dim=32, layers=2, chiral=chiral).eval()
        with torch.no_grad():
            o1 = net(b); o2 = net(_apply(b, P, torch.zeros(3)))
        same = torch.allclose(o1["res_logit"], o2["res_logit"], atol=1e-4)
        assert same == expect_same


def test_invariant_ablation_runs_and_losses_are_finite():
    torch.manual_seed(0)
    b = _batch()
    net = M.EquiCaveNet(dict(res=5, probe=4, surf=6), dim=32, layers=2, equivariant=False)
    o = net(b)
    y = (torch.rand(12) > 0.5).float()
    l1 = M.seg_loss(o["res_logit"], y)
    reg, conf = M.center_set_loss(o["center"], o["conf_logit"], b["pos"][b["slices"]["probe"]], torch.randn(2, 3))
    l3 = M.focal_bce(o["hot_logit"], (torch.rand(10, 7) > 0.8).float())
    tot = l1 + reg + conf + l3
    assert torch.isfinite(tot)
    tot.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())


def test_ablation_drops_surface_and_probe_edges():
    torch.manual_seed(0)
    b = _batch()
    full = M.EquiCaveNet(dict(res=5, probe=4, surf=6), dim=32, layers=1).eval()
    nosurf = M.EquiCaveNet(dict(res=5, probe=4, surf=6), dim=32, layers=1, use_surface=False).eval()
    nosurf.load_state_dict(full.state_dict())
    with torch.no_grad():
        assert not torch.allclose(full(b)["res_logit"], nosurf(b)["res_logit"])


def test_recycling_changes_output_and_stays_equivariant():
    torch.manual_seed(0)
    b = _batch()
    b["edge_scalar"] = torch.zeros(b["edge_index"].shape[1], 11)
    b["res_type"] = torch.randint(0, 21, (12,))
    net0 = M.EquiCaveNet(dict(res=5, probe=4, surf=6), dim=32, layers=2, recycles=0, n_edge_scalar=11).eval()
    net2 = M.EquiCaveNet(dict(res=5, probe=4, surf=6), dim=32, layers=2, recycles=2, n_edge_scalar=11).eval()
    net2.load_state_dict(net0.state_dict(), strict=False)
    R, t = _rot(3), torch.tensor([5.0, 1.0, -2.0])
    with torch.no_grad():
        o0, o2 = net0(b), net2(b)
        o2r = net2(_apply(b, R, t))
    assert len(o0["passes"]) == 1 and len(o2["passes"]) == 3
    assert not torch.allclose(o0["center"], o2["center"], atol=1e-4)          # recycling moves the prediction
    assert torch.allclose(o2["res_logit"], o2r["res_logit"], atol=1e-4)        # and stays invariant
    assert torch.allclose(o2["center"] @ R.T + t, o2r["center"], atol=1e-3)    # centres stay equivariant


def test_hungarian_matching_assigns_distinct_proposals():
    torch.manual_seed(0)
    probe = torch.randn(40, 3) * 6
    sites = torch.tensor([[0.0, 0, 0], [12.0, 0, 0], [-9.0, 4, 2]])
    conf = torch.randn(40)
    greedy, _ = M.center_set_loss(probe, conf, probe, sites, hungarian=False)
    hung, _ = M.center_set_loss(probe, conf, probe, sites, hungarian=True)
    assert torch.isfinite(greedy) and torch.isfinite(hung)
    # a single probe cannot serve two sites under matching, so the matched cost is never below the greedy one
    assert hung >= greedy - 1e-5


def test_sequence_edge_scalars_and_masking():
    import numpy as np
    from training.pockets import data as D
    ei = np.array([[0, 1, 0, 5], [1, 0, 5, 0]])
    types = np.array([0, 0, 0, 0, 0, 1])
    res_index = np.array([0, 1, 2, 3, 4, -10000]); chain = np.array([0, 0, 0, 1, 1, -1])
    es = D.edge_scalars(ei, types, res_index, chain)
    assert es.shape == (4, len(D.SEQ_BUCKETS) + 3)
    assert es[0, 0] == 1.0 and es[0, -2] == 1.0 and es[0, -1] == 1.0       # neighbours in the same chain
    assert es[2, -1] == 0.0                                                # residue to probe: not a residue pair
    b = dict(feat_res=torch.ones(10, 23), res_type=torch.randint(0, 21, (10,)))
    b2, m = D.mask_residues(b, frac=1.0)
    assert m.all() and float(b2["feat_res"].abs().sum()) == 0.0
    b3, m0 = D.mask_residues(b, frac=0.0)
    assert not m0.any() and float(b3["feat_res"].abs().sum()) > 0


def test_auxiliary_losses_are_finite():
    seq_logit = torch.randn(10, 21); types = torch.randint(0, 21, (10,))
    mask = torch.zeros(10, dtype=torch.bool); mask[:3] = True
    assert torch.isfinite(M.masked_residue_loss(seq_logit, types, mask))
    assert float(M.masked_residue_loss(seq_logit, types, torch.zeros(10, dtype=torch.bool))) == 0.0
    assert torch.isfinite(M.size_loss(torch.randn(3), torch.tensor([12.0, 30.0, 8.0])))


def test_invariant_arm_sees_the_geometry_and_is_invariant():
    """The invariant ablation must change the mechanism, not take the geometry away.

    With `invariant_mode="frames"` the initial vectors are scalarised in each node's own local frame, so the arm
    receives the same geometric information as the equivariant model at the same depth and width. Two things must
    hold: the outputs must be invariant under rotation (it is an invariant model), and they must *depend* on the
    initial vectors (it is not blind to them). The old behaviour, which simply discarded them, is still available as
    `invariant_mode="distances"` and is asserted to be blind, which is why it is not the default.
    """
    torch.manual_seed(0)
    b = _batch()
    R, t = _rot(5), torch.tensor([4.0, -1.0, 2.0])

    frames = M.EquiCaveNet(dict(res=5, probe=4, surf=6), dim=32, layers=2, equivariant=False,
                           invariant_mode="frames").eval()
    with torch.no_grad():
        o1, o2 = frames(b), frames(_apply(b, R, t))
    assert torch.allclose(o1["res_logit"], o2["res_logit"], atol=1e-4), "the invariant arm must be rotation invariant"
    assert torch.allclose(o1["hot_logit"], o2["hot_logit"], atol=1e-4)

    b2 = dict(b); b2["vec0"] = torch.zeros_like(b["vec0"])           # take the geometry away on purpose
    with torch.no_grad():
        o3 = frames(b2)
    assert not torch.allclose(o1["res_logit"], o3["res_logit"], atol=1e-4), \
        "the frames arm must actually use the initial vectors, otherwise it is blind, not invariant"

    blind = M.EquiCaveNet(dict(res=5, probe=4, surf=6), dim=32, layers=2, equivariant=False,
                          invariant_mode="distances").eval()
    with torch.no_grad():
        o4, o5 = blind(b), blind(b2)
    assert torch.allclose(o4["res_logit"], o5["res_logit"], atol=1e-6), \
        "the distances-only arm ignores the vectors by construction; that is why it is not the default"


def test_local_frame_scalars_are_rotation_invariant():
    torch.manual_seed(0)
    net = M.EquiCaveNet(dict(res=5, probe=4, surf=6), dim=16, layers=1, equivariant=False, invariant_mode="frames")
    v = torch.randn(20, 5, 3)
    R = _rot(7)
    a = net.local_frame_scalars(v)
    b = net.local_frame_scalars(torch.einsum("nkc,dc->nkd", v, R))
    assert torch.allclose(a, b, atol=1e-4), "frame coordinates must not change when the whole structure rotates"
    deg = v.clone(); deg[:, 1] = 0.0                                  # degenerate frame: documented fallback
    assert torch.isfinite(net.local_frame_scalars(deg)).all()
