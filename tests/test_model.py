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
