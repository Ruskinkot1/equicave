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


def _symmetric_neighbourhood(kind: str, radius: float = 4.0):
    """Six neighbours of one central probe, as an octahedron or as a planar hexagon of the same radius.

    Both configurations have the same multiset of distances to the centre, the same multiset of pairwise distances
    up to the pairs' labelling, and `sum(u) == 0`, so every degree-0 invariant and every degree-1 aggregate is
    identical. They differ only in the traceless second moment: `sum(u u^T) / 6` is `I / 3` for the octahedron and
    has a zero eigenvalue along the hexagon's normal. A model whose messages stop at degree 1 therefore cannot tell
    them apart; degree-2 channels can.
    """
    if kind == "octahedron":
        u = torch.tensor([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0], [0, 0, 1.0], [0, 0, -1.0]])
    else:
        a = torch.arange(6, dtype=torch.float32) * (torch.pi / 3)
        u = torch.stack([torch.cos(a), torch.sin(a), torch.zeros(6)], 1)
    return torch.cat([torch.zeros(1, 3), u * radius])


def _one_probe_batch(pos):
    """A batch of one probe node (index 0) and its neighbours, all neighbours pointing at the centre."""
    n = len(pos)
    node_type = torch.ones(n, dtype=torch.long)                      # all probes: one input block, one edge type
    src = torch.arange(1, n); dst = torch.zeros(n - 1, dtype=torch.long)
    ei = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    return dict(pos=pos, node_type=node_type, edge_index=ei, edge_type=node_type[ei[0]] * 3 + node_type[ei[1]],
                slices=dict(res=torch.arange(0), probe=torch.arange(n), surf=torch.arange(0)),
                feat_res=torch.zeros(0, 5), feat_probe=torch.ones(n, 4), feat_surf=torch.zeros(0, 6),
                vec0=torch.zeros(n, 3, 3), site_probe_mask=torch.ones(1, n))


def test_degree2_separates_what_degree1_cannot():
    """The expressivity premise behind the degree-2 claim, checked on the cheapest configuration that isolates it.

    If this fails, no training run can rescue the `no_tensors` ablation: the arms would be comparing two models
    that receive provably identical information, and any measured difference would be noise or capacity.
    """
    octa, hexa = _symmetric_neighbourhood("octahedron"), _symmetric_neighbourhood("hexagon")
    # the premise itself: distances and the degree-1 aggregate agree, the traceless second moment does not
    for p in (octa, hexa):
        assert torch.allclose(p[1:].norm(dim=1), torch.full((6,), 4.0), atol=1e-5)
        assert torch.allclose(p[1:].sum(0), torch.zeros(3), atol=1e-5)
    second = lambda p: (p[1:, :, None] * p[1:, None, :]).mean(0) / 16.0
    assert torch.allclose(second(octa), torch.eye(3) / 3, atol=1e-5)
    assert not torch.allclose(second(hexa), torch.eye(3) / 3, atol=1e-2)

    torch.manual_seed(0)
    kw = dict(dim=32, layers=2, heads=4, chiral=False)
    with_t = M.EquiCaveNet(dict(res=5, probe=4, surf=6), use_tensors=True, **kw).eval()
    with torch.no_grad():
        a, b = with_t(_one_probe_batch(octa))["occ_logit"][0], with_t(_one_probe_batch(hexa))["occ_logit"][0]
    assert not torch.allclose(a, b, atol=1e-4), (
        f"degree-2 channels do not separate an octahedron from a hexagon ({a.item():.6f} vs {b.item():.6f}); "
        "the no_tensors ablation cannot conclude anything")

    without_t = M.EquiCaveNet(dict(res=5, probe=4, surf=6), use_tensors=False, **kw).eval()
    with torch.no_grad():
        c, d = without_t(_one_probe_batch(octa))["occ_logit"][0], without_t(_one_probe_batch(hexa))["occ_logit"][0]
    assert torch.allclose(c, d, atol=1e-5), (
        f"the degree-<=1 model separates them ({c.item():.6f} vs {d.item():.6f}), so it is reading geometry the "
        "premise says it cannot and the ablation is confounded")


def test_direction_head_is_equivariant_and_its_loss_is_invariant():
    """The direction head is a vector read-out, so it must rotate with the input; the cosine loss must not move.

    If `data.random_rotation` forgot to rotate `probe_dir`, the loss would still be finite and training would still
    run -- it would simply be supervising towards a direction in the unrotated frame, which is noise. This test is
    the reason that rotation exists, so it is asserted here rather than left to inspection.
    """
    torch.manual_seed(0)
    net = M.EquiCaveNet(dict(res=5, probe=4, surf=6), dim=32, layers=2, heads=4).eval()
    b = _batch(); R = _rot(); t = torch.tensor([3.0, -2.0, 1.0])
    n_probe = len(b["slices"]["probe"])
    g = torch.Generator().manual_seed(3)
    d = torch.randn(n_probe, 3, generator=g)
    b["probe_dir"] = d / d.norm(dim=1, keepdim=True)
    b["probe_dir_mask"] = (torch.rand(n_probe, generator=g) > 0.3).float()
    with torch.no_grad():
        o1 = net(b)
        br = _apply(b, R, t); br["probe_dir"] = b["probe_dir"] @ R.T        # as random_rotation does
        o2 = net(br)
    assert torch.allclose(o1["dir"] @ R.T, o2["dir"], atol=1e-4)            # the head rotates with the structure
    l1 = M.direction_loss(o1["dir"], b["probe_dir"], b["probe_dir_mask"])
    l2 = M.direction_loss(o2["dir"], br["probe_dir"], br["probe_dir_mask"])
    assert torch.allclose(l1, l2, atol=1e-5)                                # so the loss is invariant
    # and a target left unrotated gives a different loss: the failure this test exists to catch
    l_bug = M.direction_loss(o2["dir"], b["probe_dir"], b["probe_dir_mask"])
    assert not torch.allclose(l1, l_bug, atol=1e-3)


def test_direction_loss_edge_cases():
    pred = torch.tensor([[1.0, 0, 0], [0.0, 1.0, 0]])
    target = torch.tensor([[1.0, 0, 0], [1.0, 0.0, 0]])
    assert M.direction_loss(pred, target, torch.tensor([1.0, 0.0])) == pytest.approx(0.0, abs=1e-6)
    assert M.direction_loss(pred, target, torch.tensor([0.0, 1.0])) == pytest.approx(1.0, abs=1e-6)
    assert M.direction_loss(pred, target, torch.zeros(2)) == pytest.approx(0.0)     # nothing usable: no contribution
    zero = M.direction_loss(torch.zeros(1, 3), torch.tensor([[1.0, 0, 0]]), torch.ones(1))
    assert torch.isfinite(zero)                                                     # a collapsed vector, not a NaN


def test_listwise_loss_prefers_ordering_the_true_site_first():
    centers = torch.tensor([[0.0, 0, 0], [20.0, 0, 0], [40.0, 0, 0]])
    sites = torch.tensor([[0.0, 0, 0]])
    probe_pos = centers.clone()
    good = M.listwise_site_loss(torch.tensor([5.0, 0.0, 0.0]), centers, sites, probe_pos)
    bad = M.listwise_site_loss(torch.tensor([0.0, 0.0, 5.0]), centers, sites, probe_pos)
    assert good < bad                                              # confident on the hit beats confident on a miss
    assert torch.isfinite(good) and good > 0
    # degenerate lists contribute nothing rather than exploding
    assert M.listwise_site_loss(torch.tensor([1.0]), centers[:1], torch.zeros(0, 3), probe_pos[:1]) == pytest.approx(0.0)
    all_hit = M.listwise_site_loss(torch.tensor([1.0, 2.0]), sites.repeat(2, 1), sites, sites.repeat(2, 1))
    assert all_hit == pytest.approx(0.0)                           # nothing to reject


def test_a_full_training_step_runs_under_bf16_autocast():
    """Mixed precision is the default on a GPU (`optim.amp: bf16`), and it was broken on the first step of a real run.

    Autocast leaves some operations in float32 -- normalisation, softmax -- and runs the linear layers in bf16, so a
    buffer allocated in one dtype and written from the other raises outright. It surfaced three layers deep: the
    node embedding into the residual stream, then the attention message accumulation, then the normalised channels
    handed to the next layer. A CPU autocast context reproduces all of it, so this test guards a GPU-only default
    without needing a GPU.
    """
    torch.manual_seed(0)
    net = M.EquiCaveNet(dict(res=5, probe=4, surf=6), dim=32, layers=3, heads=4, recycles=1)
    b = _batch()
    n_probe = len(b["slices"]["probe"])
    g = torch.Generator().manual_seed(1)
    b["y_res"] = (torch.rand(len(b["slices"]["res"]), generator=g) > 0.7).float()
    b["y_occ"] = (torch.rand(n_probe, generator=g) > 0.8).float()
    b["y_hot"] = (torch.rand(n_probe, 7, generator=g) > 0.9).float()
    b["site_centers"] = torch.randn(2, 3, generator=g)
    b["y_prop"] = (torch.rand(2, 14, generator=g) > 0.5).float()
    d = torch.randn(n_probe, 3, generator=g)
    b["probe_dir"] = d / d.norm(dim=1, keepdim=True)
    b["probe_dir_mask"] = (torch.rand(n_probe, generator=g) > 0.3).float()
    w = dict(w_res=1.0, w_occ=1.0, w_center=0.5, w_conf=0.5, w_hot=1.0, w_prop=0.5, w_dir=0.5, w_listwise=0.3,
             occ_pos_weight=3.0, res_pos_weight=3.0, hungarian=True, recycle_weight=0.5)

    from training.pockets import net_task as NT
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = net(b)
        total, parts = NT.losses(out, b, w)
    total.backward()
    opt.step()
    assert torch.isfinite(total), "the loss is not finite under bf16"
    assert {"dir", "listwise"} <= set(parts), "the new terms did not contribute under autocast"
    assert all(np.isfinite(v) for v in parts.values())
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_bf16_and_fp32_agree_to_low_precision():
    """The dtype casts must not change what the model computes, only how precisely."""
    torch.manual_seed(0)
    net = M.EquiCaveNet(dict(res=5, probe=4, surf=6), dim=32, layers=2, heads=4).eval()
    b = _batch()
    with torch.no_grad():
        fp32 = net(b)["occ_logit"]
        with torch.autocast("cpu", dtype=torch.bfloat16):
            bf16 = net(b)["occ_logit"]
    assert torch.allclose(fp32, bf16.float(), atol=0.2), (fp32[:4], bf16[:4])


# --- the site decoder: sites as entities that compete, instead of probes scored one at a time ---

def test_site_scores_are_rotation_invariant_and_centres_equivariant():
    """The decoder may only see invariants, or a pocket's rank would depend on how the crystal was oriented."""
    torch.manual_seed(0)
    net = M.EquiCaveNet(dict(res=5, probe=4, surf=6), dim=32, layers=2, heads=4, site_decoder=True).eval()
    b = _batch(); R = _rot(1); t = torch.tensor([-4.0, 1.5, 2.0])
    with torch.no_grad():
        o1, o2 = net(b), net(_apply(b, R, t))
    assert len(o1["site_logit"]) > 1, "the decoder produced no list to rank"
    assert torch.allclose(o1["site_logit"], o2["site_logit"], atol=1e-4)
    assert torch.allclose(o1["site_center"] @ R.T + t, o2["site_center"], atol=1e-4)


def test_a_site_token_is_not_merely_its_seed_probe():
    """Each site pools the probes that support it, so its score must move when a supporting probe's input changes."""
    torch.manual_seed(0)
    net = M.EquiCaveNet(dict(res=5, probe=4, surf=6), dim=32, layers=2, heads=4, site_decoder=True).eval()
    b = _batch()
    with torch.no_grad():
        base = net(b)
        w = base["site_member"]
        # a probe that is a member of site 0 but is not its seed
        member = [j for j in range(w.shape[1]) if w[0, j] > 1e-2 and j != int(base["site_seed"][0])]
        assert member, "site 0 gathered no probe besides its seed"
        c = dict(b); c["feat_probe"] = b["feat_probe"].clone()
        c["feat_probe"][member[0]] += 3.0
        moved = net(c)
    assert not torch.allclose(base["site_logit"][0], moved["site_logit"][0], atol=1e-5)


def test_the_site_list_is_non_redundant_at_the_evaluation_radius():
    """The tokens are formed by the rule the benchmark scores with, so no two of them may sit within it."""
    torch.manual_seed(0)
    net = M.EquiCaveNet(dict(res=5, probe=4, surf=6), dim=32, layers=2, heads=4, site_decoder=True,
                        site_nms=6.0).eval()
    with torch.no_grad():
        o = net(_batch(n_probe=24))
    seeds = o["site_seed"]
    centres = o["center"][seeds]
    d = torch.cdist(centres, centres) + torch.eye(len(seeds)) * 99
    assert float(d.min()) > 6.0


def test_the_margin_loss_is_the_comparison_top1_decides():
    """Zero when the best correct site already leads by the margin, positive when a wrong one is on top."""
    centres = torch.tensor([[0.0, 0, 0], [20.0, 0, 0]])
    true = torch.tensor([[0.0, 0, 0]])
    good = M.site_margin_loss(torch.tensor([5.0, 0.0]), centres, true, margin=1.0)
    bad = M.site_margin_loss(torch.tensor([0.0, 5.0]), centres, true, margin=1.0)
    assert float(good) == 0.0
    assert float(bad) == pytest.approx(6.0)


def test_the_site_losses_are_silent_when_there_is_nothing_to_order():
    """No true site, every site correct, or no site at all: a structure must not be charged for an absent decision."""
    centres = torch.tensor([[0.0, 0, 0], [20.0, 0, 0]])
    logit = torch.tensor([1.0, 2.0])
    empty = torch.zeros((0, 3))
    for f in (M.site_rank_loss, M.site_margin_loss):
        assert float(f(logit, centres, empty)) == 0.0
        assert float(f(logit, centres, centres)) == 0.0                      # every site hits: nothing to reject
        assert float(f(torch.zeros(0), empty, centres)) == 0.0


def test_site_rank_loss_falls_when_the_right_site_rises():
    centres = torch.tensor([[0.0, 0, 0], [20.0, 0, 0], [40.0, 0, 0]])
    true = torch.tensor([[1.0, 0, 0]])
    worse = M.site_rank_loss(torch.tensor([0.0, 3.0, 3.0]), centres, true)
    better = M.site_rank_loss(torch.tensor([3.0, 0.0, 0.0]), centres, true)
    assert float(better) < float(worse)


def test_probe_flow_moves_probes_along_the_predicted_direction():
    """In flow mode a recycling step is a bounded walk along the direction head, not a jump to the centre."""
    torch.manual_seed(0)
    net = M.EquiCaveNet(dict(res=5, probe=4, surf=6), dim=32, layers=2, heads=4, recycles=1,
                        probe_update="flow", flow_step=2.0).eval()
    b = _batch()
    with torch.no_grad():
        o = net(b)
    first, second = o["passes"][0], o["passes"][1]
    step = (second["probe_pos"] - first["probe_pos"]).norm(dim=-1)
    assert torch.allclose(step, torch.full_like(step, 2.0), atol=1e-4)
    cos = torch.nn.functional.cosine_similarity(second["probe_pos"] - first["probe_pos"], first["dir"], dim=-1)
    assert float(cos.min()) > 0.99


def test_the_decoder_survives_a_bf16_step():
    """Mixed precision has broken this model twice; the decoder's attention is new surface for it."""
    torch.manual_seed(0)
    net = M.EquiCaveNet(dict(res=5, probe=4, surf=6), dim=32, layers=2, heads=4, site_decoder=True)
    b = _batch()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = net(b)
        loss = (M.site_rank_loss(out["site_logit"], out["site_center"], torch.tensor([[0.0, 0.0, 0.0]]))
                + M.site_margin_loss(out["site_logit"], out["site_center"], torch.tensor([[0.0, 0.0, 0.0]])))
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in net.site_decoder.parameters())


def test_without_the_decoder_the_model_is_unchanged():
    torch.manual_seed(0)
    net = M.EquiCaveNet(dict(res=5, probe=4, surf=6), dim=32, layers=2, heads=4, site_decoder=False).eval()
    with torch.no_grad():
        o = net(_batch())
    assert "site_logit" not in o and net.site_decoder is None


# --- mixed precision: the CPU autocast policy is not the CUDA one, so test the CUDA one explicitly ---

def test_seg_softmax_survives_a_float32_exp_in_a_bf16_stream():
    """CUDA autocast forces `exp` to float32 while the stream stays bf16; CPU autocast does not.

    A CPU bf16 test therefore passes while the CUDA run dies on its first batch -- which is what happened. The
    reduction must not care which side comes back in which dtype.
    """
    logits = torch.randn(40, 4, dtype=torch.bfloat16)
    index = torch.randint(0, 7, (40,))
    out = M.seg_softmax(logits, index, 7)
    assert out.dtype == torch.bfloat16
    ref = M.seg_softmax(logits.float(), index, 7)
    assert torch.allclose(out.float(), ref, atol=2e-2)
    for j in range(7):                                      # each group must still sum to one
        m = index == j
        if m.any():
            assert torch.allclose(ref[m].sum(0), torch.ones(4), atol=1e-4)


def test_a_full_step_survives_cudas_autocast_policy_simulated_on_cpu(monkeypatch):
    """Force every `exp` to float32, the way CUDA autocast does, and take a full training step under CPU bf16.

    This is the regression test the previous bf16 fixes needed and did not have: they were written against CPU
    autocast, whose promotion list is different, so the one operation that mattered was never exercised.
    """
    real_exp = torch.exp
    monkeypatch.setattr(torch, "exp", lambda t, *a, **k: real_exp(t.float(), *a, **k))
    torch.manual_seed(0)
    net = M.EquiCaveNet(dict(res=5, probe=4, surf=6), dim=32, layers=2, heads=4, site_decoder=True)
    b = _batch()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = net(b)
        loss = (out["occ_logit"].float().mean() + out["conf_logit"].float().mean()
                + M.site_rank_loss(out["site_logit"], out["site_center"], torch.tensor([[0.0, 0.0, 0.0]])))
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def _to_bf16_stream(b):
    """Node features in bf16, geometry left in float32: the dtype mixture a GPU run actually sees.

    This does not use autocast on purpose. Autocast's promotion list differs between its CPU and CUDA
    implementations, so a CPU autocast test can pass while the CUDA run dies -- which happened twice. Setting the
    dtypes by hand reproduces the mixture deterministically on any machine: the parameters and the residual stream
    are bf16, while `pos` and `vec0` arrive from the feature cache as float32 and are never cast by the loader.
    """
    c = dict(b)
    for k in list(c):
        if k.startswith("feat_") or k == "site_probe_mask":
            c[k] = c[k].to(torch.bfloat16)
    return c


@pytest.mark.parametrize("use_tensors,chiral,recycles", [(True, True, 0), (True, True, 2), (False, False, 0)])
def test_a_bf16_stream_with_float32_geometry_runs(use_tensors, chiral, recycles):
    """Every accumulation must take a float32 intermediate without raising, whatever promotes where."""
    torch.manual_seed(0)
    net = M.EquiCaveNet(dict(res=5, probe=4, surf=6), dim=32, layers=2, heads=4, use_tensors=use_tensors,
                        chiral=chiral, recycles=recycles, site_decoder=True).to(torch.bfloat16).eval()
    with torch.no_grad():
        out = net(_to_bf16_stream(_batch()))
    assert out["occ_logit"].dtype == torch.bfloat16
    assert torch.isfinite(out["occ_logit"].float()).all()
    assert torch.isfinite(out["site_logit"].float()).all()


def test_a_bf16_stream_takes_a_training_step():
    torch.manual_seed(0)
    net = M.EquiCaveNet(dict(res=5, probe=4, surf=6), dim=32, layers=2, heads=4,
                        site_decoder=True).to(torch.bfloat16)
    out = net(_to_bf16_stream(_batch()))
    loss = (out["occ_logit"].float().mean() + out["conf_logit"].float().mean()
            + M.site_rank_loss(out["site_logit"].float(), out["site_center"].float(),
                               torch.tensor([[0.0, 0.0, 0.0]])))
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad.float()).all() for p in net.parameters())


# --- Gaussian dynamic attention ---------------------------------------------------------------------------------
# A kernel on variance-normalised feature differences instead of the edge MLP's learned logits, after Gaussian
# Dynamic Attention (Wang et al. 2026). It reads only the invariant stream, so the claim it has to survive is that
# equivariance is untouched; and it must actually differ from the dot-product form, or the arm measures nothing.


@pytest.mark.parametrize("use_tensors,chiral", [(True, True), (False, False)])
def test_gaussian_attention_is_still_so3_equivariant(use_tensors, chiral):
    torch.manual_seed(0)
    net = M.EquiCaveNet(dict(res=5, probe=4, surf=6), dim=32, layers=2, heads=4, use_tensors=use_tensors,
                        chiral=chiral, attn_kernel="gaussian").eval()
    b = _batch(); R = _rot(); t = torch.tensor([3.0, -2.0, 1.0])
    with torch.no_grad():
        o1 = net(b); o2 = net(_apply(b, R, t))
    for k in ("res_logit", "occ_logit", "conf_logit", "hot_logit", "prop_logit"):
        assert torch.allclose(o1[k], o2[k], atol=1e-4), k
    assert torch.allclose(o1["offset"] @ R.T, o2["offset"], atol=1e-4)
    assert torch.allclose(o1["center"] @ R.T + t, o2["center"], atol=1e-4)


def test_gaussian_attention_changes_the_model():
    torch.manual_seed(0)
    kw = dict(dim=32, layers=2, heads=4)
    torch.manual_seed(0); a = M.EquiCaveNet(dict(res=5, probe=4, surf=6), attn_kernel="dot", **kw).eval()
    torch.manual_seed(0); g = M.EquiCaveNet(dict(res=5, probe=4, surf=6), attn_kernel="gaussian", **kw).eval()
    b = _batch()
    with torch.no_grad():
        assert not torch.allclose(a(b)["occ_logit"], g(b)["occ_logit"], atol=1e-5)


def test_gaussian_attention_costs_one_parameter_per_head_per_layer():
    kw = dict(dim=32, layers=2, heads=4)
    a = sum(p.numel() for p in M.EquiCaveNet(dict(res=5, probe=4, surf=6), attn_kernel="dot", **kw).parameters())
    g = sum(p.numel() for p in M.EquiCaveNet(dict(res=5, probe=4, surf=6), attn_kernel="gaussian", **kw).parameters())
    assert g - a == 2 * 4, (a, g)      # layers x heads, against the O(dim^2) a key-query projection would cost


def test_the_kernel_width_adapts_to_the_neighbourhood():
    # The point of the thing: a destination whose neighbours differ from it in a varied way must not get the same
    # attention profile as one whose neighbours are uniformly similar. If the normalisation were global rather
    # than per destination node, these two would be driven to the same distribution.
    torch.manual_seed(0)
    layer = M.GeoTensorAttention(dim=8, heads=2, attn_kernel="gaussian").eval()
    x = torch.zeros(7, 8)
    x[1:4] = torch.tensor([0.1, 0.2, 0.3])[:, None]          # node 0: a tight neighbourhood
    x[4:7] = torch.tensor([1.0, 5.0, 20.0])[:, None]         # node 6: a spread-out one
    src = torch.tensor([1, 2, 3, 4, 5, 6])
    dst = torch.tensor([0, 0, 0, 6, 6, 6])
    with torch.no_grad():
        lg = layer.attn_logits(torch.zeros(6, 2), x, src, dst, 7)
        a = M.seg_softmax(lg, dst, 7)
    tight, spread = a[:3, 0], a[3:, 0]
    assert torch.isfinite(a).all()
    assert abs(float(tight.std() - spread.std())) > 1e-6, (tight, spread)


def test_seg_mean_is_a_mean_per_destination():
    v = torch.tensor([[1.0], [3.0], [10.0]])
    idx = torch.tensor([0, 0, 1])
    got = M.seg_mean(v, idx, 3)
    assert torch.allclose(got, torch.tensor([[2.0], [10.0], [0.0]]))


# --- the site aggregator --------------------------------------------------------------------------------------
# Three accurate methods use three different rules to collapse a site's members into one score -- P2Rank sums the
# squares, GrASP and VN-EGNN average, YuelPocket takes the peak -- and nobody has compared them. In every one of
# them it is a hyperparameter, never an ablation, so this is a missing comparison rather than a tuning knob.


@pytest.mark.parametrize("agg", M.SiteDecoder.AGGREGATORS)
def test_every_aggregator_keeps_the_model_equivariant(agg):
    torch.manual_seed(0)
    net = M.EquiCaveNet(dict(res=5, probe=4, surf=6), dim=32, layers=2, heads=4, site_decoder=True,
                        site_agg=agg).eval()
    b = _batch(); R = _rot(); t = torch.tensor([3.0, -2.0, 1.0])
    with torch.no_grad():
        o1 = net(b); o2 = net(_apply(b, R, t))
    assert torch.allclose(o1["occ_logit"], o2["occ_logit"], atol=1e-4), agg
    if "site_center" in o1:
        assert torch.allclose(o1["site_center"] @ R.T + t, o2["site_center"], atol=1e-3), agg


def test_the_aggregators_are_parameter_matched():
    n = {a: sum(p.numel() for p in M.EquiCaveNet(dict(res=5, probe=4, surf=6), dim=32, layers=2, heads=4,
                                                 site_decoder=True, site_agg=a).parameters())
         for a in M.SiteDecoder.AGGREGATORS}
    assert len(set(n.values())) == 1, n       # otherwise the arm measures capacity as well as the rule


def test_the_aggregators_actually_differ():
    torch.manual_seed(0)
    dec = M.SiteDecoder(dim=16, heads=2, agg="sum_sq")
    w = torch.tensor([[1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    occ = torch.tensor([0.9, 0.1, 0.1, 0.9])
    got = {}
    for a in M.SiteDecoder.AGGREGATORS:
        dec.agg = a
        got[a] = dec.aggregate(w, occ).squeeze(-1).tolist()
    assert len({tuple(round(x, 6) for x in v) for v in got.values()}) == len(got), got
    # the rule that matters: one confident probe plus two weak ones should beat three lukewarm ones under
    # sum-of-squares, which is the reason P2Rank uses it, while the mean cannot tell them apart.
    dec.agg = "mean"
    flat = dec.aggregate(torch.ones(2, 3), torch.tensor([0.4, 0.4, 0.4])).squeeze(-1)
    peak = dec.aggregate(torch.ones(2, 3), torch.tensor([0.9, 0.15, 0.15])).squeeze(-1)
    assert abs(float(flat[0]) - float(peak[0])) < 0.02, "the mean is blind to the shape of the distribution"
    dec.agg = "sum_sq"
    flat2 = dec.aggregate(torch.ones(2, 3), torch.tensor([0.4, 0.4, 0.4])).squeeze(-1)
    peak2 = dec.aggregate(torch.ones(2, 3), torch.tensor([0.9, 0.15, 0.15])).squeeze(-1)
    assert float(peak2[0]) > float(flat2[0]), "sum of squares must prefer the peaked site"


def test_an_unknown_aggregator_is_refused():
    with pytest.raises(AssertionError):
        M.SiteDecoder(dim=16, heads=2, agg="median")
