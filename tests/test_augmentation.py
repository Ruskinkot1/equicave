"""Noise augmentation, and the graph surgery that node dropout needs.

Rotation augmentation is a no-op for an equivariant model, so until now the training loop augmented nothing at
all. The three published augmentations for this task -- coordinate jitter, feature noise, node dropout -- are
the ones Shiota et al. (PLOS ONE 2024) measured at +0.0976 combined PR-AUC, against +0.0256 for 2.6x the
training structures. Dropping a node is the one that can corrupt a batch rather than perturb it: the edges that
touched it must go, and every surviving edge must be renumbered, or the model reads the wrong node's features.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from training.pockets import data as D  # noqa: E402


def batch(n_res=6, n_probe=5, n_surf=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = n_res + n_probe + n_surf
    pos = torch.randn(n, 3, generator=g) * 4
    node_type = torch.tensor([0] * n_res + [1] * n_probe + [2] * n_surf)
    src, dst = [], []
    for i in range(n):
        d = (pos - pos[i]).norm(dim=1); d[i] = 1e9
        for j in d.topk(3, largest=False).indices.tolist():
            src.append(j); dst.append(i)
    ei = torch.tensor([src, dst])
    return dict(pos=pos, node_type=node_type, edge_index=ei, edge_type=node_type[ei[0]] * 3 + node_type[ei[1]],
                slices=dict(res=torch.arange(n_res), probe=torch.arange(n_res, n_res + n_probe),
                            surf=torch.arange(n_res + n_probe, n)),
                feat_res=torch.randn(n_res, 5, generator=g), feat_probe=torch.randn(n_probe, 4, generator=g),
                feat_surf=torch.randn(n_surf, 6, generator=g), vec0=torch.randn(n, 3, 3, generator=g))


def test_no_sigma_is_no_change():
    b = batch()
    assert D.jitter(b, np.random.default_rng(0)) is b


def test_coordinate_jitter_moves_positions_by_about_sigma():
    b = batch()
    out = D.jitter(b, np.random.default_rng(0), pos=0.5)
    d = (out["pos"] - b["pos"]).norm(dim=1)
    assert d.min() > 0
    assert 0.2 < float(d.mean()) < 2.0, float(d.mean())      # ~sigma*sqrt(3) for three independent components


def test_feature_noise_touches_features_and_not_geometry():
    b = batch()
    out = D.jitter(b, np.random.default_rng(0), feat=0.3)
    assert not torch.allclose(out["feat_res"], b["feat_res"])
    assert torch.equal(out["pos"], b["pos"])


def test_dropout_removes_nodes_and_their_edges():
    b = batch()
    out = D.jitter(b, np.random.default_rng(1), drop=0.4)
    n = out["pos"].shape[0]
    assert n < b["pos"].shape[0]
    # every surviving edge must point at a node that still exists, under the new numbering
    assert int(out["edge_index"].max()) < n
    assert int(out["edge_index"].min()) >= 0
    assert out["edge_type"].shape[0] == out["edge_index"].shape[1]


def test_dropout_keeps_each_node_with_its_own_features():
    # The failure this guards: renumbering the edges but not the per-node tensors, so the model reads a
    # neighbour's features for every node after the first dropped one. Silent, and it would look like noise.
    b = batch()
    keep = torch.tensor([i % 3 != 0 for i in range(b["pos"].shape[0])])
    out = D.drop_nodes(b, keep)
    assert torch.equal(out["pos"], b["pos"][keep])
    assert torch.equal(out["node_type"], b["node_type"][keep])
    assert torch.equal(out["vec0"], b["vec0"][keep])


def test_dropout_rewrites_the_slices():
    b = batch()
    keep = torch.tensor([i % 2 == 0 for i in range(b["pos"].shape[0])])
    out = D.drop_nodes(b, keep)
    n = out["pos"].shape[0]
    for name, idx in out["slices"].items():
        assert idx.numel() == 0 or (int(idx.max()) < n and int(idx.min()) >= 0), name
    # and the slices still name nodes of the right type
    for name, t in (("res", 0), ("probe", 1), ("surf", 2)):
        idx = out["slices"][name]
        if idx.numel():
            assert set(out["node_type"][idx].tolist()) == {t}, name


def test_dropout_never_empties_the_graph():
    b = batch()
    out = D.jitter(b, np.random.default_rng(0), drop=1.0)
    assert out["pos"].shape[0] >= 1
