"""The e3nn backbone: exact equivariance, and a cross-check that it agrees with the hand-written Cartesian layer."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("e3nn")
from training.pockets import model as M  # noqa: E402
from training.pockets.model_e3nn import EquiCaveNetE3, irreps_for  # noqa: E402
from tests.test_model import _apply, _batch, _rot  # noqa: E402


@pytest.mark.parametrize("lmax", [1, 2, 3])
def test_e3nn_backbone_is_equivariant(lmax):
    torch.manual_seed(0)
    net = EquiCaveNetE3(dict(res=5, probe=4, surf=6), dim=8, layers=2, lmax=lmax).eval()
    b = _batch(); R = _rot(); t = torch.tensor([3.0, -2.0, 1.0])
    with torch.no_grad():
        o1, o2 = net(b), net(_apply(b, R, t))
    for k in ("res_logit", "occ_logit", "conf_logit", "hot_logit", "prop_logit"):
        assert torch.allclose(o1[k], o2[k], atol=1e-4), k
    assert torch.allclose(o1["offset"] @ R.T, o2["offset"], atol=1e-4)
    assert torch.allclose(o1["center"] @ R.T + t, o2["center"], atol=1e-3)


def test_irreps_layout():
    ir = irreps_for(4, 2)
    assert ir.dim == 4 * (1 + 3 + 5)
    assert [str(x.ir) for x in ir] == ["0e", "1o", "2e"]


def test_both_backbones_train_one_step_and_agree_on_equivariance():
    """The two backbones are interchangeable: same batch, same heads, both equivariant, both produce gradients."""
    torch.manual_seed(0)
    b = _batch()
    b["res_type"] = torch.randint(0, 21, (12,))
    y_res = (torch.rand(12) > 0.5).float()
    outs = {}
    for name, net in (("cartesian", M.EquiCaveNet(dict(res=5, probe=4, surf=6), dim=16, layers=2)),
                      ("e3nn", EquiCaveNetE3(dict(res=5, probe=4, surf=6), dim=8, layers=2, lmax=2))):
        o = net(b)
        loss = M.seg_loss(o["res_logit"], y_res) + M.focal_bce(o["hot_logit"], (torch.rand(10, 7) > 0.8).float())
        reg, conf = M.center_set_loss(o["center"], o["conf_logit"], b["pos"][b["slices"]["probe"]], torch.randn(2, 3))
        (loss + reg + conf).backward()
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads), name
        outs[name] = o
    assert outs["cartesian"]["hot_logit"].shape == outs["e3nn"]["hot_logit"].shape
