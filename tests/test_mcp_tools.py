import numpy as np

from equicave import mcp_server, structure
from tests.conftest import ball_with_pocket


def test_mcp_tool_functions_without_models(tmp_path):
    xyz = ball_with_pocket()
    p = structure.write_pdb(tmp_path / "x.pdb", xyz, element="C", resname="ALA", record="ATOM")
    out = mcp_server.detect_pockets(str(p), top_k=3, ranker=None)
    assert out["sites"] and out["sites"][0]["rank"] == 1 and len(out["sites"][0]["box"]) == 3
    c = out["sites"][0]["center"]
    assert len(mcp_server.cavity_mask(str(p), c)["points"]) > 0
    props = mcp_server.pocket_properties(str(p), c, model=None)
    assert "classes" in props and "geometry-only" in props["note"]
    hf = mcp_server.hotspot_field(str(p), c, model=None)
    assert len(hf["points"]) == len(hf["probabilities"])


def test_modes_and_predict_without_models(tmp_path):
    from equicave import modes, predict
    from tests.conftest import ball_with_pocket
    xyz = ball_with_pocket()
    p = structure.write_pdb(tmp_path / "x.pdb", xyz, element="C", resname="ALA", record="ATOM")
    assert set(modes.MODES) == {"fast", "accurate", "peptide"}
    for name in modes.MODES:
        cfg = modes.mode(name)
        assert cfg["detect"]["max_sites"] > 0 and cfg["network"]["dim"] > 0 and "sites" in cfg["heads"]
    out = predict.predict(p, mode="fast", top_k=3)
    assert out["sites"] and out["sites"][0]["rank"] == 1 and out["mode"] == "fast"
    assert "network not used" in out["note"]                 # no trained model: said explicitly, not hidden
    assert all(s["score"] >= t["score"] for s, t in zip(out["sites"], out["sites"][1:]))
    pep = predict.predict(p, mode="peptide", top_k=3)
    assert pep["n_candidates"] >= out["n_candidates"] - 1     # the groove tier adds candidates (or dedups to the same)


def test_a_stale_model_fails_loudly_instead_of_reading_zeros():
    """A model trained before a feature definition changed must not be served with zeros in place of its columns."""
    import pytest
    from equicave import pocket_features as pf
    produced = {"nat_score": 1.0, "cav_volume": 2.0}
    pf.check_features(["nat_score", "cav_volume"], produced)                      # all present: fine
    pf.check_features(["nat_score", "net_seg", "esm_mean0", "cav_volume_z"], produced)  # optional columns: fine
    with pytest.raises(RuntimeError, match="features this checkout does not produce"):
        pf.check_features(["nat_score", "pot_n_hbd", "pot_n_hba"], produced)
