"""A hybrid ranker served without its network, and a plain ranker labelled as a hybrid.

Both produce valid-looking numbers: the first reads every net_* column as 0.0, the second publishes the plain
ranker's top-1 under the hybrid's name. Neither raises anywhere else in the pipeline.
"""
import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _ev():
    spec = importlib.util.spec_from_file_location("evaluate_mod", ROOT / "scripts/eval/evaluate.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _ranker(tmp_path, feats):
    p = tmp_path / "ranker.txt"
    p.write_text("")
    (tmp_path / "ranker.txt.features.json").write_text(json.dumps(feats))
    return str(p)


def test_hybrid_ranker_without_net_is_refused(tmp_path):
    ev = _ev()
    path = _ranker(tmp_path, ["nat_rank", "net_site_score", "net_occ_mean_4"])
    with pytest.raises(SystemExit) as e:
        ev.check_net_args(path, "")
    assert "2 net_* columns" in str(e.value)


def test_hybrid_ranker_with_net_and_plain_ranker_pass(tmp_path):
    ev = _ev()
    ev.check_net_args(_ranker(tmp_path, ["nat_rank", "net_site_score"]), "fold0/model.pt")
    ev.check_net_args(_ranker(tmp_path, ["nat_rank", "geo_depth"]), "")
    ev.check_net_args("", "")


def test_label_follows_what_the_ranker_reads():
    ev = _ev()
    assert ev.reads_net(["nat_rank", "net_hot3_max_8"])
    assert not ev.reads_net(["nat_rank", "geo_depth", "esm_pc0"])
    src = (ROOT / "scripts/eval/evaluate.py").read_text()
    assert 'if extra is not None and reads_net(feats):' in src


def test_network_gets_its_own_point_model():
    """The network's probe placement must not depend on whether the *ranker* reads point aggregates."""
    ev = _ev()
    lig = {"data": {"probe_sampling": "ligandable", "point_model_tag": "native"}}
    assert ev.net_point_model_path(lig).endswith("models/point_native.txt")
    assert ev.net_point_model_path(lig, "custom.txt") == "custom.txt"
    assert ev.net_point_model_path({"data": {"probe_sampling": "tiered"}}) == ""
    assert ev.net_point_model_path(None) == ""
    src = (ROOT / "scripts/eval/evaluate.py").read_text()
    assert "point_model=net_point_model" in src and "net_point_model=net_pm" in src


def test_report_survives_metadata_entries_and_writes_numbers_first():
    src = (ROOT / "scripts/eval/evaluate.py").read_text()
    assert 'if m.startswith("_"):' in src                      # `_composition` is not a method row
    main = src[src.index("def main():"):]
    assert main.index('f"eval_{a.set}{a.tag}.json"') < main.index('f"eval_{a.set}{a.tag}.md"')
    assert main.index("eval_candidates_") < main.index('f"eval_{a.set}{a.tag}.md"')
