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
