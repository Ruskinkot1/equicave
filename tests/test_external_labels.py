"""The two ways a baseline run on a new benchmark fails silently.

Both were found trying to run P2Rank on HOLO4K, LIGYSIS and CryptoBench, where it had never been run before, and
both produce no error at all -- one an empty table, the other a table that never appears.

* P2Rank resolves a dataset's protein paths relative to the dataset file. A relative path written into `all.ds`
  becomes `<work>/<relative path>`, every structure is reported missing, and the run ends in three seconds.
* `eval_set_to_manifest.py` writes the ligand code "ANY" for a benchmark that names no specific ligands. Matching
  it as a literal code selects nothing, so every structure of such a set is dropped for having no ligand and the
  driver writes no table -- which looks like "the tool produced nothing" rather than a label bug.
"""
import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/baselines"))
import run_external as rx  # noqa: E402

PDB = REPO / "data/external/pdb/1A52.pdb"       # a LIGYSIS entry; its only ligand group is EST
ROW = [dict(center=np.array([0.0, 0.0, 0.0]), tool_score=1.0, tool_rank=1, tool_extra=0.5)]
META = dict(cluster30="x", fold=0)

pytestmark = pytest.mark.skipif(not PDB.exists(), reason="needs the fetched PDB for 1A52")


def test_the_any_sentinel_is_not_a_ligand_code():
    assert rx.label(ROW, "1A52", PDB, {"ANY"}, META), "a set that names no codes must still be labelled"


def test_an_unnamed_set_matches_what_evaluate_matches():
    # Same rule as evaluate.py and the deep-learning drivers: no codes named means every group but EXCLUDE.
    assert rx.label(ROW, "1A52", PDB, {"ANY"}, META) == rx.label(ROW, "1A52", PDB, set(), META)


def test_a_named_code_still_restricts():
    assert rx.label(ROW, "1A52", PDB, {"EST"}, META)
    assert rx.label(ROW, "1A52", PDB, {"ZZZ"}, META) == []


def test_the_p2rank_dataset_lists_absolute_paths(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen["ds"] = pathlib.Path(cmd[2]).read_text()
        class R: returncode = 0; stdout = ""; stderr = ""
        return R()

    monkeypatch.setattr(rx.subprocess, "run", fake_run)
    rx.run_p2rank_batch("prank", [pathlib.Path("prot/1A52.pdb")], tmp_path, 1)
    paths = [l for l in seen["ds"].splitlines() if l.strip() and not l.startswith("HEADER")]
    assert paths and all(pathlib.Path(p).is_absolute() for p in paths), paths
