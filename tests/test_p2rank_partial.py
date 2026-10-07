"""A baseline run that mostly succeeded must not be thrown away.

P2Rank exits non-zero when any one dataset item fails, having processed every other structure normally. The
driver treated that as a failed run, so one LIGYSIS structure -- 4CBO, which carries a residue BioJava cannot
type -- discarded 3323 completed predictions and three hours of compute. It also reported the last 2000
characters of stdout, which for P2Rank is per-pocket scores from whichever structure finished last, so the
message said nothing about the cause.
"""
import pathlib
import sys
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts/baselines"))
import run_external as rx  # noqa: E402

NOISE = "\n".join(f"[INFO] PocketPredictor - pocket{i} - score: {i}.5" for i in range(400))
REAL = ("[ERROR] Dataset - error processing dataset item [4CBO.pdb]\n"
        "cz.siret.prank.program.PrankException: Failed to load structure from '/x/4CBO.pdb'\n"
        "Caused by: java.lang.NullPointerException: getAminoType() is null\n")


def result(code, stdout="", stderr=""):
    return types.SimpleNamespace(returncode=code, stdout=stdout, stderr=stderr)


def test_the_error_lines_are_found_under_the_noise():
    got = rx._errors(result(1, stdout=REAL + NOISE))
    assert "error processing dataset item [4CBO.pdb]" in got
    assert "PrankException" in got
    assert "pocket399" not in got, "the tail of the log is not the error"


def test_errors_falls_back_to_the_tail_when_nothing_matches():
    got = rx._errors(result(1, stdout=NOISE))
    assert got, "a message with no recognisable error line must still say something"


def run_batch(tmp_path, code, n_done, monkeypatch):
    out = tmp_path / "out"; out.mkdir()
    for i in range(n_done):
        (out / f"p{i}.pdb_predictions.csv").write_text("name,score,center_x,center_y,center_z,probability\n")
    monkeypatch.setattr(rx.subprocess, "run", lambda *a, **k: result(code, stdout=REAL + NOISE))
    return rx.run_p2rank_batch("prank", [pathlib.Path(f"p{i}.pdb") for i in range(10)], tmp_path, 1)


def test_a_partial_run_is_kept(tmp_path, monkeypatch):
    # Nine of ten structures done and a non-zero exit: the nine must survive.
    res = run_batch(tmp_path, 1, 9, monkeypatch)
    assert isinstance(res, dict) and len(res) == 10      # every requested id answered, absent ones with no rows


def test_a_run_that_produced_nothing_still_fails(tmp_path, monkeypatch):
    with pytest.raises(SystemExit) as e:
        run_batch(tmp_path, 1, 0, monkeypatch)
    assert "4CBO" in str(e.value), "the refusal has to name the cause"
