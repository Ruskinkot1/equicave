"""The progress reporting must never change what a loop produces, and must survive a missing tqdm."""
import builtins
import sys

import pytest

from equicave import progress


def run(monkeypatch, env, body):
    for k in ("EQUICAVE_PROGRESS",):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return body()


def test_track_yields_every_item_unchanged(monkeypatch, capsys):
    out = run(monkeypatch, dict(EQUICAVE_PROGRESS="plain"),
              lambda: list(progress.track(iter(range(7)), "items", total=7)))
    assert out == list(range(7))
    assert "items" in capsys.readouterr().err


def test_a_total_is_taken_from_the_iterable_when_it_has_one(monkeypatch, capsys):
    run(monkeypatch, dict(EQUICAVE_PROGRESS="plain"), lambda: list(progress.track([1, 2, 3], "sized")))
    assert "3/3" in capsys.readouterr().err


def test_off_prints_nothing(monkeypatch, capsys):
    out = run(monkeypatch, dict(EQUICAVE_PROGRESS="off"), lambda: list(progress.track(range(4), "quiet")))
    assert out == [0, 1, 2, 3]
    assert capsys.readouterr().err == ""


def test_breaking_out_of_the_loop_closes_the_bar(monkeypatch, capsys):
    def body():
        for x in progress.track(range(100), "partial", 100):
            if x == 2:
                break
        return x
    assert run(monkeypatch, dict(EQUICAVE_PROGRESS="plain"), body) == 2
    assert "partial: 2" in capsys.readouterr().err


def test_without_tqdm_the_bar_falls_back_to_plain_lines(monkeypatch, capsys):
    """tqdm is a soft dependency: the loops must still report when it is not installed."""
    real = builtins.__import__

    def no_tqdm(name, *a, **k):
        if name.split(".")[0] == "tqdm":
            raise ImportError("no tqdm")
        return real(name, *a, **k)

    monkeypatch.delitem(sys.modules, "tqdm", raising=False)
    monkeypatch.delitem(sys.modules, "tqdm.auto", raising=False)
    monkeypatch.setattr(builtins, "__import__", no_tqdm)
    monkeypatch.setenv("EQUICAVE_PROGRESS", "bar")          # asking for a bar must not raise
    with progress.Bar("fallback", 2) as b:
        b.update(1); b.update(1)
    assert "fallback: 2/2" in capsys.readouterr().err


def test_a_postfix_carries_the_loops_own_numbers(monkeypatch, capsys):
    def body():
        with progress.Bar("featurising", 2, unit="pdb") as b:
            b.update(1, postfix="ceiling 0.500")
            b.update(1, postfix="ceiling 0.750")
    run(monkeypatch, dict(EQUICAVE_PROGRESS="plain"), body)
    assert "ceiling 0.750" in capsys.readouterr().err


def test_a_bar_with_no_total_still_counts(monkeypatch, capsys):
    def body():
        return list(progress.track(iter("abc"), "unsized"))
    assert run(monkeypatch, dict(EQUICAVE_PROGRESS="plain"), body) == ["a", "b", "c"]
    assert "unsized: 3" in capsys.readouterr().err
