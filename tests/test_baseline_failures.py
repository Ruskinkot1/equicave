"""The bookkeeping that keeps a competitor's score honest when a structure produces no answer.

A structure that yields no prediction is either the method refusing (a wrong answer, which belongs in its score),
a limit inside the method's own code (the method's, and excluded with the reason stated), or this machine giving
out (ours, and excluded as well but worth retrying). Confusing the last two moves a published number: on COACH420
it decides whether 14 structures are scored as misses, about 5 percentage points, in a comparison whose
differences already span zero. These tests pin the classifier and the exclusion list that keep them apart.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts/baselines"))
import failures as rd  # noqa: E402


def test_their_size_cap_is_theirs():
    # readSurfPoints returns None above 100000 DMS points and simplify_dms unpacks it: their cap, not our memory.
    assert not rd.ours("TypeError: cannot unpack non-iterable NoneType object")


def test_a_killed_process_is_ours():
    assert rd.ours(rd.KILLED)
    assert rd.ours("MemoryError")
    assert rd.ours("")


def test_excluded_reads_the_id_not_the_reason(tmp_path):
    f = tmp_path / "failed.txt"
    f.write_text("1E5Q\ttheirs\tTypeError: cannot unpack non-iterable NoneType object\n"
                 "1F8G\tours\ttheir process died before reporting\n")
    assert rd.read(f) == {"1E5Q", "1F8G"}


def test_retry_ours_lets_our_failures_back_in_only(tmp_path):
    f = tmp_path / "failed.txt"
    f.write_text("1E5Q\ttheirs\tTypeError: cannot unpack non-iterable NoneType object\n"
                 "1F8G\tours\ttheir process died before reporting\n"
                 "1Q51\ttheirs\tTypeError: cannot unpack non-iterable NoneType object\n")
    assert rd.read(f, retry_ours=True) == {"1E5Q", "1Q51"}


def test_a_line_with_no_reason_stays_excluded(tmp_path):
    # Written by a run from before the reason was recorded. Retrying it costs a process start-up to fail again,
    # so the unknown case keeps the structure out rather than guessing it was ours.
    f = tmp_path / "failed.txt"
    f.write_text("2WVA\n\n1F8G\tours\tkilled\n")
    assert rd.read(f, retry_ours=True) == {"2WVA"}


def test_missing_file_excludes_nothing(tmp_path):
    assert rd.read(tmp_path / "nope.txt") == set()


def test_record_writes_a_line_read_back_as_one_id(tmp_path):
    f = tmp_path / "failed.txt"
    rd.record(f, "1F8G", rd.KILLED)
    rd.record(f, "2WVA", "TypeError: cannot unpack non-iterable NoneType object")
    assert rd.read(f) == {"1F8G", "2WVA"}
    assert rd.read(f, retry_ours=True) == {"2WVA"}


def test_every_reader_parses_the_same_format(tmp_path):
    # The three consumers each read failed.txt for a different decision -- what to skip on resume, what counts as
    # a refusal, what to re-attempt -- and a reader that took the whole tab-separated line as an id would charge
    # the method for our failures again. They must therefore all go through this module.
    import add_missing_as_misses, repair_attempted, run_deeppocket, run_deepsurf
    for m in (add_missing_as_misses, repair_attempted, run_deeppocket, run_deepsurf):
        assert m.failures is rd, m.__name__
