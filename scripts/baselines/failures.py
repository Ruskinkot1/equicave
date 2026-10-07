"""The record of structures that produced no prediction, and whose fault that was.

A structure a competitor's driver got nothing out of is one of three things, and the three must not be summed:

* the method ran and answered "no site" -- a prediction, and a wrong one, which belongs in its score as a miss;
* a limit written into the method's own code -- the method's, excluded with the reason stated;
* this machine giving out -- ours, excluded as well, and worth another try when it is idle.

The first is handled where the rows are written. The other two go in `failed.txt`, and the difference between
them is the one that reversed a published claim once already: five structures our pipeline died on had been
scored as GrASP's misses, which moved its paired difference against P2Rank from significant to not. So the
reason is kept with the id, in one format, parsed in one place -- the file is read by the drivers on resume, by
`add_missing_as_misses.py` deciding what counts as a refusal, and by `repair_attempted.py`.

Format: `id<TAB>ours|theirs<TAB>reason`. A bare id is a line from before the reason was recorded and is read as
theirs, since retrying costs a process start-up to fail again while wrongly retrying nothing costs a structure.
"""
import pathlib

# A failure signature belonging to the method rather than to us. DeepSurf's `readSurfPoints` returns None above
# 100000 DMS surface points and `simplify_dms` unpacks it unguarded, so their cap on receptor size arrives as a
# TypeError; on COACH420 it fires on the three structures above 20000 atoms and no others.
THEIR_LIMITS = ("cannot unpack non-iterable NoneType",)

KILLED = "their process died before reporting"


def ours(reason: str) -> bool:
    """True when the failure is this machine's rather than a limit written into their code."""
    return not any(s in reason for s in THEIR_LIMITS)


def record(path: pathlib.Path, pdb_id: str, reason: str = "") -> None:
    """Append one failure, with whose it is, as it happens."""
    with open(path, "a") as fh:
        fh.write(f"{pdb_id}\t{'ours' if ours(reason) else 'theirs'}\t{reason}\n")


def read(path: pathlib.Path, retry_ours: bool = False) -> set:
    """The recorded failures' ids, dropping the ones worth retrying when `retry_ours`."""
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        f = line.split("\t")
        if not f[0].strip():
            continue
        if retry_ours and len(f) > 1 and f[1] == "ours":
            continue
        out.add(f[0].strip())
    return out
