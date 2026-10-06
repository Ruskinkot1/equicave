#!/usr/bin/env python
"""Run DeepSurf (Mylonas et al. 2021) on our structures and write its predictions in our table layout.

DeepSurf scores surface points with a 3D ResNet in a local frame, clusters the ligandable ones with mean shift and
ranks the resulting sites by mean score, so it returns ranked sites and is directly comparable with us on DCA
top-N. Its own code does all of that; this script only decides which structures to run and converts its output.

It cannot run in this repository's interpreter: it needs `tensorflow.contrib` (TF 1.x, removed in TF2) and
openbabel 2's top-level `pybel`, neither of which exists for Python 3.11. It therefore runs in its own interpreter
through `deepsurf_bridge.py`, which prints its ranked sites as JSON.

Deviations from the authors' setup, recorded because they are ours:
  * CPU inference, TensorFlow 1.15.5 from PyPI instead of `tensorflow-gpu==1.13.1`, numpy 1.18 and
    scikit-learn 0.22 instead of the pinned 1.13/0.20; openbabel 2.4.1 and DMS as published.
  * `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` is required: with the C++ protobuf implementation, importing
    TensorFlow after openbabel segfaults in this environment. The pure-Python implementation is slower to import
    and changes no arithmetic.
  * Heteroatoms are stripped from the input (their README requires a ligand-free structure and their `Protein`
    raises otherwise). Every other method in this comparison is given the same deposited assembly, and the
    stripping removes only the records that the labels are built from, never protein.
  * The `orig` ResNet-18 arm of their published models, which their paper reports as its best DCA arm.

Usage:
    DEEPSURF_REPO=/path/to/DeepSurf DEEPSURF_PYTHON=/path/to/their/python DEEPSURF_MODELS=/path/to/models \
    python scripts/baselines/run_deepsurf.py --set coach420
Output: data/processed/candidates_deepsurf_<set>.csv.gz, written batch by batch into
data/processed/deepsurf_<set>_chunks/ so an interrupted run resumes (--no-resume to start over).
"""
import argparse, json, os, pathlib, subprocess, sys, tempfile

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts/eval"))
from equicave import labels as LB, pockets as pk, progress, structure, tables  # noqa: E402

BRIDGE = pathlib.Path(__file__).resolve().parent / "deepsurf_bridge.py"


def run_bridge(a, tasks, work: pathlib.Path, errlog: pathlib.Path):
    """Yield `(pdb_id, sites or None)` from one bridge process over `tasks`, streamed as they finish."""
    lst = work / "list.txt"
    lst.write_text("".join(f"{i} {p}\n" for i, p in tasks))
    env = dict(os.environ, PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python")
    with open(errlog, "a") as err:
        proc = subprocess.Popen([a.python, str(BRIDGE), a.repo, str(lst), a.models, str(work), a.model,
                                 str(a.threshold)], stdout=subprocess.PIPE, stderr=err, text=True, env=env)
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line.startswith("RESULT "):
                _, pdb_id, blob = line.split(" ", 2)
                yield pdb_id, [(np.asarray(s["center"], float), float(s["score"])) for s in json.loads(blob)]
            elif line.startswith("FAILED "):
                _, pdb_id, msg = line.split(" ", 2)
                print(f"  {pdb_id}: {msg}", flush=True)
                yield pdb_id, None
        proc.wait()


def predict_batch(a, tasks, work: pathlib.Path, errlog: pathlib.Path):
    """Yield `(pdb_id, sites or None)` for every task, restarting the bridge when it dies mid-list.

    One subprocess per list, because building their TensorFlow graph costs more than the inference -- but a
    process that dies on a structure would otherwise take every structure after it with it, silently. Their
    stderr goes to `errlog` rather than being discarded, since a crash is the thing worth reading.
    """
    left = list(tasks)
    while left:
        before = len(left)
        got = set()
        for pdb_id, sites in run_bridge(a, left, work, errlog):
            got.add(pdb_id)
            yield pdb_id, sites
        left = [t for t in left if t[0] not in got]
        if left and len(left) == before:
            # The process died before reporting anything at all: drop the first structure, which is where it died,
            # and carry on with the rest rather than looping on it for ever.
            print(f"  {left[0][0]}: their process died before reporting; skipped", flush=True)
            yield left[0][0], None
            left = left[1:]
        elif left:
            print(f"  their process stopped with {len(left)} structures left; restarting it", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--repo", default=os.environ.get("DEEPSURF_REPO", ""))
    ap.add_argument("--python", default=os.environ.get("DEEPSURF_PYTHON", ""))
    ap.add_argument("--models", default=os.environ.get("DEEPSURF_MODELS", ""))
    ap.add_argument("--model", default="orig", choices=["orig", "lds"])
    ap.add_argument("--threshold", type=float, default=0.9, help="their ligandability threshold T")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per structure")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    a = ap.parse_args()
    for name, val in (("repo", a.repo), ("python", a.python), ("models", a.models)):
        if not val:
            sys.exit(f"set DEEPSURF_{name.upper()} or pass --{name}")

    from evaluate import load_set, fetch, EXCLUDE
    rows = load_set(a.set, a.limit)
    print(f"{a.set}: {len(rows)} entries", flush=True)

    chunk_dir = REPO / "data/processed" / f"deepsurf_{a.set}_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    attempted = chunk_dir / "attempted.txt"
    recs, seen, part = [], set(), 0
    if a.resume:
        for f in sorted(chunk_dir.glob("part_*.csv")):
            try:
                d = pd.read_csv(f)
            except Exception:                            # noqa: BLE001 -- a batch interrupted mid-write
                print(f"  unreadable chunk {f.name}; ignored"); continue
            recs += d.to_dict("records"); seen |= set(d["pdb"]); part += 1
        if attempted.exists():
            seen |= {l.strip() for l in attempted.read_text().splitlines() if l.strip()}
        if seen:
            print(f"  resuming: {len(seen)} structures already done, {len(recs)} predictions kept")
    rows = [r for r in rows if r["pdb"] not in seen]

    by_id = {r["pdb"]: r for r in rows}
    tasks = []
    for r in rows:
        path = fetch(r["pdb"])
        if path is not None:
            tasks.append((r["pdb"], path))
    print(f"{len(tasks)} structures with a local file", flush=True)

    batch_recs, done = [], 0
    with tempfile.TemporaryDirectory() as tmp, progress.Bar(f"DeepSurf on {a.set}", len(tasks), unit="pdb") as bar:
        for pdb_id, sites in predict_batch(a, tasks, pathlib.Path(tmp), chunk_dir / "their_stderr.log"):
            r = by_id[pdb_id]
            path = dict(tasks)[pdb_id]
            done += 1
            with open(attempted, "a") as fh:
                fh.write(f"{pdb_id}\n")
            if sites:
                codes = set(filter(None, r.get("ligand_codes", "").replace(";", ",").split(","))) or None
                ligs = [l for l in structure.read_ligands(path, min_heavy=8,
                                                          exclude=EXCLUDE if codes is None else set())
                        if (codes is None or l["comp"] in codes)]
                if ligs and not sites:
                    # The method was asked and answered "no site here". That is a prediction and a wrong one, so it
                    # has to appear in the table: a structure with no row at all drops out of the intersection that
                    # `compare_on_set.py` takes, which would quietly delete this method's failures from its own score.
                    batch_recs.append(dict(
                        pdb=r["pdb"], center="nan;nan;nan", tool_score=float("-inf"), tool_rank=1, tool_rel=0.0,
                        dca=float("inf"), dcc_min=float("inf"), label=0,
                        n_sites=len(LB.group_sites(ligs)), n_cands=0))
                if ligs and sites:
                    groups = LB.group_sites(ligs)
                    site_atoms = [np.vstack([ligs[j]["xyz"] for j in g]) for g in groups]
                    L = np.vstack([l["xyz"] for l in ligs])
                    for rank, (centre, score) in enumerate(sites, 1):
                        batch_recs.append(dict(
                            pdb=r["pdb"], center=";".join(f"{x:.2f}" for x in centre),
                            tool_score=score, tool_rank=rank, tool_rel=score / max(1e-9, sites[0][1]),
                            dca=pk.dca(centre, L), dcc_min=min(pk.dcc(centre, s) for s in site_atoms),
                            label=int(pk.dca(centre, L) <= 4.0), n_sites=len(groups), n_cands=len(sites)))
            if len(batch_recs) >= 100:
                pd.DataFrame(batch_recs).to_csv(chunk_dir / f"part_{part:04d}.csv", index=False)
                part += 1; recs += batch_recs; batch_recs = []
            bar.update(1, postfix=f"{len(recs) + len(batch_recs)} predictions from {done} structures")
    if batch_recs:
        pd.DataFrame(batch_recs).to_csv(chunk_dir / f"part_{part:04d}.csv", index=False)
        recs += batch_recs

    if not recs:
        sys.exit("DeepSurf produced no predictions")
    df = pd.DataFrame(recs)
    tables.write_table(df, REPO / "data/processed", f"candidates_deepsurf_{a.set}")
    hit = df.groupby("pdb")["label"].max().mean()
    print(f"{len(df)} predictions for {df.pdb.nunique()} structures; "
          f"mean {len(df) / df.pdb.nunique():.1f} per structure; ceiling {hit:.3f}")


if __name__ == "__main__":
    main()
