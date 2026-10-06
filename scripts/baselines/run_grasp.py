#!/usr/bin/env python
"""Run GrASP (Tiwary group, MIT) on our structures and write its predictions in our candidate-table layout.

GrASP is the only modern deep-learning pocket predictor whose **weights are public** -- they are committed in its
repository (`trained_models/`), under MIT. Checked against the primary sources on 2026-10-05: VN-EGNN publishes
datasets on Zenodo but no checkpoints and evaluates from a Weights-and-Biases run id, so it cannot be run without
retraining; EquiPocket ships only as a baseline implementation inside the VN-EGNN repository, also without weights;
DeepPocket publishes weights but needs libmolgrid, whose build is CUDA-oriented. GrASP therefore gives us the first
head-to-head against a current method rather than against P2Rank (2018) and fpocket (2009) alone.

What this script does per structure: GrASP's own `parse_files.py` builds the graph, its own `infer_test_set.py`
scores every surface atom, and its own `site_metrics.cluster_atoms_meanshift` clusters the scored atoms into ranked
sites -- so the method is theirs end to end. We only convert the result into the table our evaluation reads, and we
apply the identical labels, ligand rule and non-redundancy that every other method in `compare_on_set.py` gets.

Four deviations from their pinned environment, recorded because they are ours and not the authors':
  * openbabel 3.1 instead of 2.4.1 (the 2.4 series has no wheel; `parse_files.py`'s import is adjusted)
  * networkx 2.8.8 instead of 2.5 (`from_scipy_sparse_matrix` is gone in 3.x)
  * torch.load is called with weights_only=False for the graph cache their own parse step writes
  * CPU inference
The first two could in principle change their features; the comparison is reported with that stated.

Usage: GRASP_REPO=/path/to/GrASP python scripts/baselines/run_grasp.py --set coach420 [--limit N] [--jobs 4]
Output: data/processed/candidates_grasp_<set>.csv.gz, in the same layout as the other baseline tables. Results are
written batch by batch into data/processed/grasp_<set>_chunks/ and a re-run resumes from them (--no-resume to start
over), because a full set is hours of inference and the first attempt was lost to an interruption.
"""
import argparse, csv, json, os, pathlib, shutil, subprocess, sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts/eval"))
from equicave import labels as LB, pockets as pk, progress, structure, tables  # noqa: E402

MODEL_DEFAULT = "train_full"       # GrASP's model trained on its full training set


def grasp_predict(repo: pathlib.Path, pdb_paths: list, model: str, threshold: float, jobs: int, log=print) -> dict:
    """Run GrASP's own pipeline on a batch and return {pdb_id: [(center, score), ...]} best first."""
    work = repo / "benchmark_data_dir" / "production"
    # Both sides are cleared, not just the inputs: GrASP writes its scores under test_metrics/production and keeps
    # them, so a later batch would find a previous batch's prediction whose mol2 has been deleted and fail on it.
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(repo / "test_metrics" / "production", ignore_errors=True)
    (work / "unprocessed_inputs").mkdir(parents=True)
    for p in pdb_paths:
        shutil.copy(p, work / "unprocessed_inputs" / pathlib.Path(p).name)
    env = dict(os.environ, PYTHONPATH=f"{repo}:{os.environ.get('PYTHONPATH', '')}")
    for step in (["parse_files.py", "production"], ["infer_test_set.py"]):
        r = subprocess.run([sys.executable] + step, cwd=repo, env=env, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"GrASP {step[0]} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        log(f"  {step[0]} ok")

    import MDAnalysis as mda
    sys.path.insert(0, str(repo))
    import site_metrics as SM

    out = {}
    probs_root = repo / "test_metrics" / "production" / "probs"
    model_dirs = sorted(d for d in probs_root.glob(f"{model}/*/*/*") if d.is_dir())
    if not model_dirs:
        raise RuntimeError(f"no GrASP predictions under {probs_root / model}")
    mdir = model_dirs[0]
    rel = mdir.relative_to(probs_root)
    for f in sorted(mdir.glob("*.npy")):
        pdb = f.stem
        p = np.load(f)
        idx = np.load(probs_root.parent / "indices" / rel / f"{pdb}.npy")
        sasa = np.load(probs_root.parent / "SASAs" / f"{pdb}.npy")
        u = mda.Universe(str(work / "mol2" / f"{pdb}.mol2"))
        coords = u.atoms[idx].positions[sasa]
        if len(coords) != len(p):
            log(f"  {pdb}: {len(coords)} coordinates against {len(p)} scores; skipped")
            continue
        bind, ranks, _ = SM.cluster_atoms_meanshift(coords, p, threshold=threshold)
        if bind is None or ranks is None or not len(bind):
            # Their clustering returns (None, None, None) when no atom passes the threshold: the method predicts no
            # site for this structure. That is a prediction, not an error, and it counts as a miss in the metrics.
            out[pdb] = []
            continue
        ranks = np.asarray(ranks)
        lab = p[:, 1] > threshold
        sites = []
        for c in np.unique(ranks):
            sel = bind[ranks == c]
            if not len(sel):
                continue
            sites.append((sel.mean(0), float(p[:, 1][lab][ranks == c].mean())))
        sites.sort(key=lambda s: -s[1])
        out[pdb] = sites
    return out


def predict_bisect(repo, paths, model, threshold, jobs, log=print) -> dict:
    """GrASP on a batch, halving it on failure so one bad structure costs a retry and not the whole batch.

    Their parse step runs joblib across every core and a single pathological structure kills the whole Parallel call.
    Isolating every structure would be correct but costs a process start-up each -- measured at about 70 s per
    structure, since torch, PyTorch Geometric and MDAnalysis are imported twice per invocation. Halving on failure
    keeps batch speed where structures are fine and narrows down to the single offender where they are not, at a
    cost logarithmic rather than linear in the batch size.
    """
    try:
        return grasp_predict(repo, paths, model, threshold, jobs, log=lambda _m: None)
    except Exception as ex:                                     # noqa: BLE001 -- see the comment below
        # Any exception, not only the RuntimeError we raise for a non-zero exit: a surprise from their code or from
        # our conversion of it must cost the structure that caused it, never the rest of the benchmark.
        if len(paths) == 1:
            log(f"  {pathlib.Path(paths[0]).stem}: GrASP failed, skipped ({str(ex).splitlines()[0][:80]})")
            return {}
        mid = len(paths) // 2
        log(f"  a batch of {len(paths)} failed; halving")
        out = predict_bisect(repo, paths[:mid], model, threshold, jobs, log)
        out.update(predict_bisect(repo, paths[mid:], model, threshold, jobs, log))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--repo", default=os.environ.get("GRASP_REPO", ""))
    ap.add_argument("--model", default=MODEL_DEFAULT,
                    help="which trained_models subdirectory to use; train_full is trained on GrASP's full "
                         "training set, coach420_mlig and holo4k_mlig are the arms they evaluate on those sets")
    ap.add_argument("--threshold", type=float, default=0.5, help="GrASP's own per-atom site threshold")
    # One structure per invocation by default. Their parse step runs joblib across every core and a single
    # pathological structure takes the whole Parallel call down with it, so a larger batch loses every
    # structure in it: measured on COACH420, batches of 40 lost 220 of 300 that way. Per-structure isolation
    # costs a few seconds of start-up each and loses only the structure that actually fails.
    ap.add_argument("--batch", type=int, default=24); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="ignore the partial chunks of an earlier run and start the set again")
    a = ap.parse_args()
    if not a.repo:
        sys.exit("set GRASP_REPO or pass --repo: a checkout of https://github.com/tiwarylab/GrASP")
    repo = pathlib.Path(a.repo)

    from evaluate import load_set, fetch, EXCLUDE
    rows = load_set(a.set, a.limit)
    print(f"{a.set}: {len(rows)} entries")

    # Written batch by batch and resumed from what is there. A full set takes hours of inference inside their code,
    # and a run that only writes its table at the end loses all of it to one interruption -- which is how the first
    # COACH420 run was lost.
    chunk_dir = REPO / "data/processed" / f"grasp_{a.set}_chunks"   # *_chunks is the gitignored convention
    chunk_dir.mkdir(parents=True, exist_ok=True)
    recs, seen, part = [], set(), 0
    if a.resume:
        for f in sorted(chunk_dir.glob("part_*.csv")):
            try:
                d = pd.read_csv(f)
            except Exception:                                   # noqa: BLE001 -- a batch interrupted mid-write
                print(f"  unreadable chunk {f.name}; ignored"); continue
            recs += d.to_dict("records"); seen |= set(d["pdb"]); part += 1
        attempted = chunk_dir / "attempted.txt"
        if attempted.exists():
            seen |= {l.strip() for l in attempted.read_text().splitlines() if l.strip()}
        if seen:
            print(f"  resuming: {len(seen)} structures already done, {len(recs)} predictions kept")
    else:
        attempted = chunk_dir / "attempted.txt"
    rows = [r for r in rows if r["pdb"] not in seen]

    done = 0
    bar = progress.Bar(f"GrASP on {a.set}", len(rows), unit="pdb")
    for i in range(0, len(rows), a.batch):
        chunk = rows[i:i + a.batch]
        paths, keep = [], []
        for r in chunk:
            p = fetch(r["pdb"])
            if p is not None:
                paths.append(p); keep.append(r)
        if not paths:
            bar.update(len(chunk), postfix=f"{len(recs)} predictions"); continue
        batch_recs = []
        pred = predict_bisect(repo, paths, a.model, a.threshold, a.jobs)
        for r, path in zip(keep, paths):
            sites = pred.get(r["pdb"], [])
            codes = set(filter(None, r.get("ligand_codes", "").replace(";", ",").split(","))) or None
            ligs = [l for l in structure.read_ligands(path, min_heavy=8, exclude=EXCLUDE if codes is None else set())
                    if (codes is None or l["comp"] in codes)]
            if not ligs or not sites:
                continue
            groups = LB.group_sites(ligs)
            site_atoms = [np.vstack([ligs[j]["xyz"] for j in g]) for g in groups]
            L = np.vstack([l["xyz"] for l in ligs])
            for rank, (centre, score) in enumerate(sites, 1):
                batch_recs.append(dict(pdb=r["pdb"], center=";".join(f"{x:.2f}" for x in centre),
                                       tool_score=score, tool_rank=rank, tool_rel=score / max(1e-9, sites[0][1]),
                                       dca=pk.dca(centre, L), dcc_min=min(pk.dcc(centre, s) for s in site_atoms),
                                       label=int(pk.dca(centre, L) <= 4.0), n_sites=len(groups), n_cands=len(sites)))
        # Both halves of the batch's outcome are recorded: the predictions, and the ids attempted. Without the
        # second a resumed run would retry every structure GrASP predicted nothing for, which on a set with many
        # of those never finishes.
        if batch_recs:
            pd.DataFrame(batch_recs).to_csv(chunk_dir / f"part_{part:04d}.csv", index=False); part += 1
            recs += batch_recs
        with open(attempted, "a") as fh:
            fh.write("".join(f"{r['pdb']}\n" for r in keep))
        done += len(keep)
        bar.update(len(chunk), postfix=f"{len(recs)} predictions from {done} structures")
    bar.close()

    if not recs:
        sys.exit("GrASP produced no predictions")
    df = pd.DataFrame(recs)
    tables.write_table(df, REPO / "data/processed", f"candidates_grasp_{a.set}")
    hit = df.groupby("pdb")["label"].max().mean()
    print(f"{len(df)} predictions for {df.pdb.nunique()} structures; "
          f"mean {len(df) / df.pdb.nunique():.1f} per structure; ceiling {hit:.3f}")


if __name__ == "__main__":
    main()
