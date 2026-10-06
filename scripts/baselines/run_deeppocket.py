#!/usr/bin/env python
"""Run DeepPocket (Aggarwal et al. 2021, MIT) on our structures and write its predictions in our table layout.

DeepPocket ranks fpocket's candidate pockets with a 3D CNN, so it is directly comparable with us on DCA top-N: the
pockets it proposes are fpocket's, and the ranking is its own. Its own pipeline does the work here -- its
`clean_pdb`, the fpocket run, its `get_centers`, its `gninatype`/`create_types` gridding and its `Model` with the
published weights. We write only the inference loop (theirs is hard-wired to CUDA) and the conversion of its ranked
barycentres into our candidate table, then apply the identical labels, ligand rule and non-redundancy that every
other method in `compare_on_set.py` gets.

Four deviations from the authors' setup, recorded because they are ours and not theirs:
  * CPU inference. Their `rank_pockets.test_model` allocates its input tensor with `device='cuda'` and
    `get_model_gmaker_eprovider` calls `model.cuda()` unconditionally, so the loop here is written against the same
    molgrid calls with the device left to torch.
  * libmolgrid from the PyPI `molgrid` wheel (0.5.5) rather than a source build against CUDA.
  * The classification checkpoint comes from a **third-party Zenodo mirror** (record 13833813, CC-BY-4.0,
    `first_model_fold1_best_test_auc_85001.pth.tar`), because the authors' own SharePoint link answers 403 from
    here. The file loads into their `Model` with no missing or unexpected keys, which is checked before use.
  * Segmentation (their step 6) is not run: it refines pocket *shape* and cannot change the ranking, which is what
    a DCA/top-N comparison measures.

Usage:
    DEEPPOCKET_REPO=/path/to/DeepPocket FPOCKET=/path/to/fpocket \
    python scripts/baselines/run_deeppocket.py --set coach420 --checkpoint /path/to/first_model_fold1_*.pth.tar
Output: data/processed/candidates_deeppocket_<set>.csv.gz, written batch by batch into
data/processed/deeppocket_<set>_chunks/ so an interrupted run resumes (--no-resume to start over).
"""
import argparse, os, pathlib, shutil, subprocess, sys, tempfile

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts/eval"))
from equicave import labels as LB, pockets as pk, progress, structure, tables  # noqa: E402


def load_their_modules(repo: pathlib.Path):
    """Import DeepPocket's own modules from its checkout. They import each other by bare name."""
    sys.path.insert(0, str(repo))
    from clean_pdb import clean_pdb                      # noqa: PLC0415
    from get_centers import get_centers                  # noqa: PLC0415
    from model import Model                              # noqa: PLC0415
    from types_and_gninatyper import create_types, gninatype  # noqa: PLC0415
    return dict(clean_pdb=clean_pdb, get_centers=get_centers, Model=Model,
                gninatype=gninatype, create_types=create_types)


def load_model(their, checkpoint: pathlib.Path):
    import torch
    model = their["Model"]()
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ck.get("model_state_dict", ck)
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"the checkpoint does not match their Model: missing {missing}, unexpected {unexpected}")
    return model.eval()


def score_centers(model, types_file: pathlib.Path, batch: int = 16):
    """[(centre, probability), ...] for the barycentres in `types_file`, best first.

    The same molgrid calls their loop makes -- an ExampleProvider over the types file, a GridMaker at its default
    dimensions, the first 14 channels (the receptor ones) into their Model -- with the device left to torch so this
    runs without a GPU.
    """
    import molgrid
    import torch

    n_rows = sum(1 for _ in open(types_file))
    if not n_rows:
        return []
    ep = molgrid.ExampleProvider(shuffle=False, stratify_receptor=False, labelpos=0, balanced=False,
                                 iteration_scheme=molgrid.IterationScheme.LargeEpoch,
                                 default_batch_size=min(batch, n_rows))
    ep.populate(str(types_file))
    gmaker = molgrid.GridMaker()
    dims = gmaker.grid_dimensions(ep.num_types())
    bs = min(batch, n_rows)
    grid = torch.zeros((bs,) + dims, dtype=torch.float32)
    lab = torch.zeros((bs, 4), dtype=torch.float32)
    out = []
    with torch.no_grad():
        for ex in ep:
            ex.extract_labels(lab)
            for b in range(bs):
                c = molgrid.float3(float(lab[b][1]), float(lab[b][2]), float(lab[b][3]))
                gmaker.forward(c, ex[b].coord_sets[0], grid[b])
            p = torch.softmax(model(grid[:, :14]), dim=1)[:, 1]
            for b in range(bs):
                out.append((np.array([float(lab[b][1]), float(lab[b][2]), float(lab[b][3])]), float(p[b])))
            if len(out) >= n_rows:
                break
    out = out[:n_rows]
    out.sort(key=lambda s: -s[1])
    return out


def predict_one(their, model, pdb_path: str, fpocket: str, work: pathlib.Path):
    """DeepPocket's own six steps, minus the segmentation that cannot change the ranking."""
    prot = work / "prot.pdb"
    shutil.copy(pdb_path, prot)
    nowat = work / "prot_nowat.pdb"
    their["clean_pdb"](str(prot), str(nowat))
    r = subprocess.run([fpocket, "-f", str(nowat)], capture_output=True, text=True)
    pockets = nowat.with_name("prot_nowat_out") / "pockets"
    if r.returncode or not pockets.is_dir():
        raise RuntimeError(f"fpocket failed: {(r.stderr or r.stdout)[-400:]}")
    their["get_centers"](str(pockets))
    bary = pockets / "bary_centers.txt"
    if not bary.exists() or not bary.stat().st_size:
        return []                                        # fpocket found nothing: a prediction of no site
    types = their["create_types"](str(bary), their["gninatype"](str(nowat)))
    return score_centers(model, pathlib.Path(types))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--repo", default=os.environ.get("DEEPPOCKET_REPO", ""))
    ap.add_argument("--checkpoint", required=True, help="their classification checkpoint (.pth.tar)")
    ap.add_argument("--fpocket", default=os.environ.get("FPOCKET", "fpocket"))
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    a = ap.parse_args()
    if not a.repo:
        sys.exit("set DEEPPOCKET_REPO or pass --repo: a checkout of https://github.com/devalab/DeepPocket")
    their = load_their_modules(pathlib.Path(a.repo))
    model = load_model(their, pathlib.Path(a.checkpoint))

    from evaluate import load_set, fetch, EXCLUDE
    rows = load_set(a.set, a.limit)
    print(f"{a.set}: {len(rows)} entries", flush=True)

    chunk_dir = REPO / "data/processed" / f"deeppocket_{a.set}_chunks"
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

    batch_recs, done = [], 0
    with progress.Bar(f"DeepPocket on {a.set}", len(rows), unit="pdb") as bar:
        for r in rows:
            path = fetch(r["pdb"])
            if path is None:
                bar.update(1, postfix=f"{len(recs)} predictions"); continue
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    sites = predict_one(their, model, path, a.fpocket, pathlib.Path(tmp))
                except Exception as ex:                  # noqa: BLE001 -- one structure must not cost the set
                    print(f"  {r['pdb']}: {type(ex).__name__}: {str(ex).splitlines()[0][:120]}", flush=True)
                    sites = None
            done += 1
            with open(attempted, "a") as fh:
                fh.write(f"{r['pdb']}\n")
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
            if len(batch_recs) >= 200:
                pd.DataFrame(batch_recs).to_csv(chunk_dir / f"part_{part:04d}.csv", index=False)
                part += 1; recs += batch_recs; batch_recs = []
            bar.update(1, postfix=f"{len(recs) + len(batch_recs)} predictions from {done} structures")
    if batch_recs:
        pd.DataFrame(batch_recs).to_csv(chunk_dir / f"part_{part:04d}.csv", index=False)
        recs += batch_recs

    if not recs:
        sys.exit("DeepPocket produced no predictions")
    df = pd.DataFrame(recs)
    tables.write_table(df, REPO / "data/processed", f"candidates_deeppocket_{a.set}")
    hit = df.groupby("pdb")["label"].max().mean()
    print(f"{len(df)} predictions for {df.pdb.nunique()} structures; "
          f"mean {len(df) / df.pdb.nunique():.1f} per structure; ceiling {hit:.3f}")


if __name__ == "__main__":
    main()
