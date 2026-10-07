#!/usr/bin/env python
"""Run competing binding-site predictors on our structures and write their candidates in our table layout.

Supported, all optional and all installed by the user (none is imported by `src/` or `training/`):

| key | method | how it is invoked | licence of code / weights |
|-----|--------|-------------------|---------------------------|
| fpocket   | fpocket 4.x (alpha spheres)        | `$FPOCKET -f prot.pdb`, alpha-sphere centroids per pocket | MIT |
| p2rank    | P2Rank 2.5.x (random forest, SAS)  | `$PRANK predict dataset.ds` in one batch (needs Java)     | MIT |
| deeppocket| DeepPocket (CNN rescoring)         | `$DEEPPOCKET` wrapper script, fpocket candidates rescored | MIT (weights in release) |
| deepsurf  | DeepSurf (surface CNN)             | `$DEEPSURF` wrapper script                                | check before use |
| grasp     | GrASP (graph attention)            | `$GRASP` wrapper script                                   | check before use |
| vnegnn    | VN-EGNN (equivariant, virtual nodes)| `$VNEGNN` wrapper script                                 | code MIT, weights per release |

For the four learned methods there is no stable command line across releases, so each is called through a **wrapper
script** that the user points at with an environment variable. The wrapper receives

    <wrapper> <protein.pdb> <output.json>

and must write JSON: `[{"center": [x, y, z], "score": float, "rank": int}, ...]`, best first, in the coordinate frame
of the input PDB. `scripts/baselines/wrappers/` holds templates to fill in. This keeps their environments (conda,
CUDA, Java) entirely outside ours while the comparison stays exactly paired: same structures, same ligand filter,
same labels, same metrics.

Usage:
    PRANK=/opt/p2rank/prank   python scripts/baselines/run_competitors.py --tool p2rank
    # add --suffix _probe when testing on a few structures, so the full table is not overwritten
    VNEGNN=./scripts/baselines/wrappers/vnegnn.sh python scripts/baselines/run_competitors.py --tool vnegnn
    python scripts/baselines/run_competitors.py --tool all --manifest data/processed/manifest.csv

Output: data/processed/candidates_<tool>.csv, then
    python scripts/eval/compare_methods.py --tags native:nat_rank p2rank:tool_rank vnegnn:tool_rank
"""
import argparse, csv, json, os, pathlib, shutil, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts/train")); sys.path.insert(0, str(REPO / "scripts/baselines"))
from equicave import pockets as pk, structure, tables  # noqa: E402
from build_native import n_sites  # noqa: E402
from run_external import protein_only, run_fpocket, run_fpocket_prank_batch, run_p2rank_batch  # noqa: E402

WRAPPER_TOOLS = {"deeppocket": "DEEPPOCKET", "deepsurf": "DEEPSURF", "grasp": "GRASP", "vnegnn": "VNEGNN"}
ALL = ["fpocket", "p2rank", "fpocket_prank", *WRAPPER_TOOLS]


def run_wrapper(exe: str, prot: pathlib.Path, work: pathlib.Path, timeout: int = 900) -> list[dict]:
    """Call a user-provided wrapper and read its JSON predictions."""
    out = work / f"{prot.stem}.json"
    r = subprocess.run([exe, str(prot), str(out)], capture_output=True, text=True, timeout=timeout)
    if r.returncode or not out.exists():
        raise RuntimeError(f"wrapper failed ({r.returncode}): {(r.stderr or r.stdout)[-300:]}")
    data = json.loads(out.read_text())
    rows = []
    for i, d in enumerate(data, 1):
        rows.append(dict(center=np.asarray(d["center"], float), tool_score=float(d.get("score", -i)),
                         tool_rank=int(d.get("rank", i)), tool_extra=float(d.get("probability", d.get("score", 0.0)) or 0.0)))
    rows.sort(key=lambda x: x["tool_rank"])
    return rows


def label(rows, pdb, raw, codes, meta):
    """Label a method's predictions exactly as build_native labels ours: DCA <= 4 A to any kept ligand copy."""
    ligs = [l for l in structure.read_ligands(raw, min_heavy=8) if l["comp"] in codes]
    if not ligs or not rows:
        return []
    copies = [l["xyz"] for l in ligs]; L = np.vstack(copies); ns = n_sites(copies)
    finite = [r["tool_score"] for r in rows if r["tool_score"] == r["tool_score"]]
    mx = max(finite) if finite else 1e-9
    out = []
    for r in rows:
        d = pk.dca(r["center"], L)
        out.append(dict(pdb=pdb, center=";".join(f"{x:.2f}" for x in r["center"]), tool_score=r["tool_score"],
                        tool_rank=r["tool_rank"], tool_rel=(r["tool_score"] / mx if mx else 0.0), tool_extra=r["tool_extra"],
                        dca=d, dcc_min=min(pk.dcc(r["center"], c) for c in copies), label=int(d <= 4.0),
                        n_sites=ns, n_cands=len(rows), **meta))
    return out


def _exe(tool: str):
    """The executable for a tool, from its environment variable or the path, or None with a message."""
    var = {"fpocket": "FPOCKET", "p2rank": "PRANK"}.get(tool) or WRAPPER_TOOLS.get(tool, tool.upper())
    found = os.environ.get(var) or shutil.which({"p2rank": "prank"}.get(tool, tool))
    if not found:
        print(f"  {tool}: not available (set {var}); skipped")
    return found


def one_tool(tool: str, items, work: pathlib.Path, jobs: int) -> pd.DataFrame:
    if tool == "fpocket_prank":
        # The cascade needs both binaries; it is one method, so a missing half is a missing method.
        fp, pr = _exe("fpocket"), _exe("p2rank")
        if not (fp and pr):
            return pd.DataFrame()
        res = run_fpocket_prank_batch(fp, pr, [p for _, _, p in items], work, jobs)
        rows = []
        for r, raw, prot in items:
            rows += label(res.get(r["pdb"], []), r["pdb"], raw, {l[0] for l in json.loads(r["ligands"])},
                          dict(cluster30=r["cluster30"], fold=int(r["fold"])))
        return pd.DataFrame(rows)
    exe = _exe(tool)
    if not exe:
        return pd.DataFrame()
    rows = []
    if tool == "p2rank":
        res = run_p2rank_batch(exe, [p for _, _, p in items], work, jobs)
        for r, raw, prot in items:
            rows += label(res.get(r["pdb"], []), r["pdb"], raw, {l[0] for l in json.loads(r["ligands"])},
                          dict(cluster30=r["cluster30"], fold=int(r["fold"])))
    else:
        def go(it):
            r, raw, prot = it
            try:
                with tempfile.TemporaryDirectory(dir=work) as td:
                    res = run_fpocket(exe, prot, pathlib.Path(td)) if tool == "fpocket" else run_wrapper(exe, prot, pathlib.Path(td))
            except Exception as ex:  # noqa: BLE001
                print(f"    {r['pdb']}: {type(ex).__name__}: {ex}"[:160]); return []
            return label(res, r["pdb"], raw, {l[0] for l in json.loads(r["ligands"])},
                         dict(cluster30=r["cluster30"], fold=int(r["fold"])))
        with ThreadPoolExecutor(jobs) as ex:
            for got in ex.map(go, items):
                rows += got
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", default="all", choices=ALL + ["all"])
    ap.add_argument("--manifest", default=str(REPO / "data/processed/manifest.csv"))
    ap.add_argument("--pdb-dir", default=str(REPO / "data/pockets_ds/pdb"))
    ap.add_argument("--work", default=str(REPO / "data/pockets_ds/baselines"))
    ap.add_argument("--jobs", type=int, default=4); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--suffix", default="", help="appended to the output name, e.g. _coach420")
    a = ap.parse_args()
    man = list(csv.DictReader(open(a.manifest)))
    if a.limit:
        man = man[:a.limit]
    work = pathlib.Path(a.work); work.mkdir(parents=True, exist_ok=True)
    prot_dir = work / "prot"; prot_dir.mkdir(exist_ok=True)
    items = []
    for r in man:
        raw = pathlib.Path(a.pdb_dir) / f"{r['pdb']}.pdb"
        if raw.exists():
            items.append((r, raw, protein_only(raw, prot_dir / f"{r['pdb']}.pdb")))
    print(f"{len(items)} structures; tools: {a.tool}")
    for tool in (ALL if a.tool == "all" else [a.tool]):
        tw = work / tool; tw.mkdir(exist_ok=True)
        df = one_tool(tool, items, tw, a.jobs)
        if df.empty:
            continue
        out = tables.write_table(df, REPO / "data/processed", f"candidates_{tool}{a.suffix}")
        g = df.groupby("pdb")
        print(f"  {tool}: {len(df)} candidates in {g.ngroups} structures -> {out.name}; ceiling "
              f"{(g['label'].max() == 1).mean():.3f}; mean candidates {len(df) / g.ngroups:.1f}; "
              f"top-1 by its own order {(df[df.tool_rank == 1].groupby('pdb')['label'].max() == 1).sum() / g.ngroups:.3f}")


if __name__ == "__main__":
    main()
