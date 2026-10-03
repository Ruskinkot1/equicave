#!/usr/bin/env python
"""Run fpocket or P2Rank on the manifest structures and write a candidate table in the native layout (comparison only).

Labels and n_sites follow scripts/train/build_native.py exactly (same ligands, same DCA <= 4 A rule), so numbers are
comparable. External tools see the protein-only PDB (ATOM records, all chains), as the native detector does.
"""
import argparse, csv, json, os, pathlib, shutil, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/train"))
from equicave import pockets as pk, structure, tables  # noqa: E402
from build_native import n_sites  # noqa: E402

ENV = dict(os.environ, JAVA_TOOL_OPTIONS=os.environ.get("JAVA_TOOL_OPTIONS", ""))


def protein_only(src: pathlib.Path, dst: pathlib.Path) -> pathlib.Path:
    keep = [ln for ln in src.read_text(errors="ignore").splitlines(keepends=True) if ln.startswith(("ATOM", "TER")) or (ln.startswith("HETATM") and ln[17:20] == "MSE")]
    dst.write_text("".join(keep) + "END\n")
    return dst


def run_fpocket(exe: str, prot: pathlib.Path, work: pathlib.Path) -> list[dict]:
    local = work / prot.name; shutil.copy(prot, local)
    subprocess.run([exe, "-f", str(local)], cwd=work, check=True, capture_output=True)
    od = work / f"{local.stem}_out"
    info = (od / f"{local.stem}_info.txt").read_text().split("Pocket ")[1:]
    out = []
    for i, blk in enumerate(info, 1):
        sc = {k.strip(): v.strip() for k, v in (ln.split(":", 1) for ln in blk.splitlines()[1:] if ":" in ln)}
        vert = od / "pockets" / f"pocket{i}_vert.pqr"
        xyz = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in vert.read_text().splitlines() if l.startswith(("ATOM", "HETATM"))])
        if len(xyz):
            out.append(dict(center=xyz.mean(0), tool_score=float(sc.get("Score", "nan")), tool_rank=i, tool_extra=len(xyz)))
    return out


def run_p2rank_batch(exe: str, prots: list[pathlib.Path], work: pathlib.Path, threads: int) -> dict[str, list[dict]]:
    ds = work / "all.ds"; ds.write_text("HEADER: protein\n\n" + "".join(f"{p}\n" for p in prots))
    r = subprocess.run([exe, "predict", str(ds), "-o", str(work / "out"), "-threads", str(threads)], capture_output=True, text=True, env=ENV)
    if r.returncode:
        sys.exit("P2Rank failed:\n" + (r.stderr or r.stdout)[-2000:])
    res = {}
    for p in prots:
        f = work / "out" / f"{p.name}_predictions.csv"
        rows = []
        if f.exists():
            for i, d in enumerate(csv.DictReader(open(f), skipinitialspace=True), 1):
                d = {k.strip(): v.strip() for k, v in d.items()}
                rows.append(dict(center=np.array([float(d["center_x"]), float(d["center_y"]), float(d["center_z"])]),
                                 tool_score=float(d["score"]), tool_rank=i, tool_extra=float(d["probability"])))
        res[p.stem] = rows
    return res


def label(rows, pdb, raw, codes, meta):
    ligs = [l for l in structure.read_ligands(raw, min_heavy=8) if l["comp"] in codes]
    if not ligs:
        return []
    copies = [l["xyz"] for l in ligs]; L = np.vstack(copies); ns = n_sites(copies)
    out = []
    mx = max([r["tool_score"] for r in rows if r["tool_score"] == r["tool_score"]] + [1e-9])
    for r in rows:
        d = pk.dca(r["center"], L)
        out.append(dict(pdb=pdb, center=";".join(f"{x:.2f}" for x in r["center"]), tool_score=r["tool_score"], tool_rank=r["tool_rank"],
                        tool_rel=r["tool_score"] / mx if r["tool_score"] == r["tool_score"] else 0.0, tool_extra=r["tool_extra"],
                        dca=d, dcc_min=min(pk.dcc(r["center"], c) for c in copies), label=int(d <= 4.0), n_sites=ns, n_cands=len(rows), **meta))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", choices=["fpocket", "p2rank"], required=True)
    ap.add_argument("--manifest", default=str(REPO / "data/processed/manifest.csv"))
    ap.add_argument("--pdb-dir", default=str(REPO / "data/pockets_ds/pdb"))
    ap.add_argument("--work", default=str(REPO / "data/pockets_ds/baselines"))
    ap.add_argument("--jobs", type=int, default=4); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    exe = os.environ.get(a.tool.upper() if a.tool == "fpocket" else "PRANK") or shutil.which("fpocket" if a.tool == "fpocket" else "prank")
    if not exe:
        sys.exit(f"{a.tool} not installed: set {'FPOCKET' if a.tool == 'fpocket' else 'PRANK'}")
    man = list(csv.DictReader(open(a.manifest)))
    if a.limit:
        man = man[:a.limit]
    work = pathlib.Path(a.work) / a.tool; work.mkdir(parents=True, exist_ok=True)
    prot_dir = work / "prot"; prot_dir.mkdir(exist_ok=True)
    items = []
    for r in man:
        raw = pathlib.Path(a.pdb_dir) / f"{r['pdb']}.pdb"
        if raw.exists():
            items.append((r, raw, protein_only(raw, prot_dir / f"{r['pdb']}.pdb")))
    print(f"{len(items)} structures", flush=True)
    rows = []
    if a.tool == "fpocket":
        def one(it):
            r, raw, prot = it
            try:
                with tempfile.TemporaryDirectory(dir=work) as td:
                    res = run_fpocket(exe, prot, pathlib.Path(td))
            except Exception as ex:  # noqa: BLE001
                print(f"  {r['pdb']}: {type(ex).__name__}", flush=True); return []
            return label(res, r["pdb"], raw, {l[0] for l in json.loads(r["ligands"])}, dict(cluster30=r["cluster30"], fold=int(r["fold"])))
        with ThreadPoolExecutor(a.jobs) as ex:
            for out in ex.map(one, items):
                rows += out
    else:
        res = run_p2rank_batch(exe, [p for _, _, p in items], work, a.jobs)
        for r, raw, prot in items:
            rows += label(res.get(r["pdb"], []), r["pdb"], raw, {l[0] for l in json.loads(r["ligands"])}, dict(cluster30=r["cluster30"], fold=int(r["fold"])))
    df = pd.DataFrame(rows)
    out = tables.write_table(df, REPO / "data/processed", f"candidates_{a.tool}")
    g = df.groupby("pdb")
    print(f"{len(df)} candidates in {g.ngroups} structures; ceiling {(g['label'].max() == 1).mean():.3f}; mean candidates {len(df) / g.ngroups:.1f}; "
          f"top-1 by tool order {(df[df.tool_rank == 1].groupby('pdb')['label'].max() == 1).sum() / g.ngroups:.3f}")


if __name__ == "__main__":
    main()
