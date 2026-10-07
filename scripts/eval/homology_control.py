"""How much of each benchmark is homologous to what the competing methods were trained on.

The field's own leakage control, where it exists at all, is a difference of PDB-code sets, which catches only
literal re-use. Nobody has published the overlap as a percentage of 30 %-identity clusters at a stated threshold
with a stated tool, and P2Rank's disjointness guarantee is against CHEN11/JOINED rather than scPDB, so the
scPDB-trained methods -- EquiPocket, VN-EGNN, GDEGAN, PUResNet, Kalasanty, DeepSurf and DeepPocket among them --
inherit no guarantee at all. DeepPocket is the one method that removes homologues and reports the cost: 2418
structures dropped for COACH420 and 7951 for HOLO4K.

The exact training list of each method is not obtainable here, and for several methods it was never published. So
the corpus used is a **superset**: every X-ray protein-ligand complex in the PDB deposited on or before the scPDB
2017 release. Any method trained on scPDB, or on a subset of it, has its training structures inside this set, so
removing homologues to the superset over-removes and never under-removes. That direction is the conservative one
-- it shrinks the evaluation set rather than flattering anyone, ourselves included -- and it does not depend on an
author having published a split.

Homology is RCSB's own 30 %-identity entity clustering, the same `cluster30` the rest of the project uses, so the
threshold and the tool are both stated and neither is ours to tune.

This script only measures the overlap and the surviving subset sizes. That is the go/no-go: a benchmark whose
surviving subset is too small to carry a cluster bootstrap is not worth building, and that has to be known before
any accuracy is computed, not after.

    PYTHONPATH=src:. python scripts/eval/homology_control.py --cutoff 2017-12-31
"""
import argparse
import json
import pathlib
import sys

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts/data"))
from build_manifest import SEARCH, info, post  # noqa: E402

DS = REPO / "data/processed"
SETS = ["coach420", "holo4k", "ligysis", "cryptobench_test"]
TAGS = {"coach420": "", "holo4k": "_best", "ligysis": "_best", "cryptobench_test": "_best"}


def corpus_ids(cutoff: str, max_res: float = 3.0) -> list:
    """Every X-ray protein-ligand complex deposited on or before `cutoff`: the superset of scPDB-era training."""
    node = lambda a, o, v: {"type": "terminal", "service": "text",
                            "parameters": {"attribute": a, "operator": o, "value": v}}
    q = {"query": {"type": "group", "logical_operator": "and", "nodes": [
        node("exptl.method", "exact_match", "X-RAY DIFFRACTION"),
        node("rcsb_entry_info.resolution_combined", "less_or_equal", max_res),
        node("rcsb_entry_info.selected_polymer_entity_types", "exact_match", "Protein (only)"),
        node("rcsb_entry_info.nonpolymer_entity_count", "greater_or_equal", 1),
        node("rcsb_accession_info.deposit_date", "less_or_equal", cutoff)]},
        "return_type": "entry", "request_options": {"return_all_hits": True}}
    return [x["identifier"] for x in post(SEARCH, q)["result_set"]]


def clusters_of(ids: list) -> dict:
    """pdb id -> its RCSB 30 %-identity cluster ids."""
    out = {}
    for pid, e in info(ids, desc=f"clusters for {len(ids)} entries").items():
        cl = set()
        for pe in e.get("polymer_entities") or []:
            for g in pe.get("rcsb_polymer_entity_group_membership") or []:
                if g["aggregation_method"] == "sequence_identity" and g.get("similarity_cutoff") == 30.0:
                    cl.add(g["group_id"])
        out[pid] = cl
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2017-12-31", help="scPDB 2017's release; the corpus is everything before it")
    ap.add_argument("--max-res", type=float, default=3.0)
    ap.add_argument("--cache", default=str(DS / "homology_corpus_clusters.json"))
    ap.add_argument("--out", default=str(REPO / "docs/results/homology_control.json"))
    a = ap.parse_args()

    cache = pathlib.Path(a.cache)
    if cache.exists():
        train_cl = set(json.loads(cache.read_text())["clusters"])
        print(f"corpus from cache: {len(train_cl)} clusters")
    else:
        ids = corpus_ids(a.cutoff, a.max_res)
        print(f"{len(ids)} entries deposited on or before {a.cutoff}", flush=True)
        train_cl = {c for cl in clusters_of(ids).values() for c in cl}
        cache.write_text(json.dumps(dict(cutoff=a.cutoff, max_res=a.max_res, n_entries=len(ids),
                                         clusters=sorted(train_cl))))
        print(f"  {len(train_cl)} distinct 30 % clusters")

    ours = {c for c in pd.read_csv(DS / "manifest.csv")["cluster30"].dropna().astype(str) if c}
    rows = []
    for s in SETS:
        f = DS / f"eval_candidates_{s}{TAGS[s]}.csv"
        if not f.exists():
            print(f"  {s}: no evaluation table; skipped"); continue
        d = pd.read_csv(f, usecols=["pdb", "cluster30"]).drop_duplicates("pdb")
        d["cluster30"] = d["cluster30"].astype(str)
        theirs = d["cluster30"].isin(train_cl)
        mine = d["cluster30"].isin(ours)
        both = ~theirs & ~mine
        rows.append(dict(set=s, structures=len(d), clusters=d["cluster30"].nunique(),
                         similar_to_theirs=int(theirs.sum()),
                         similar_to_ours=int(mine.sum()),
                         novel_to_all=int(both.sum()),
                         novel_clusters=int(d.loc[both, "cluster30"].nunique())))
    t = pd.DataFrame(rows)
    print()
    print(t.to_string(index=False))
    pathlib.Path(a.out).write_text(json.dumps(dict(cutoff=a.cutoff, max_res=a.max_res,
                                                   corpus_clusters=len(train_cl), rows=rows), indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
