#!/usr/bin/env python
"""Turn a benchmark list into a manifest the candidate builders and the baseline runners accept.

This is what makes the head-to-head honest: fpocket, P2Rank and any wrapped competitor then run on the *same*
structures, with the *same* ligand filter and the *same* labels as our own evaluation, instead of being compared
against a number copied from their paper. Clusters come from the evaluation's own candidate table when it exists
(so the train-similar split is identical), otherwise from RCSB.

Usage: python scripts/data/eval_set_to_manifest.py --set coach420 [--out data/processed/manifest_coach420.csv]
"""
import argparse, csv, json, pathlib, sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
EXT = REPO / "data/external/eval_sets"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True); ap.add_argument("--out", default="")
    ap.add_argument("--ligand-rule", choices=["mlig", "all"], default="mlig")
    a = ap.parse_args()
    sys.path.insert(0, str(REPO / "scripts/eval"))
    from evaluate import load_set
    rows = load_set(a.set, 0, a.ligand_rule)
    clusters = {}
    cand = REPO / "data/processed" / f"eval_candidates_{a.set}.csv"
    if cand.exists():
        for r in csv.DictReader(open(cand)):
            clusters[r["pdb"]] = r.get("cluster30") or r["pdb"]
        print(f"clusters taken from {cand.name} ({len(clusters)} structures), so the splits match the evaluation")
    out_rows = []
    for r in rows:
        codes = [c for c in r.get("ligand_codes", "").replace(";", ",").split(",") if c]
        chains = list(r.get("chain") or "")
        out_rows.append(dict(pdb=r["pdb"], resolution="", cluster30=clusters.get(r["pdb"], r["pdb"]), fold=0,
                             uniprot="", chains="".join(chains),
                             ligands=json.dumps([[c, chains] for c in codes] or [["ANY", chains]])))
    out = pathlib.Path(a.out or REPO / "data/processed" / f"manifest_{a.set}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)
    print(f"{len(out_rows)} entries -> {out}")


if __name__ == "__main__":
    main()
