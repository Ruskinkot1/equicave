#!/usr/bin/env python
"""Build a peptide-binding-site benchmark from the RCSB PDB (CC0): protein-peptide complexes with clusters and folds.

No public benchmark of *peptide binding sites* with homology-controlled splits was found that we may redistribute
(PepBDB and Propedia are web databases with their own terms; BioLiP has no machine-readable licence), so the set is
built here from RCSB primary data and only PDB ids are stored.

Selection: X-ray, resolution <= --max-res, protein-only polymers, at least two polymer entities, at least one entity
whose sample sequence is --min-pep..--max-pep residues long (the peptide) and at least one entity of >= --min-rec
residues (the receptor). Entries whose 30 %-identity cluster or UniProt accession belongs to a held-out target family
are dropped. The receptor cluster (the cluster of the longest entity) defines redundancy: at most --per-cluster
entries per receptor cluster, and folds are assigned per receptor cluster so no cluster spans folds.
`--disjoint-from manifest.csv` additionally drops receptor clusters that occur in the small-molecule training manifest,
so a model trained there can be evaluated here without homology leakage.

Output: data/processed/manifest_peptide.csv with pdb, resolution, cluster30 (receptor), fold, uniprot, rec_chains,
pep_chains, pep_lengths, pep_seqs.
Usage: python scripts/data/build_peptide_manifest.py --n 4000 --seed 0
"""
import argparse, csv, hashlib, json, pathlib, random, sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from equicave.targets import TARGETS  # noqa: E402
from build_manifest import GQL, SEARCH, fold_of, groups, info, post  # noqa: E402

FIELDS = '''rcsb_id rcsb_entry_info { resolution_combined }
 polymer_entities { rcsb_id entity_poly { rcsb_sample_sequence_length pdbx_seq_one_letter_code_can }
   rcsb_polymer_entity_container_identifiers { auth_asym_ids }
   rcsb_polymer_entity_group_membership { aggregation_method group_id similarity_cutoff } }'''


def search_ids(max_res, min_pep, max_pep):
    node = lambda attr, op, val: {"type": "terminal", "service": "text", "parameters": {"attribute": attr, "operator": op, "value": val}}
    q = {"query": {"type": "group", "logical_operator": "and", "nodes": [
        node("exptl.method", "exact_match", "X-RAY DIFFRACTION"),
        node("rcsb_entry_info.resolution_combined", "less_or_equal", max_res),
        node("rcsb_entry_info.selected_polymer_entity_types", "exact_match", "Protein (only)"),
        node("rcsb_entry_info.polymer_entity_count_protein", "greater_or_equal", 2),
        {"type": "terminal", "service": "text", "parameters": {"attribute": "entity_poly.rcsb_sample_sequence_length",
                                                               "operator": "range", "value": {"from": min_pep, "to": max_pep}}}]},
        "return_type": "entry", "request_options": {"return_all_hits": True}}
    return [x["identifier"] for x in post(SEARCH, q)["result_set"]]


def entry_info(ids):
    out = {}
    for i in range(0, len(ids), 100):
        d = post(GQL, {"query": "{ entries(entry_ids:%s) { %s } }" % (json.dumps(ids[i:i + 100]), FIELDS)})
        for e in (d.get("data") or {}).get("entries") or []:
            if e:
                out[e["rcsb_id"]] = e
    return out


def summarize(e, min_pep, max_pep, min_rec):
    peps, recs = [], []
    for pe in e.get("polymer_entities") or []:
        ep = pe.get("entity_poly") or {}
        n = ep.get("rcsb_sample_sequence_length") or 0
        chains = pe["rcsb_polymer_entity_container_identifiers"]["auth_asym_ids"] or []
        seq = (ep.get("pdbx_seq_one_letter_code_can") or "").replace("\n", "")
        item = dict(n=n, chains=chains, seq=seq, entity=pe)
        if min_pep <= n <= max_pep:
            peps.append(item)
        if n >= min_rec:
            recs.append(item)
    if not peps or not recs:
        return None
    rec = max(recs, key=lambda x: x["n"])
    cl = groups(dict(polymer_entities=[rec["entity"]]), "sequence_identity", 30.0)
    up = groups(e, "matching_uniprot_accession")
    if not cl:
        return None
    return dict(cluster30=cl, uniprot=up, rec_chains=sorted({c for r in recs for c in r["chains"]}),
                pep_chains=sorted({c for p in peps for c in p["chains"]}),
                pep_lengths=[p["n"] for p in peps], pep_seqs=[p["seq"][:60] for p in peps])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000); ap.add_argument("--per-cluster", type=int, default=2)
    ap.add_argument("--max-res", type=float, default=2.5); ap.add_argument("--min-pep", type=int, default=4)
    ap.add_argument("--max-pep", type=int, default=25); ap.add_argument("--min-rec", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default=str(REPO / "data/processed"))
    ap.add_argument("--disjoint-from", default="")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    held = json.loads((out / "heldout_targets.json").read_text()) if (out / "heldout_targets.json").exists() else {}
    bad_cl = {c for h in held.values() for c in h.get("cluster30", [])}
    bad_up = {u for h in held.values() for u in h.get("uniprot", [])}
    train_cl = set()
    if a.disjoint_from:
        train_cl = {r["cluster30"] for r in csv.DictReader(open(a.disjoint_from))}
    ids = search_ids(a.max_res, a.min_pep, a.max_pep)
    print(f"{len(ids)} entries contain a {a.min_pep}-{a.max_pep}-residue polymer entity", flush=True)
    rnd = random.Random(a.seed); rnd.shuffle(ids)
    rows, per, dropped = [], {}, dict(no_pep_or_rec=0, target_family=0, cluster_cap=0, train_cluster=0)
    for e in entry_info(ids[:a.n]).values():
        s = summarize(e, a.min_pep, a.max_pep, a.min_rec)
        if s is None:
            dropped["no_pep_or_rec"] += 1; continue
        if set(s["cluster30"]) & bad_cl or set(s["uniprot"]) & bad_up:
            dropped["target_family"] += 1; continue
        key = s["cluster30"][0]
        if key in train_cl:
            dropped["train_cluster"] += 1; continue
        if per.get(key, 0) >= a.per_cluster:
            dropped["cluster_cap"] += 1; continue
        per[key] = per.get(key, 0) + 1
        rows.append(dict(pdb=e["rcsb_id"], resolution=(e["rcsb_entry_info"]["resolution_combined"] or [None])[0], cluster30=key,
                         fold=fold_of(key), uniprot=";".join(s["uniprot"]), rec_chains="".join(s["rec_chains"]),
                         pep_chains="".join(s["pep_chains"]), pep_lengths=json.dumps(s["pep_lengths"]), pep_seqs=json.dumps(s["pep_seqs"])))
    rows.sort(key=lambda r: r["pdb"])
    with open(out / "manifest_peptide.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    folds = [sum(r["fold"] == k for r in rows) for k in range(5)]
    print(f"kept {len(rows)} protein-peptide entries in {len(per)} receptor clusters; folds {folds}; dropped {dropped}")


if __name__ == "__main__":
    main()
