#!/usr/bin/env python
"""Step 0: choose training structures from the RCSB PDB (CC0) and write the manifest with clusters and folds.

Selection: X-ray, resolution <= --max-res, protein-only polymers, <= 8 polymer instances, at least one non-polymer
ligand of 150-900 Da that is not a solvent / buffer / ion / common additive (EXCLUDE). Redundancy and leakage control
with RCSB's own 30 % sequence-identity clusters: at most --per-cluster entries per cluster, 5 folds assigned per
cluster (a cluster never spans folds), and every cluster / UniProt accession of the held-out targets removed.

Writes data/processed/manifest.csv (pdb, resolution, cluster30, fold, uniprot, chains, ligands as JSON
[[comp, chains, type, weight, charge], ...]) and data/processed/heldout_targets.json.
Network: RCSB search + GraphQL only. Usage: python scripts/data/build_manifest.py --n 3200 --seed 0
"""
import argparse, csv, hashlib, json, pathlib, random, sys, time, urllib.request

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from equicave import progress  # noqa: E402
from equicave.targets import TARGETS  # noqa: E402

EXCLUDE = set("""HOH DOD WAT NA K CL BR IOD F MG CA ZN MN FE FE2 CU CU1 NI CO CD HG PT AU SR BA CS LI RB SO4 PO4 SO3 NO3 NH4 CO3
ACT ACY FMT GOL EDO PEG PGE PG4 PG5 PG6 1PE 2PE P6G PE3 PE4 PEU DMS MPD TRS EPE MES HEZ BME DTT TAM IPA EOH MOH CIT TLA FLC OXL
MLI SUC IMD BCT ACE NH2 SCN AZI CYN NO2 OH O DIO PGO PDO BU1 BU2 12P 15P 16P OCT 4MO PHB URE SPD SPM PUT NAG NDG BMA MAN FUC
GAL GLC BGC FRU SIA XYS A2G NGA MAL TRE LAT CEL SGN RIB OLC OLA OLB PLM STE MYR LDA SDS BOG LMT DDM UNL UNK""".split())
SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
GQL = "https://data.rcsb.org/graphql"
FIELDS = '''rcsb_id rcsb_entry_info { resolution_combined }
 polymer_entities { rcsb_polymer_entity_container_identifiers { auth_asym_ids }
   rcsb_polymer_entity_group_membership { aggregation_method group_id similarity_cutoff } }
 nonpolymer_entities { pdbx_entity_nonpoly { comp_id } rcsb_nonpolymer_entity_container_identifiers { auth_asym_ids }
   nonpolymer_comp { chem_comp { formula_weight type pdbx_formal_charge name } } }'''


def post(url, obj, tries=4):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, json.dumps(obj).encode(), {"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(req, timeout=180))
        except Exception as ex:  # noqa: BLE001
            if t == tries - 1:
                raise
            time.sleep(3 * (t + 1))


def search_ids(max_res):
    node = lambda attr, op, val: {"type": "terminal", "service": "text", "parameters": {"attribute": attr, "operator": op, "value": val}}
    q = {"query": {"type": "group", "logical_operator": "and", "nodes": [
        node("exptl.method", "exact_match", "X-RAY DIFFRACTION"),
        node("rcsb_entry_info.resolution_combined", "less_or_equal", max_res),
        node("rcsb_entry_info.selected_polymer_entity_types", "exact_match", "Protein (only)"),
        node("rcsb_entry_info.nonpolymer_entity_count", "greater_or_equal", 1),
        node("rcsb_entry_info.deposited_polymer_entity_instance_count", "less_or_equal", 8)]},
        "return_type": "entry", "request_options": {"return_all_hits": True}}
    return [x["identifier"] for x in post(SEARCH, q)["result_set"]]


def info(ids, desc=""):
    """Entry metadata from the GraphQL endpoint, 100 ids per request. Minutes at full scale, so it reports.

    `desc=None` silences the bar, for callers that batch the ids themselves and report their own total.
    """
    out = {}
    bar = progress.Bar(desc or f"metadata for {len(ids)} entries", len(ids), unit="entry") if desc is not None else None
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        d = post(GQL, {"query": "{ entries(entry_ids:%s) { %s } }" % (json.dumps(chunk), FIELDS)})
        for e in (d.get("data") or {}).get("entries") or []:
            if e:
                out[e["rcsb_id"]] = e
        if bar is not None:
            bar.update(len(chunk), postfix=f"{len(out)} resolved")
    if bar is not None:
        bar.close()
    return out


def groups(e, method, cutoff=None):
    out = set()
    for pe in e.get("polymer_entities") or []:
        for g in pe.get("rcsb_polymer_entity_group_membership") or []:
            if g["aggregation_method"] == method and (cutoff is None or g.get("similarity_cutoff") == cutoff):
                out.add(g["group_id"])
    return sorted(out)


def summarize(e):
    cl = groups(e, "sequence_identity", 30.0); up = groups(e, "matching_uniprot_accession")
    chains = sorted({c for pe in e.get("polymer_entities") or [] for c in pe["rcsb_polymer_entity_container_identifiers"]["auth_asym_ids"] or []})
    ligs = []
    for ne in e.get("nonpolymer_entities") or []:
        comp = ne["pdbx_entity_nonpoly"]["comp_id"]
        cc = ((ne.get("nonpolymer_comp") or {}).get("chem_comp") or {})
        fw = cc.get("formula_weight") or 0
        if comp not in EXCLUDE and 150 <= fw <= 900:
            ligs.append([comp, ne["rcsb_nonpolymer_entity_container_identifiers"]["auth_asym_ids"] or [],
                         cc.get("type") or "", round(fw, 1), cc.get("pdbx_formal_charge") or 0])
    if not cl or not ligs:
        return None
    return dict(cluster30=cl, uniprot=up, chains=chains, ligands=ligs)


def fold_of(cluster, k=5):
    return int(hashlib.md5(cluster.encode()).hexdigest(), 16) % k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3200, help="entries to inspect after shuffling")
    ap.add_argument("--per-cluster", type=int, default=3)
    ap.add_argument("--max-res", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "data/processed"))
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(a.seed)

    tinfo = info([t["pdb"] for t in TARGETS.values()], desc="metadata for the held-out targets")
    held = {}
    for name, t in TARGETS.items():
        e = tinfo.get(t["pdb"], {})
        held[name] = dict(pdb=t["pdb"], lig=t["lig"], cluster30=groups(e, "sequence_identity", 30.0), uniprot=groups(e, "matching_uniprot_accession"))
    (out / "heldout_targets.json").write_text(json.dumps(held, indent=1))
    bad_cl = {c for h in held.values() for c in h["cluster30"]}
    bad_up = {u for h in held.values() for u in h["uniprot"]}

    ids = search_ids(a.max_res)
    print(f"{len(ids)} entries match the RCSB query", flush=True)
    rnd.shuffle(ids)
    rows, per, dropped = [], {}, dict(no_ligand_or_cluster=0, target_family=0, cluster_cap=0)
    for e in info(ids[:a.n], desc=f"metadata for {min(a.n, len(ids))} candidate entries").values():
        s = summarize(e)
        if s is None:
            dropped["no_ligand_or_cluster"] += 1; continue
        if set(s["cluster30"]) & bad_cl or set(s["uniprot"]) & bad_up:
            dropped["target_family"] += 1; continue
        key = s["cluster30"][0]
        if per.get(key, 0) >= a.per_cluster:
            dropped["cluster_cap"] += 1; continue
        per[key] = per.get(key, 0) + 1
        rows.append(dict(pdb=e["rcsb_id"], resolution=(e["rcsb_entry_info"]["resolution_combined"] or [None])[0],
                         cluster30=key, fold=fold_of(key), uniprot=";".join(s["uniprot"]), chains="".join(s["chains"]),
                         ligands=json.dumps(s["ligands"])))
    rows.sort(key=lambda r: r["pdb"])
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    folds = [sum(r["fold"] == k for r in rows) for k in range(5)]
    print(f"kept {len(rows)} structures in {len(per)} clusters; folds {folds}; dropped {dropped}")
    print(f"held-out families: {len(bad_cl)} clusters, {len(bad_up)} UniProt accessions")


if __name__ == "__main__":
    main()
