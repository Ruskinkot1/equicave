#!/usr/bin/env python
"""Fetch the public benchmark lists (not redistributed here) into data/external/eval_sets and normalise them.

Sets and sources (see docs/DATA_CARD.md for licences):
  coach420, holo4k   PDB id lists from rdk/p2rank-datasets (ids only; structures from RCSB). The `.ds` files list
                     relative paths like coach420/1a26.pdb; `_mlig` variants mark the DeepPocket "relevant ligand" subsets.
  ligysis            Zenodo record 13121414 (CC-BY-4.0): the chain list of the LIGYSIS benchmark. The record holds
                     pandas pickles, so `LIGYSIS_3448_chains.pkl` is read with a **restricted unpickler** that allows
                     only numpy and pandas reconstruction calls, never arbitrary imports.
  cryptobench        skrhakv/CryptoBench (apo/holo pairs with cryptic sites); the dataset JSON is fetched from the
                     repository when reachable, otherwise the script reports the manual download step.
Output: data/external/eval_sets/<set>.csv with columns pdb, chain (may be empty), ligand_codes (may be empty), note.
"""
import csv, io, json, pathlib, pickle, sys, urllib.request

OUT = pathlib.Path(__file__).resolve().parents[2] / "data/external/eval_sets"
P2 = "https://raw.githubusercontent.com/rdk/p2rank-datasets/master/"
SOURCES = dict(coach420=P2 + "coach420.ds", coach420_mlig=P2 + "coach420(mlig).ds", holo4k=P2 + "holo4k.ds", holo4k_mlig=P2 + "holo4k(mlig).ds")
LIGYSIS_API = "https://zenodo.org/api/records/13121414"
OSF = "https://files.osf.io/v1/resources/pz4a9/providers/osfstorage/"


def get(url, timeout=120):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "equicave"}), timeout=timeout).read()


def parse_ds(txt: str):
    rows = []
    for ln in txt.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith(("#", "HEADER", "PARAM")):
            continue
        parts = ln.split()
        name = pathlib.Path(parts[0]).stem.lower()
        pdb, chain = name[:4].upper(), ""
        if len(name) > 4 and name[4] in "_":
            chain = name[5:].upper()
        elif len(name) == 5:
            chain = name[4].upper()
        ligs = parts[1] if len(parts) > 1 else ""
        rows.append(dict(pdb=pdb, chain=chain, ligand_codes=ligs, note=""))
    return rows


def write(name, rows, src):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pdb", "chain", "ligand_codes", "note"]); w.writeheader(); w.writerows(rows)
    print(f"{name}: {len(rows)} rows from {src}")


class SafeUnpickler(pickle.Unpickler):
    """Unpickler that allows only the numpy and pandas reconstruction calls a DataFrame needs.

    The LIGYSIS record publishes pandas pickles; plain `pickle.load` on a downloaded file would execute whatever the
    file asks for, so every other global is refused.
    """
    ALLOWED = {("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "_reconstruct"),
               ("numpy._core.multiarray", "_reconstruct"), ("numpy", "_frombuffer"),
               ("pandas.core.frame", "DataFrame"), ("pandas.core.series", "Series"),
               ("pandas.core.internals.managers", "BlockManager"), ("pandas.core.internals.managers", "SingleBlockManager"),
               ("pandas.core.internals.blocks", "new_block"), ("pandas.core.internals.blocks", "Block"),
               ("pandas.core.indexes.base", "_new_Index"), ("pandas.core.indexes.base", "Index"),
               ("pandas.core.indexes.range", "RangeIndex"), ("pandas.core.indexes.numeric", "Int64Index"),
               ("pandas._libs.internals", "_unpickle_block"), ("builtins", "object"), ("builtins", "list"),
               ("builtins", "dict"), ("builtins", "set"), ("builtins", "tuple"), ("builtins", "str")}

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED or module.startswith(("numpy", "pandas")) and not name.startswith("_"):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refused {module}.{name}")


def write_ligysis(status: dict):
    """The LIGYSIS chain list (PDB id + chain) from the published pickle."""
    rec = json.loads(get(LIGYSIS_API))
    files = {f["key"]: f["links"]["self"] for f in rec["files"]}
    key = next((k for k in files if "3448_chains" in k), None) or next((k for k in files if k.endswith(".pkl")), None)
    if key is None:
        raise RuntimeError(f"no pickle in the record: {sorted(files)}")
    obj = SafeUnpickler(io.BytesIO(get(files[key], 600))).load()
    entries = []
    if hasattr(obj, "itertuples"):                      # DataFrame
        cols = [str(c).lower() for c in obj.columns]
        for row in obj.astype(str).itertuples(index=False):
            d = dict(zip(cols, row))
            pdb = d.get("pdb_id") or d.get("pdb") or d.get("entry") or ""
            chain = d.get("chain_id") or d.get("chain") or d.get("auth_asym_id") or ""
            if len(pdb) >= 4:
                entries.append((pdb[:4].upper(), chain.upper().lstrip("_")))
    else:
        for x in (obj if isinstance(obj, (list, tuple, set)) else list(obj)):
            t = str(x)
            if len(t) >= 4:
                entries.append((t[:4].upper(), t[4:].upper().lstrip("_")))
    rows = [dict(pdb=p, chain=c, ligand_codes="", note=key) for p, c in dict.fromkeys(entries)]
    if not rows:
        raise RuntimeError(f"{key} parsed but held no chain ids")
    write("ligysis", rows, f"{LIGYSIS_API} :: {key}")
    status["ligysis"] = "ok"


def osf_tree(url=OSF):
    """{name: download link} of the CryptoBench OSF project, walked one level at a time."""
    out = {}

    def walk(u, prefix=""):
        d = json.loads(get(u))
        for x in d.get("data", []):
            a = x.get("attributes", {})
            name = prefix + (a.get("name") or "")
            if a.get("kind") == "folder":
                walk(x["links"]["move"], name + "/")
            else:
                out[name] = x["links"].get("download") or x["links"]["move"]
    walk(url)
    return out


def write_cryptobench(status: dict):
    """Apo structures and their cryptic-site residues; the test fold is kept as a separate set."""
    tree = osf_tree()
    want = {k: v for k, v in tree.items() if k.endswith(("dataset.json", "folds.json", "test.json"))}
    if not want:
        raise RuntimeError(f"no dataset.json in the OSF project; files seen: {sorted(tree)[:10]}")
    def rows_of(js):
        rows = []
        for apo, entries in js.items():
            pdb, chain = apo[:4].upper(), apo[4:].upper().lstrip("_")
            for e in (entries if isinstance(entries, list) else [entries]):
                lig, note = "", str(e)[:300]
                if isinstance(e, dict):
                    lig = e.get("ligand") or e.get("ligands") or e.get("holo_ligand") or ""
                    if isinstance(lig, list):
                        lig = ",".join(map(str, lig))
                    chain = (e.get("apo_chain") or chain).split("-")[0].upper()
                    note = json.dumps({k: e[k] for k in ("uniprot_id", "holo_pdb_id", "holo_chain", "apo_chain", "ligand") if k in e})
                rows.append(dict(pdb=pdb, chain=chain, ligand_codes=str(lig), note=note[:200]))
        return rows
    for key, url in sorted(want.items()):
        js = json.loads(get(url, 300))
        name = "cryptobench" if key.endswith("dataset.json") else ("cryptobench_test" if key.endswith("test.json") else None)
        if name is None:
            continue
        rows = rows_of(js)
        write(name, rows, f"OSF 10.17605/OSF.IO/PZ4A9 :: {key}")
        status[name] = "ok"


def main():
    status = {}
    for name, url in SOURCES.items():
        try:
            write(name, parse_ds(get(url).decode()), url); status[name] = "ok"
        except Exception as ex:  # noqa: BLE001
            status[name] = f"failed: {ex}"
    try:
        write_ligysis(status)
    except Exception as ex:  # noqa: BLE001
        status["ligysis"] = f"failed: {ex}"
    try:
        write_cryptobench(status)
    except Exception as ex:  # noqa: BLE001
        status["cryptobench"] = f"failed: {ex}"
    (OUT / "status.json").write_text(json.dumps(status, indent=1))
    print(json.dumps(status, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
