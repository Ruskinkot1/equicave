#!/usr/bin/env python
"""Fetch the public benchmark lists (not redistributed here) into data/external/eval_sets and normalise them.

Sets and sources (see docs/DATA_CARD.md for licences):
  coach420, holo4k   PDB id lists from rdk/p2rank-datasets (ids only; structures from RCSB). The `.ds` files list
                     relative paths like coach420/1a26.pdb; `_mlig` variants mark the DeepPocket "relevant ligand" subsets.
  ligysis            Zenodo record 13121414 (CC-BY-4.0): chain list of the LIGYSIS benchmark.
  cryptobench        skrhakv/CryptoBench (apo/holo pairs with cryptic sites); the dataset JSON is fetched from the
                     repository when reachable, otherwise the script reports the manual download step.
Output: data/external/eval_sets/<set>.csv with columns pdb, chain (may be empty), ligand_codes (may be empty), note.
"""
import csv, io, json, pathlib, sys, urllib.request, zipfile

OUT = pathlib.Path(__file__).resolve().parents[2] / "data/external/eval_sets"
P2 = "https://raw.githubusercontent.com/rdk/p2rank-datasets/master/"
SOURCES = dict(coach420=P2 + "coach420.ds", coach420_mlig=P2 + "coach420(mlig).ds", holo4k=P2 + "holo4k.ds", holo4k_mlig=P2 + "holo4k(mlig).ds")
LIGYSIS_API = "https://zenodo.org/api/records/13121414"
CRYPTO = ["https://raw.githubusercontent.com/skrhakv/CryptoBench/master/dataset/dataset.json",
          "https://raw.githubusercontent.com/skrhakv/CryptoBench/main/dataset/dataset.json"]


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


def main():
    status = {}
    for name, url in SOURCES.items():
        try:
            write(name, parse_ds(get(url).decode()), url); status[name] = "ok"
        except Exception as ex:  # noqa: BLE001
            status[name] = f"failed: {ex}"
    try:
        rec = json.loads(get(LIGYSIS_API))
        files = {f["key"]: f["links"]["self"] for f in rec["files"]}
        rows = []
        for key, link in files.items():
            if key.endswith(".zip") and len(rows) == 0:
                z = zipfile.ZipFile(io.BytesIO(get(link, 600)))
                for n in z.namelist():
                    if n.endswith((".csv", ".txt")) and "ligysis" in n.lower() and ("chain" in n.lower() or "list" in n.lower()):
                        for ln in z.read(n).decode(errors="ignore").splitlines()[1:]:
                            p = ln.replace(",", " ").split()
                            if p and len(p[0]) >= 4:
                                rows.append(dict(pdb=p[0][:4].upper(), chain=p[1] if len(p) > 1 else "", ligand_codes="", note=n))
        if not rows:
            rows = [dict(pdb="", chain="", ligand_codes="", note=f"files in record: {sorted(files)}")]
        write("ligysis", rows, LIGYSIS_API); status["ligysis"] = "ok" if rows and rows[0]["pdb"] else "record fetched, chain list not auto-parsed"
    except Exception as ex:  # noqa: BLE001
        status["ligysis"] = f"failed: {ex}"
    rows = None
    for url in CRYPTO:
        try:
            js = json.loads(get(url)); rows = []
            for apo, entries in js.items():
                for e in (entries if isinstance(entries, list) else [entries]):
                    rows.append(dict(pdb=apo[:4].upper(), chain=apo[4:].upper(), ligand_codes=e.get("ligand", "") if isinstance(e, dict) else "",
                                     note=json.dumps(e)[:200] if isinstance(e, dict) else str(e)[:200]))
            write("cryptobench", rows, url); status["cryptobench"] = "ok"; break
        except Exception as ex:  # noqa: BLE001
            status["cryptobench"] = f"failed: {ex}"
    (OUT / "status.json").write_text(json.dumps(status, indent=1))
    print(json.dumps(status, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
