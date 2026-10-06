#!/usr/bin/env python
"""Bridge run **inside DeepSurf's own Python 3.7 environment**: predict a list of structures, one JSON line each.

DeepSurf needs TensorFlow 1.x (`tensorflow.contrib`, removed in TF2) and openbabel 2's top-level `pybel`, neither
of which exists for the Python this repository runs on, so it lives in its own interpreter and talks over stdout.
Its own classes do the work -- `Protein` (DMS surface, simplification), `Network` (their ResNet and checkpoint),
`Bsite_extractor` (their mean-shift clustering and their ranking by mean ligandability). Nothing is reimplemented
here: the script reads `prot.binding_sites` after their `sort_bsites()` has ordered it, which is their ranking.

A whole list is processed in one process on purpose. Building their TensorFlow graph costs over a minute here --
the pure-Python protobuf implementation this environment needs is slow to import -- and that cost is paid once for
the list instead of once per structure.

Usage (not called directly; `run_deepsurf.py` calls it):
    <their python> deepsurf_bridge.py <their repo> <list file> <model dir> <work dir> [orig|lds] [threshold]
where <list file> has one `<id> <path to pdb>` per line. One line of output per structure:
    RESULT <id> <json list of {"center", "score"}>      or      FAILED <id> <message>
"""
from __future__ import print_function

import json
import os
import sys
import traceback

KEEP = ("ATOM  ", "TER", "END")


def strip_heteroatoms(src, dst):
    """Keep only polymer ATOM records. Their README requires a ligand-free structure ("waters, ions, ligands
    should be removed") and their Protein() raises on anything else, so this is input preparation, not a change
    to their method."""
    with open(src) as fin, open(dst, "w") as fout:
        for ln in fin:
            if ln.startswith(KEEP):
                fout.write(ln)
    return dst


def main():
    repo, list_file, model_path, work = sys.argv[1:5]
    model = sys.argv[5] if len(sys.argv) > 5 else "orig"
    thres = float(sys.argv[6]) if len(sys.argv) > 6 else 0.9
    sys.path.insert(0, repo)
    os.chdir(repo)                                   # their modules import each other by bare name
    from protein import Protein
    from network import Network
    from bsite_extraction import Bsite_extractor

    net = Network(model_path, model, 1.0)            # their graph and checkpoint, built once for the whole list
    extractor_threshold = thres
    print("READY", flush=True)

    for line in open(list_file):
        line = line.strip()
        if not line:
            continue
        pdb_id, path = line.split(None, 1)
        d = os.path.join(work, pdb_id)
        try:
            if not os.path.isdir(d):
                os.makedirs(d)
            prot = Protein(strip_heteroatoms(path, os.path.join(d, "receptor.pdb")),
                           False, False, 10, d, True, 2020)
            scores = net.get_lig_scores(prot, 32)
            Bsite_extractor(extractor_threshold).extract_bsites(prot, scores)
            sites = [dict(center=[float(x) for x in b.center], score=float(b.score)) for b in prot.binding_sites]
            print("RESULT %s %s" % (pdb_id, json.dumps(sites)), flush=True)
        except Exception as ex:                      # noqa: BLE001 -- one structure must not cost the list
            print("FAILED %s %s: %s" % (pdb_id, type(ex).__name__, str(ex).splitlines()[0][:160]), flush=True)
            traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    main()
