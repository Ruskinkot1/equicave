"""pockets-labels: build property and hotspot labels for every manifest structure and report their statistics.

Writes data/processed/labels_sites.csv (one row per ligand site: pdb, site index, centre, n ligand atoms, property
vector) and data/processed/labels_summary.json (class prevalence, hotspot point prevalence per class), using the
network cache when it exists (same probes as training) or featurising on the fly otherwise.
Usage: python -m training pockets-labels [--set data.limit=200]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from equicave import labels as LB
from training.common.config import load_config
from training.pockets import data as D
from training.pockets.net_task import DEFAULTS


def run(a) -> int:
    cfg = load_config(a.config or DEFAULTS, a.set); dc = cfg["data"]
    cache = Path(dc["cache_dir"])
    ids = D.build_cache(Path(dc["manifest"]), Path(dc["pdb_dir"]), cache, None if cfg.get("no_esm") else dc.get("esm"), dc.get("limit", 0), dc["n_probe"], dc["n_surf"], "cpu")
    rows, hot, occ = [], [], []
    for i in ids:
        d = D.load(cache / f"{i}.npz")
        for s, (c, y) in enumerate(zip(d["site_centers"], d["y_prop"])):
            rows.append(dict(pdb=i, site=s, cluster30=str(d["cluster30"]), fold=int(d["fold"]), center=";".join(f"{x:.2f}" for x in c),
                             **{k: int(v) for k, v in zip(LB.PROPERTY_CLASSES, y)}))
        hot.append(d["y_hot"]); occ.append(d["y_occ"])
    df = pd.DataFrame(rows); out = Path("data/processed"); df.to_csv(out / "labels_sites.csv", index=False)
    H = np.concatenate(hot); O = np.concatenate(occ)
    summ = dict(n_structures=len(ids), n_sites=len(df), property_prevalence={k: float(df[k].mean()) for k in LB.PROPERTY_CLASSES},
                hotspot_point_prevalence=dict(zip(LB.HOTSPOT_CLASSES, H.mean(0).round(4).tolist())), occupancy_prevalence=float(O.mean()),
                probes_per_structure=float(len(O) / max(1, len(ids))))
    (out / "labels_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    return 0
