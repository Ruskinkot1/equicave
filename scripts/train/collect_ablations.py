#!/usr/bin/env python
"""Collect ablation runs into one table: mean +- seed sd per metric and the paired difference against `full`.

Reads every runs/**/model_card.json under --runs, groups by ablation, and writes a markdown table plus JSON.
Usage: python scripts/train/collect_ablations.py --runs runs/training/ablations --out docs/results/ablations.md
"""
import argparse, json, pathlib
import numpy as np

KEYS = [("net_sites.top1", "site top-1"), ("net_sites.topN2", "site top-(N+2)"), ("occ_ap", "probe occupancy AP"),
        ("res_ap", "residue AP"), ("hot_mean_ap", "hotspot mean AP"), ("prop_mean_auroc", "property mean AUROC")]


def dig(d, path):
    for p in path.split("."):
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--out", default="docs/results/ablations.md")
    a = ap.parse_args()
    runs = {}
    for f in pathlib.Path(a.runs).rglob("model_card.json"):
        c = json.loads(f.read_text()); m = c.get("metrics") or {}
        if m.get("hot_ap_per_class"):
            m["hot_mean_ap"] = float(np.nanmean(m["hot_ap_per_class"]))
        if m.get("prop_auroc_per_class"):
            m["prop_mean_auroc"] = float(np.nanmean(m["prop_auroc_per_class"]))
        runs.setdefault(c.get("ablation", "?"), []).append(m)
    if not runs:
        raise SystemExit(f"no model_card.json under {a.runs}")
    rows = []
    for abl, ms in sorted(runs.items(), key=lambda kv: (kv[0] != "full", kv[0])):
        row = dict(ablation=abl, seeds=len(ms))
        for path, name in KEYS:
            v = [dig(m, path) for m in ms]
            v = [x for x in v if x is not None]
            row[name] = f"{np.mean(v):.3f} ± {np.std(v):.3f}" if v else "not run"
            row[f"_{name}"] = float(np.mean(v)) if v else float("nan")
        rows.append(row)
    base = next((r for r in rows if r["ablation"] == "full"), None)
    lines = ["# EquiCave-Net ablations", "", "Mean ± seed standard deviation on the validation fold. "
             "`full - variant` is the drop caused by removing the component; a positive drop means the component helps.", "",
             "| ablation | seeds | " + " | ".join(n for _, n in KEYS) + " | drop in site top-1 |", "|" + "---|" * (len(KEYS) + 3)]
    for r in rows:
        d = "" if base is None or r is base else f"{base['_site top-1'] - r['_site top-1']:+.3f}"
        lines.append(f"| {r['ablation']} | {r['seeds']} | " + " | ".join(str(r[n]) for _, n in KEYS) + f" | {d} |")
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text("\n".join(lines) + "\n")
    pathlib.Path(str(a.out).replace(".md", ".json")).write_text(json.dumps({k: v for k, v in runs.items()}, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
