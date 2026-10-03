"""pockets-ranker: LightGBM LambdaRank over native candidates (wrapper of scripts/train/train_ranker.py).

Usage: python -m training pockets-ranker [--set tag=native seeds=5 ablate=true model=models/ranker_native.txt extra=net]
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from training.common.config import load_config


def run(a) -> int:
    cfg = load_config("", a.set)
    cmd = [sys.executable, str(Path(__file__).resolve().parents[2] / "scripts/train/train_ranker.py"), "--tag", str(cfg.get("tag", "native")),
           "--seeds", str(cfg.get("seeds", 5))]
    if cfg.get("ablate"):
        cmd.append("--ablate")
    if cfg.get("model"):
        cmd += ["--model", str(cfg["model"])]
    if cfg.get("extra"):
        cmd += ["--features-extra", str(cfg["extra"])]
    return subprocess.call(cmd)
