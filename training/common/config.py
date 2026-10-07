"""Config loading: YAML defaults + `--set key=value` overrides; seeds; device selection; run directories."""
from __future__ import annotations

import datetime
import json
import subprocess
import os
import random
from pathlib import Path

import numpy as np


def load_config(path: str | Path, overrides: list[str]) -> dict:
    import yaml
    cfg = yaml.safe_load(Path(path).read_text()) if path and Path(path).exists() else {}
    for kv in overrides:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        d = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    return cfg


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def pick_device(name: str = "auto"):
    import torch
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_dir(base: str | Path, name: str) -> Path:
    d = Path(base) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_model_card(d: Path, card: dict) -> None:
    """The card, stamped with when it was written and which commit produced it.

    A metrics.json with no date is only identifiable by the directory it sits in, which is exactly what gets lost
    when runs are copied off a machine or zipped up. Both fields go in the file itself.
    """
    card = dict(card, written_at=datetime.datetime.now().isoformat(timespec="seconds"), commit=git_commit())
    (d / "model_card.json").write_text(json.dumps(card, indent=1, default=str))


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=5).stdout.strip() or "unknown"
    except Exception:            # noqa: BLE001 -- a run outside a checkout still has to write its card
        return "unknown"
