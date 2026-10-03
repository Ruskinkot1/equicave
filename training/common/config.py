"""Config loading: YAML defaults + `--set key=value` overrides; seeds; device selection; run directories."""
from __future__ import annotations

import json
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
    (d / "model_card.json").write_text(json.dumps(card, indent=1, default=str))
