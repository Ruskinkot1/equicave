"""Entry point: python -m training <task> [--out DIR] [--set key=value ...]."""
import argparse
import importlib
import sys

TASKS = {
    "pockets-net": "training.pockets.net_task",
    "pockets-labels": "training.pockets.labels_task",
    "pockets-ranker": "training.pockets.ranker_task",
}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m training")
    ap.add_argument("task", choices=list(TASKS) + ["list"])
    ap.add_argument("--out", default="runs/training")
    ap.add_argument("--config", default="")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--set", nargs="*", default=[], help="override config keys: key=value")
    a = ap.parse_args(argv)
    if a.task == "list":
        for k, v in TASKS.items():
            print(f"{k:16s} {importlib.import_module(v).__doc__.strip().splitlines()[0]}")
        return 0
    return importlib.import_module(TASKS[a.task]).run(a)


if __name__ == "__main__":
    sys.exit(main())
