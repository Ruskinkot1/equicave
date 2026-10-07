"""Entry point: python -m training <task> [--out DIR] [--set key=value ...]."""
import argparse
import datetime
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
    # A bare default would let two runs write into one directory and leave no way to tell which produced a
    # metrics.json. `now` stamps it with the launch time; an explicit --out still overrides, which is what the
    # Makefile and the ablation grid pass.
    ap.add_argument("--out", default="now",
                    help="where the run writes; the default 'now' means runs/training/<YYYYmmdd-HHMMSS>")
    ap.add_argument("--config", default="")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--set", nargs="*", default=[], help="override config keys: key=value")
    a = ap.parse_args(argv)
    if a.out == "now":
        a.out = f"runs/training/{datetime.datetime.now():%Y%m%d-%H%M%S}"
        print(f"output directory: {a.out}", flush=True)
    if a.task == "list":
        for k, v in TASKS.items():
            print(f"{k:16s} {importlib.import_module(v).__doc__.strip().splitlines()[0]}")
        return 0
    return importlib.import_module(TASKS[a.task]).run(a)


if __name__ == "__main__":
    sys.exit(main())
