"""Reading and writing the derived tables of data/processed, gzip by default.

Candidate tables are tens of megabytes as plain CSV, which is wasteful in git, so they are written as `.csv.gz`
(pandas compresses by extension). `read_table` accepts either form, so a checkout with the old uncompressed files
keeps working.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def table_path(ds, name: str, compressed: bool = True) -> Path:
    """Where a derived table lives: <ds>/<name>.csv.gz by default."""
    return Path(ds) / (f"{name}.csv.gz" if compressed else f"{name}.csv")


def find_table(ds, name: str) -> Path:
    """The existing file for a table, compressed or not. Raises FileNotFoundError with both names tried."""
    for p in (table_path(ds, name), table_path(ds, name, compressed=False)):
        if p.exists():
            return p
    raise FileNotFoundError(f"neither {table_path(ds, name).name} nor {table_path(ds, name, False).name} in {ds}")


def read_table(ds, name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(find_table(ds, name), **kw)


def write_table(df: pd.DataFrame, ds, name: str, compressed: bool = True) -> Path:
    p = table_path(ds, name, compressed)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


def write_table_from_parts(parts, ds, name: str, compressed: bool = True, dtype=None) -> tuple[Path, int]:
    """Write one table from an iterable of CSV part files without holding them all in memory at once.

    The per-point table is about 1700 rows per structure, so the largest manifest produces close to twenty million
    rows; concatenating those into a single DataFrame costs several gigabytes and would fail on a modest machine
    after hours of work. Each part is read, appended and released, so peak memory is one part. Returns the path and
    the total row count.
    """
    p = table_path(ds, name, compressed)
    p.parent.mkdir(parents=True, exist_ok=True)
    total, first = 0, True
    for f in parts:
        chunk = pd.read_csv(f, dtype=dtype)
        if not len(chunk):
            continue
        chunk.to_csv(p, index=False, header=first, mode="w" if first else "a",
                     compression="gzip" if compressed else None)
        total += len(chunk); first = False
        del chunk
    return p, total
