"""Progress reporting for the long loops of the pipeline: downloads, featurisation, evaluation.

At full scale these loops run for hours — 13k structures to download, 11k to featurise — and a loop that prints
nothing is indistinguishable from a hung one. `track()` wraps an iterator so every such loop says where it is and
what is left.

Two renderings, chosen by where the output goes rather than by a flag:

* a terminal gets a `tqdm` bar, redrawn in place;
* anything else (the pipeline pipes its output to a log file) gets one plain line every `LOG_SECONDS`, because a bar
  redrawn into a file collapses into one unreadable line and a 17-hour run would write thousands of them.

`tqdm` is a soft dependency: without it the plain lines are used in a terminal too, so the package still works.
Set `EQUICAVE_PROGRESS=bar` to force the bar (e.g. under `tee`, where stderr may not look like a terminal), `plain`
to force the lines, or `off` to silence progress entirely.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Iterable, Iterator

LOG_SECONDS = 30.0


def _mode() -> str:
    """'bar', 'plain' or 'off'."""
    want = os.environ.get("EQUICAVE_PROGRESS", "").strip().lower()
    if want in ("bar", "plain", "off"):
        if want != "bar":
            return want
        return "bar" if _have_tqdm() else "plain"
    if want in ("0", "none", "no", "false"):
        return "off"
    try:
        tty = sys.stderr.isatty()
    except Exception:                                  # noqa: BLE001  (a closed or exotic stderr)
        tty = False
    return "bar" if tty and _have_tqdm() else "plain"


def _have_tqdm() -> bool:
    try:
        import tqdm  # noqa: F401
    except Exception:                                  # noqa: BLE001
        return False
    return True


def _hms(sec: float) -> str:
    sec = int(max(sec, 0))
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


class Bar:
    """A counter to `update()` as work completes. Use as a context manager, or call `close()`.

    `postfix` on `update()` carries the loop's own running numbers (a hit rate, a row count) so a caller does not
    need a second print.
    """

    def __init__(self, desc: str, total: int | None = None, unit: str = "it"):
        self.desc, self.total, self.unit, self.n = desc, total, unit, 0
        self._mode = _mode()
        self._bar = None
        self._post = ""
        self._t0 = self._last = time.monotonic()
        if self._mode == "bar":
            from tqdm.auto import tqdm
            self._bar = tqdm(total=total, desc=desc, unit=unit, file=sys.stderr, smoothing=0.05)
        elif self._mode == "plain" and total:
            print(f"{self.desc}: 0/{total} {self.unit}", file=sys.stderr, flush=True)

    def update(self, n: int = 1, postfix: str = "") -> None:
        self.n += n
        if postfix:
            self._post = postfix
        if self._bar is not None:
            if postfix:
                self._bar.set_postfix_str(postfix, refresh=False)
            self._bar.update(n)
            return
        if self._mode == "off":
            return
        now = time.monotonic()
        if now - self._last < LOG_SECONDS and self.n != self.total:
            return
        self._last = now
        self._line(now)

    def _line(self, now: float) -> None:
        rate = self.n / max(now - self._t0, 1e-9)
        out = f"{self.desc}: {self.n}"
        if self.total:
            out += f"/{self.total} ({100.0 * self.n / self.total:.0f}%)"
        out += f" {self.unit} at {rate:.1f}/s" if rate >= 1 else f" {self.unit} at {60 * rate:.1f}/min"
        if self.total and self.n < self.total and rate > 0:
            out += f", eta {_hms((self.total - self.n) / rate)}"
        if self._post:
            out += f", {self._post}"
        print(out, file=sys.stderr, flush=True)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None
        elif self._mode != "off" and self.n:
            now = time.monotonic()
            if now - self._last > 1.0 or self.n != self.total:
                self._line(now)

    def __enter__(self) -> "Bar":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def track(it: Iterable, desc: str, total: int | None = None, unit: str = "it") -> Iterator:
    """Yield from `it`, reporting progress. `total` defaults to `len(it)` when the iterable has one.

    The count advances after each item is consumed, so wrapping `ThreadPoolExecutor.map` reports completed work
    rather than submitted work.
    """
    if total is None:
        try:
            total = len(it)                            # type: ignore[arg-type]
        except TypeError:
            total = None
    bar = Bar(desc, total, unit)
    try:
        for x in it:
            yield x
            bar.update(1)
    finally:
        bar.close()
