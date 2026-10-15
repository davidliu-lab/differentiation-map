"""Progress reporting: tqdm bars on terminals, periodic log lines otherwise (D30). All output goes to stderr."""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import timedelta
from typing import Iterator, Optional

from tqdm.auto import tqdm

log = logging.getLogger("rdmap")

MODES = ("auto", "bar", "log", "off")
LOG_INTERVAL_SECONDS = 60.0
_mode = "bar"


def setup(mode: Optional[str] = None) -> str:
    """Resolve the progress mode (argument > RDMAP_PROGRESS > auto) and configure logging."""
    global _mode
    mode = mode or os.environ.get("RDMAP_PROGRESS", "auto")
    if mode not in MODES:
        raise ValueError(f"Unknown progress mode '{mode}' (choose from {', '.join(MODES)}).")
    if mode == "auto":
        mode = "bar" if sys.stderr.isatty() else "log"
    _mode = mode
    if not log.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)
        log.propagate = False
    log.setLevel(logging.INFO)
    return mode


def mode() -> str:
    return _mode


def fmt_duration(seconds: float) -> str:
    return str(timedelta(seconds=int(round(seconds))))


def info(message: str) -> None:
    """Informational line; suppressed in `off` mode."""
    if _mode != "off":
        log.info(message)


def header(index: int, total: int, text: str) -> None:
    info(f"[{index}/{total}] {text}")


@contextmanager
def step(message: str) -> Iterator[None]:
    """A step without natural progress: start line, then end line with the elapsed time."""
    info(f"{message}…")
    start = time.time()
    yield
    info(f"{message}… done in {fmt_duration(time.time() - start)}")


class _NullBar:
    def update(self, n: int = 1) -> None:
        pass

    def set_postfix_str(self, s: str) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> "_NullBar":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class _LogBar(_NullBar):
    """Prints a tqdm-formatted status line at most every LOG_INTERVAL_SECONDS (for SLURM logs and pipes)."""

    def __init__(self, total: int, desc: str, unit: str):
        self.total, self.desc, self.unit = total, desc, unit
        self.n, self.postfix, self.emitted = 0, "", -1
        self.start = self.last = time.time()

    def _emit(self) -> None:
        now = time.time()
        meter = tqdm.format_meter(
            self.n, self.total, now - self.start, prefix=self.desc, unit=self.unit, unit_scale=True, ncols=0
        )
        log.info(meter + (f" {self.postfix}" if self.postfix else ""))
        self.last, self.emitted = now, self.n

    def update(self, n: int = 1) -> None:
        self.n += n
        if time.time() - self.last >= LOG_INTERVAL_SECONDS:
            self._emit()

    def set_postfix_str(self, s: str) -> None:
        self.postfix = s

    def close(self) -> None:
        if self.emitted != self.n:
            self._emit()


def bar(total: int, desc: str, unit: str = "tiles", inner: bool = False):
    """Overall bar (`inner=False`) or transient per-slide bar (`inner=True`)."""
    if _mode == "off" or (inner and _mode == "log"):
        return _NullBar()
    if _mode == "log":
        return _LogBar(total, desc, unit)
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        unit_scale=True,
        dynamic_ncols=True,
        leave=not inner,
        position=1 if inner else 0,
        file=sys.stderr,
    )


class StageSummary:
    """Counts processed/skipped/failed slides and prints one summary line at the end of a stage."""

    def __init__(self, name: str, skipped: int = 0):
        self.name = name
        self.processed = 0
        self.skipped = skipped
        self.failed: list[str] = []
        self.tiles = 0
        self.start = time.time()

    def report(self) -> None:
        elapsed = time.time() - self.start
        tiles_per_second = self.tiles / elapsed if elapsed > 0 else 0.0
        rate = f", {tiles_per_second:,.1f} tiles/s" if 0 < tiles_per_second < 10 else (
            f", {tiles_per_second:,.0f} tiles/s" if tiles_per_second else "")
        failed = f", {len(self.failed)} failed ({', '.join(self.failed)})" if self.failed else ""
        log.info(
            f"{self.name}: {self.processed} processed, {self.skipped} skipped{failed} "
            f"in {fmt_duration(elapsed)}{rate}"
        )
