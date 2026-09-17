"""Filesystem locations for the ruler, overridable for external rigs.

An outside lab points OPENSMELL_BENCH_REPORTS (and optionally
OPENSMELL_BENCH_DIR) at its own tree and the same verbs operate there.
"""

from __future__ import annotations

import os
from pathlib import Path


def _package_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[2] if len(here.parents) > 2 else Path.cwd()


def reports_dir(explicit=None) -> Path:
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get("OPENSMELL_BENCH_REPORTS")
    if env:
        return Path(env)
    reports = _package_root() / "reports"
    return reports if reports.is_dir() else Path.cwd() / "reports"


def benchmarks_dir(explicit=None) -> Path:
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get("OPENSMELL_BENCH_DIR")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    return here.parents[1] if len(here.parents) > 1 else Path.cwd()