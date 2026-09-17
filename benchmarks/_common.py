"""Shared helpers for the benchmark scripts (path setup + JSON recording)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / "opensmell", REPO_ROOT / "e-nose-evals"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

REPORTS = REPO_ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


def record_benchmark(name: str, results: dict) -> Path:
    """Write a benchmark's results dict to reports/ as JSON.

    When the smellbench package is importable the report is wrapped with
    its claim declaration and package versions (see smellbench.envelope),
    so a score can never travel without its regimen.  If the import fails the
    raw results are written unchanged.
    """
    try:
        from smellbench.envelope import wrap
        results = wrap(name, results)
    except Exception:  # noqa: BLE001 - standalone use must keep working
        pass
    path = REPORTS / f"{name}.json"
    path.write_text(json.dumps(results, indent=2, sort_keys=True, default=str))
    return path


def fmt(x, nd=4) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)