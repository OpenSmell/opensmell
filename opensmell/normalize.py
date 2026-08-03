"""Sensor-agnostic normalization interface.

Top-level `opensmell.normalize` defines the shared statistical helpers used
across sensor families plus the interface contract every sensor submodule
(e.g. `opensmell.mox.normalize`) implements: given a channel's raw series and
baseline information, produce an R0/offset and a normalized series.

Mirrors the statistical helpers in `osmograph-web/lib/osmell/normalize.ts`.
"""

from __future__ import annotations

import math
from statistics import median as _py_median
from typing import List


def median(values: List[float]) -> float:
    if not values:
        return math.nan
    return _py_median(values)


def mean(values: List[float]) -> float:
    if not values:
        return math.nan
    return sum(values) / len(values)


def std(values: List[float]) -> float:
    """Population standard deviation (ddof=0), matching the TS implementation."""
    if not values:
        return math.nan
    m = mean(values)
    variance = sum((v - m) ** 2 for v in values) / len(values)
    return math.sqrt(variance)
