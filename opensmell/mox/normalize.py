"""MOX normalization: R0 estimation and per-channel normalized series.

Mirrors `osmograph-web/lib/osmell/normalize.ts` 1:1. R0 is the median of the
explicit baseline channel when one is present (`baseline.source == "explicit"`);
otherwise auto-R0 falls back to the median of the first `r0Samples` of the
target channel (SmellNet-style session invariance without a dedicated baseline).
"""

from __future__ import annotations

from typing import List

from ..normalize import mean, std
from ..types import DEFAULT_R0_SAMPLES, ChannelStats, OsmellFile


def r0_from_samples(values: List[float], n: int = DEFAULT_R0_SAMPLES) -> float:
    window = values[:n]
    if not window:
        return float("nan")
    sorted_win = sorted(window)
    mid = len(sorted_win) // 2
    r0 = (sorted_win[mid - 1] + sorted_win[mid]) / 2 if len(sorted_win) % 2 == 0 else sorted_win[mid]
    if r0 > 0:
        return r0
    positive = [v for v in window if v > 0]
    return mean(positive) if positive else 1.0


def baseline_for_channel(
    file: OsmellFile,
    channel_id: str,
    target_values: List[float],
) -> tuple[float, List[float], float]:
    """Return (r0, window_values, cv) for a channel.

    Matches the web `baselineForChannel`: with an explicit baseline the whole
    baseline channel is used; otherwise the first `r0Samples` of the target.
    """
    baseline = file.manifest.baseline
    source = baseline.source if baseline else "none"
    r0_samples = (baseline.r0_samples if baseline and baseline.r0_samples else DEFAULT_R0_SAMPLES)

    if source == "explicit":
        b = file.data.get(channel_id, [])
        r0 = r0_from_samples(b, len(b))
        cv = std(b) / r0 if r0 else float("inf")
        return r0, b, cv

    valid = [v for v in target_values[:r0_samples] if _is_finite(v)]
    r0 = r0_from_samples(valid, r0_samples)
    cv = std(valid) / r0 if r0 else float("inf")
    return r0, valid, cv


def normalized_series(values: List[float], r0: float) -> List[float]:
    if not (_is_finite(r0) and r0 > 0):
        return [float("nan")] * len(values)
    return [(v - r0) / r0 for v in values]


def channel_stats(values: List[float], r0: float) -> ChannelStats:
    finite = [v for v in values if _is_finite(v)]
    non_finite = len(values) - len(finite)
    m = mean(finite)
    sd = std(finite)
    cv = sd / r0 if r0 > 0 else float("inf")
    lo = min(finite) if finite else float("nan")
    hi = max(finite) if finite else float("nan")
    return ChannelStats(
        id="",
        min=lo,
        max=hi,
        mean=m,
        std=sd,
        r0=r0,
        cv=cv,
        dead=cv < 0.001,
        span=(hi - lo) if finite else float("nan"),
        clipped=0,
        non_finite=non_finite,
    )


def _is_finite(v: float) -> bool:
    return v == v and v not in (float("inf"), float("-inf"))
