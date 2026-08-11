"""Sensor-agnostic quality-scoring framework.

The top-level `opensmell.quality` dispatches by sensor family declared in the
manifest to the sensor-specific implementation (e.g. `opensmell.mox.quality`).
Sensor families without a registered scorer raise `NotImplementedError` so the
framework contract stays explicit.
"""

from __future__ import annotations

from typing import List

from .mox.quality import compute_quality_mox
from .types import OsmellFile, QualityReport


def compute_quality(
    file: OsmellFile,
    sample_count: int,
    guess_sampling_rate_hz: float,
    unsorted: bool = False,
    non_finite: int = 0,
) -> QualityReport:
    sensor_type = file.manifest.sensor.sensor_type
    if sensor_type == "mox":
        return compute_quality_mox(
            file,
            sample_count=sample_count,
            guess_sampling_rate_hz=guess_sampling_rate_hz,
            unsorted=unsorted,
            non_finite=non_finite,
        )
    if sensor_type == "unknown":
        # Adopt-don't-reject: unidentifiable arrays (e.g. MQ-series) still get
        # the device-agnostic score instead of failing ingestion.
        return compute_quality_mox(
            file,
            sample_count=sample_count,
            guess_sampling_rate_hz=guess_sampling_rate_hz,
            unsorted=unsorted,
            non_finite=non_finite,
        )
    if sensor_type in ("miris", "electrochemical"):
        raise NotImplementedError(
            f"No quality scorer registered for sensor type '{sensor_type}'. "
            "Implement opensmell.miris.quality or opensmell.electrochemical.quality."
        )
    raise NotImplementedError(f"Unknown sensor type '{sensor_type}'.")
