"""Shared test fixtures for the .osmell data model."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from opensmell.types import (
    BaselineDescriptor,
    ChannelDescriptor,
    OsmellFile,
    OsmellManifest,
    SensorDescriptor,
    SessionDescriptor,
)

CHANNELS = ["VOC", "Alcohol", "LPG", "CO", "NO2", "C2H5OH"]


def make_manifest(
    role="exposure",
    label="cinnamon",
    baseline_source="auto",
    channels=None,
    sampling_rate_hz=None,
):
    return OsmellManifest(
        osmell={"formatVersion": "1.0.0"},
        sensor=SensorDescriptor(
            sensor_type="mox",
            channels=[ChannelDescriptor(id=c, unit="adc") for c in (channels or CHANNELS)],
            time_column="timestamp_ms",
            sampling_rate_hz=sampling_rate_hz,
        ),
        session=SessionDescriptor(
            role=role,
            label=label,
            group_id="group-1",
            recorded_at="2026-01-15T10:00:00",
            duration_ms=60000,
        ),
        baseline=BaselineDescriptor(source=baseline_source, r0_samples=15),
    )


def _envelope(i: int, n: int) -> float:
    """Clean-air lead-in, rise, plateau, decay back to baseline (recovery)."""
    clean = 0.15 * n
    rise_end = 0.45 * n
    plateau_end = 0.55 * n
    decay_end = 0.90 * n
    if i < clean:
        return 0.0
    if i < rise_end:
        return (i - clean) / (rise_end - clean)
    if i < plateau_end:
        return 1.0
    if i < decay_end:
        return 1.0 - (i - plateau_end) / (decay_end - plateau_end)
    return 0.0


def make_file(
    n=600,
    role="exposure",
    channels=None,
    drift=True,
    dead_channels=None,
    sampling_rate_hz=None,
):
    """Synthetic 10 Hz exposure with a recovery tail (spec-meaningful R factor)."""
    channels = channels or CHANNELS
    dead_channels = dead_channels or []
    time = [i * 100 for i in range(n)]
    data = {}
    for idx, c in enumerate(channels):
        if c in dead_channels:
            data[c] = [100.0] * n
            continue
        base = 100.0 + idx * 30
        amp = 600.0 if drift else 0.0
        data[c] = [base + amp * _envelope(i, n) for i in range(n)]
    manifest = make_manifest(role=role, channels=channels, sampling_rate_hz=sampling_rate_hz)
    return OsmellFile(manifest=manifest, time=time, data=data)
