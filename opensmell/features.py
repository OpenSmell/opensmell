"""Sensor-agnostic feature-extraction framework.

The top-level `opensmell.features` routes by the sensor family declared in the
manifest (`run_processor`) and re-exports the MOX framework-feature functions so
existing callers of the pre-v3 layout keep working unchanged.
"""

from __future__ import annotations

from .mox.features import (
    N_CHANNELS,
    compute_channel_absolute,
    compute_channel_device_agnostic,
    compute_channel_hardware,
    compute_channel_health,
    compute_channel_temporal,
    compute_multi_exp_decay,
    compute_saturation_index,
    extract_all_framework_features,
    feature_names,
    process_mox,
)

__all__ = [
    "N_CHANNELS",
    "compute_channel_absolute",
    "compute_channel_device_agnostic",
    "compute_channel_hardware",
    "compute_channel_health",
    "compute_channel_temporal",
    "compute_multi_exp_decay",
    "compute_saturation_index",
    "extract_all_framework_features",
    "feature_names",
    "process_mox",
    "run_processor",
]


def run_processor(file):
    """Dispatch feature extraction by sensor type (web runProcessor parity)."""
    sensor_type = file.manifest.sensor.sensor_type
    if sensor_type == "mox":
        return process_mox(file)
    if sensor_type in ("miris", "electrochemical"):
        return {"sensor_type": sensor_type, "normalized": file.data}
    return {"sensor_type": "other"}
