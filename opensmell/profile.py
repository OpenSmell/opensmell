"""Inspect a recording before choosing a feature path.

The SDK's implemented feature framework is MOX/chemiresistive. This module
makes that boundary visible instead of letting every numeric array inherit the
MOX label by accident. It detects the sensor family from a manifest or from CSV
column names, resolves channel names against the offline sensor-model table, and
returns a :class:`SensorProfile`.

``extract_features``/``process`` accept recordings whose profile is
``"mox"`` or ``"mox-compatible"``. A recording whose profile is
``"unsupported"`` (MIRIS, electrochemical, other) raises
:class:`UnsupportedSensorTypeError` rather than silently running MOX math.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .constants import UnknownSensorError, normalize_sensor_name
from .csv import guess_sensor_type, parse_csv
from .ingest import IngestedSession
from .io import parse_osmell_file
from .types import OsmellFile


class UnsupportedSensorTypeError(NotImplementedError):
    """The requested sensor family has no feature/quality backend yet."""


@dataclass
class SensorProfile:
    """What the SDK thinks it was given, and which feature path fits.

    ``sensor_type`` is the manifest/declared family when one exists. For a raw
    CSV it is inferred from channel names. ``feature_family`` is deliberately
    narrower:

    - ``"mox"``: a known MOX/chemiresistive model or an explicit MOX manifest;
    - ``"mox-compatible"``: no model identified, but the array can use the
      signal-level MOX framework without claiming a specific chemistry;
    - ``"unsupported"``: MIRIS, electrochemical, or another family with no
      feature backend.
    """

    source: str
    sensor_type: str
    feature_family: str
    channel_ids: List[str]
    models: Dict[str, str] = field(default_factory=dict)
    unmodeled_channels: List[str] = field(default_factory=list)
    sampling_rate_hz: Optional[float] = None
    time_source: Optional[str] = None
    time_column: Optional[str] = None
    context_columns: List[str] = field(default_factory=list)
    inferred: bool = False
    warnings: List[str] = field(default_factory=list)

    @property
    def supported(self) -> bool:
        return self.feature_family in ("mox", "mox-compatible")

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "sensorType": self.sensor_type,
            "featureFamily": self.feature_family,
            "channelIds": list(self.channel_ids),
            "models": dict(self.models),
            "unmodeledChannels": list(self.unmodeled_channels),
            "samplingRateHz": self.sampling_rate_hz,
            "timeSource": self.time_source,
            "timeColumn": self.time_column,
            "contextColumns": list(self.context_columns),
            "inferred": self.inferred,
            "warnings": list(self.warnings),
        }


def _select_channel_ids(available: List[str], channels=None) -> List[str]:
    if channels is None:
        selected = list(available)
    elif isinstance(channels, Mapping):
        selected = list(channels.keys())
    elif isinstance(channels, Sequence) and not isinstance(channels, (str, bytes)):
        selected = list(channels)
    elif isinstance(channels, (str, bytes)):
        selected = [channels]
    else:
        raise TypeError(
            "channels must be a sequence of channel ids or a "
            "{source: output} mapping."
        )

    missing = [c for c in selected if c not in available]
    if missing:
        raise ValueError(
            f"Requested channel(s) not available: {missing}. "
            f"Available channels: {available}."
        )
    if not selected:
        raise ValueError("No channels selected.")
    return selected


def _match_models(channel_ids: List[str]):
    models: Dict[str, str] = {}
    unmodeled: List[str] = []
    for cid in channel_ids:
        try:
            models[cid] = normalize_sensor_name(cid)
        except UnknownSensorError:
            unmodeled.append(cid)
    return models, unmodeled


def _feature_family(sensor_type: str, models: Dict[str, str]) -> str:
    if sensor_type == "mox":
        return "mox"
    if sensor_type == "unknown":
        return "mox" if models else "mox-compatible"
    return "unsupported"


def _profile_from_file(file: OsmellFile, source: str, channels=None) -> SensorProfile:
    sensor = file.manifest.sensor
    available = [c.id for c in sensor.channels]
    selected = _select_channel_ids(available, channels)
    models, unmodeled = _match_models(selected)

    ingest_meta = (file.manifest.extra or {}).get("ingest", {})
    warnings = list(ingest_meta.get("warnings") or [])
    sensor_type = sensor.sensor_type or "unknown"
    inferred = False

    if sensor_type == "unknown" and models:
        sensor_type = "mox"
        inferred = True
        warnings.append(
            "Sensor type inferred as mox from recognized model name(s): "
            + ", ".join(f"{cid}={model}" for cid, model in models.items())
        )
    if sensor_type == "unknown":
        warnings.append(
            "No sensor model recognized; using the MOX-compatible signal "
            "framework. Absolute concentration remains nominal."
        )
    elif sensor_type == "mox" and not models:
        warnings.append(
            "Manifest declares mox but no channel matches a tabulated model; "
            "features are signal-level only."
        )

    return SensorProfile(
        source=source,
        sensor_type=sensor_type,
        feature_family=_feature_family(sensor_type, models),
        channel_ids=selected,
        models=models,
        unmodeled_channels=unmodeled,
        sampling_rate_hz=sensor.sampling_rate_hz,
        time_source=ingest_meta.get("timeSource") or (
            "column" if sensor.time_column else "unknown"
        ),
        time_column=sensor.time_column,
        context_columns=list(ingest_meta.get("contextColumns") or []),
        inferred=inferred,
        warnings=warnings,
    )


def _profile_from_csv(path: Path, channels=None) -> SensorProfile:
    text = path.read_text(encoding="utf-8", errors="replace")
    parsed = parse_csv(text)
    selected = _select_channel_ids(parsed.channel_ids, channels)
    models, unmodeled = _match_models(selected)

    sensor_type = guess_sensor_type(parsed.header)
    warnings = list(parsed.warnings)
    if sensor_type == "unknown" and models:
        sensor_type = "mox"
        warnings.append(
            "Sensor type inferred as mox from recognized model name(s): "
            + ", ".join(f"{cid}={model}" for cid, model in models.items())
        )
    if sensor_type == "unknown":
        warnings.append(
            "No sensor model recognized; using the MOX-compatible signal "
            "framework. Absolute concentration remains nominal."
        )

    return SensorProfile(
        source=str(path),
        sensor_type=sensor_type,
        feature_family=_feature_family(sensor_type, models),
        channel_ids=selected,
        models=models,
        unmodeled_channels=unmodeled,
        sampling_rate_hz=parsed.guess_sampling_rate_hz,
        time_source=parsed.time_source,
        time_column=parsed.time_column,
        context_columns=list(parsed.context_columns),
        inferred=True,
        warnings=warnings,
    )


def inspect_source(source: Any, channels=None) -> SensorProfile:
    """Report the detected sensor family and feature path for a recording.

    Accepts the same sources as ``extract_features``: a CSV/``.osmell`` path,
    an ``OsmellFile``, or an ``IngestedSession``. This function is read-only and
    does not normalize or extract features.
    """
    if isinstance(source, OsmellFile):
        return _profile_from_file(source, "<OsmellFile>", channels)
    if isinstance(source, IngestedSession):
        if not source.ok or source.file is None:
            detail = source.error or "; ".join(source.warnings) or "unknown error"
            raise ValueError(f"Ingest failed for {source.source}: {detail}")
        return _profile_from_file(source.file, source.source, channels)
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() == ".osmell":
            return _profile_from_file(
                parse_osmell_file(path), str(path), channels
            )
        return _profile_from_csv(path, channels)
    raise TypeError(
        "source must be a CSV/.osmell path, an OsmellFile, or an "
        f"IngestedSession; got {type(source).__name__}."
    )


__all__ = [
    "SensorProfile",
    "UnsupportedSensorTypeError",
    "inspect_source",
]
