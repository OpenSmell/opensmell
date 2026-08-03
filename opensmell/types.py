"""Sensor-agnostic data model for the .osmell format.

Mirrors `osmograph-web/lib/osmell/types.ts` 1:1. All JSON serialization uses the
camelCase field names defined in the OpenSmell format spec so that `.osmell`
files round-trip between the Python and TypeScript implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

OSMELL_FORMAT_VERSION = "1.0.0"

TIME_COLUMNS = ("timestamp_ms", "elapsed_ms")
SENSOR_TYPES = ("mox", "miris", "electrochemical", "other", "unknown")
SESSION_ROLES = ("baseline", "exposure", "single")
BASELINE_SOURCES = ("explicit", "auto", "none")

# Quality constants (shared with web lib/osmell/types.ts).
DEFAULT_ADC_MAX = 4095
DEFAULT_R0_SAMPLES = 15
DEAD_CV_THRESHOLD = 0.001
NOISE_CV_LIMIT = 0.05
SNR_TARGET = 10
FULL_SCORE_DURATION_S = 60
MIN_SPAN_FRACTION = 0.1
GAP_TOLERANCE = 0.1


def _camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    head, *rest = name.split("_")
    return head + "".join(p.capitalize() for p in rest)


@dataclass
class ChannelDescriptor:
    id: str
    unit: str
    target: Optional[str] = None

    def to_dict(self) -> dict:
        return {"id": self.id, "unit": self.unit, **({"target": self.target} if self.target else {})}

    @classmethod
    def from_dict(cls, d: dict) -> "ChannelDescriptor":
        return cls(id=d["id"], unit=d.get("unit", ""), target=d.get("target"))


@dataclass
class DeviceDescriptor:
    model: Optional[str] = None
    serial: Optional[str] = None
    firmware: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in {
            "model": self.model, "serial": self.serial, "firmware": self.firmware,
        }.items() if v is not None}

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> Optional["DeviceDescriptor"]:
        if not d:
            return None
        return cls(model=d.get("model"), serial=d.get("serial"), firmware=d.get("firmware"))


@dataclass
class SensorDescriptor:
    sensor_type: str = "mox"
    channels: List[ChannelDescriptor] = field(default_factory=list)
    device: Optional[DeviceDescriptor] = None
    sampling_rate_hz: Optional[float] = None
    adc_bits: Optional[int] = None
    adc_max: Optional[int] = None
    time_column: str = "timestamp_ms"

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "sensorType": self.sensor_type,
            "channels": [c.to_dict() for c in self.channels],
        }
        if self.device:
            d["device"] = self.device.to_dict()
        for k, v in {
            "samplingRateHz": self.sampling_rate_hz,
            "adcBits": self.adc_bits,
            "adcMax": self.adc_max,
        }.items():
            if v is not None:
                d[k] = v
        d["timeColumn"] = self.time_column
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SensorDescriptor":
        return cls(
            sensor_type=d.get("sensorType", "mox"),
            channels=[ChannelDescriptor.from_dict(c) for c in d.get("channels", [])],
            device=DeviceDescriptor.from_dict(d.get("device")),
            sampling_rate_hz=d.get("samplingRateHz"),
            adc_bits=d.get("adcBits"),
            adc_max=d.get("adcMax"),
            time_column=d.get("timeColumn", "timestamp_ms"),
        )


@dataclass
class SessionDescriptor:
    role: str = "single"
    label: Optional[str] = None
    group_id: Optional[str] = None
    recorded_at: Optional[str] = None
    duration_ms: Optional[int] = None
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in {
            "role": self.role,
            "label": self.label,
            "groupId": self.group_id,
            "recordedAt": self.recorded_at,
            "durationMs": self.duration_ms,
            "notes": self.notes,
        }.items() if v is not None}

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> Optional["SessionDescriptor"]:
        if not d:
            return None
        return cls(
            role=d.get("role", "single"),
            label=d.get("label"),
            group_id=d.get("groupId"),
            recorded_at=d.get("recordedAt"),
            duration_ms=d.get("durationMs"),
            notes=d.get("notes"),
        )


@dataclass
class BaselineDescriptor:
    source: str = "none"
    file: Optional[str] = None
    r0_samples: Optional[int] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in {
            "source": self.source,
            "file": self.file,
            "r0Samples": self.r0_samples,
        }.items() if v is not None}

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> Optional["BaselineDescriptor"]:
        if not d:
            return None
        return cls(source=d.get("source", "none"), file=d.get("file"), r0_samples=d.get("r0Samples"))


@dataclass
class OsmellManifest:
    osmell: dict = field(default_factory=lambda: {"formatVersion": OSMELL_FORMAT_VERSION})
    sensor: SensorDescriptor = field(default_factory=SensorDescriptor)
    session: SessionDescriptor = field(default_factory=SessionDescriptor)
    baseline: Optional[BaselineDescriptor] = None
    software: Optional[dict] = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"osmell": self.osmell, "sensor": self.sensor.to_dict()}
        if self.session:
            d["session"] = self.session.to_dict()
        if self.baseline:
            d["baseline"] = self.baseline.to_dict()
        if self.software:
            d["software"] = self.software
        d.update(self.extra)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "OsmellManifest":
        known = {"osmell", "sensor", "session", "baseline", "software"}
        return cls(
            osmell=d.get("osmell", {"formatVersion": OSMELL_FORMAT_VERSION}),
            sensor=SensorDescriptor.from_dict(d.get("sensor", {})),
            session=SessionDescriptor.from_dict(d.get("session")),
            baseline=BaselineDescriptor.from_dict(d.get("baseline")),
            software=d.get("software"),
            extra={k: v for k, v in d.items() if k not in known},
        )


@dataclass
class SessionEvent:
    label: str
    start_ms: int
    end_ms: Optional[int] = None
    note: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in {
            "label": self.label,
            "startMs": self.start_ms,
            "endMs": self.end_ms,
            "note": self.note,
        }.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "SessionEvent":
        return cls(
            label=d["label"],
            start_ms=d["startMs"],
            end_ms=d.get("endMs"),
            note=d.get("note"),
        )


@dataclass
class OsmellFile:
    manifest: OsmellManifest
    time: List[float]
    data: dict[str, List[float]]
    events: Optional[List[SessionEvent]] = None


@dataclass
class ParsedSample:
    time: float
    values: dict[str, float]


@dataclass
class ChannelStats:
    id: str
    min: float
    max: float
    mean: float
    std: float
    r0: float
    cv: float
    dead: bool
    span: float
    clipped: int = 0
    non_finite: int = 0


@dataclass
class QualityFlags:
    dead_sensors: List[str] = field(default_factory=list)
    unsorted_rows: bool = False
    non_finite_samples: int = 0
    used_default_adc_max: bool = False
    used_median_sampling_rate: bool = False
    no_baseline: bool = False
    empty_recording: bool = False


@dataclass
class SubScore:
    value: Optional[float]
    reason: str = "ok"


@dataclass
class QualityReport:
    format: str
    version: str
    computed_at: str
    total: Optional[float]
    badge: str
    subscores: dict[str, SubScore]
    flags: QualityFlags
    reasons: dict[str, str]
    notes: List[str]
