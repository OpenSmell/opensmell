"""Sensor-agnostic data model for the .osmell format.

Mirrors `osmograph-web/lib/osmell/types.ts` 1:1. All JSON serialization uses the
camelCase field names defined in the OpenSmell format spec so that `.osmell`
files round-trip between the Python and TypeScript implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

OSMELL_FORMAT_VERSION = "1.0.0"

TIME_COLUMNS = ("timestamp_ms", "elapsed_ms")
# Broader set of accepted time-column names for tolerant CSV import. Column
# names are matched case-insensitively. `synthetic_index` is written by the
# converter when an imported CSV had no time column, so the resulting .osmell
# round-trips with an explicit, self-describing timing column.
TIME_COLUMN_ALIASES = (
    "timestamp_ms", "elapsed_ms", "timestamp", "elapsed",
    "time_ms", "time_s", "time", "synthetic_index",
)
# Timing assumed for CSVs with no time column. Kept visible and honest: the
# manifest records `timeSource: "synthetic"` and the quality report surfaces a
# note so no one mistakes synthesized timing for measured timing.
DEFAULT_SYNTHETIC_RATE_HZ = 10.0
# Environmental/context columns are detected and preserved (never scored as
# sensor channels, so they cannot corrupt feature or quality statistics).
CONTEXT_COLUMN_HINTS = (
    "temperature", "pressure", "humidity", "gas_res", "resistance", "altitude",
)
SENSOR_TYPES = ("mox", "miris", "electrochemical", "other", "unknown")
SESSION_ROLES = ("baseline", "exposure", "single")
BASELINE_SOURCES = ("explicit", "auto", "none")

# Quality constants (shared with web lib/osmell/types.ts).
DEFAULT_ADC_MAX = 4095
DEAD_CV_THRESHOLD = 0.001
NOISE_CV_LIMIT = 0.05
SNR_TARGET = 10
FULL_SCORE_DURATION_S = 60
MIN_SPAN_FRACTION = 0.1
GAP_TOLERANCE = 0.1

# --- R0 baseline window (SAMPLING_CONTRACT.md, "The R0 window contract") ---
#
# A declared window (`BaselineDescriptor.r0_samples`, a preset's
# `baseline.r0_samples`, or an explicit `r0_samples` argument) always wins and is
# used verbatim: whoever declares a window owns the duration-to-count conversion
# the contract requires (`round(duration_s * sr)`, see
# `opensmell.presets.BaselineSpec.resolve_r0_samples`).
#
# `None` (or `0`) here means "no window declared", not "zero samples": reduce the
# window with the contract default below. Rust carries the same meaning in
# `R0_WINDOW_DEFAULT = 0` (it has no `Option<usize>` sentinel to spare), and JS in
# `DEFAULT_R0_SAMPLES = undefined`, so all three SDKs accept
# `None`/`undefined`/`0` interchangeably and none of them treats a non-positive
# window as meaningful.
DEFAULT_R0_SAMPLES: Optional[int] = None
# Fraction of the recording the baseline window spans when nothing is declared.
# The same fraction `HARDWARE.md` (`cutoff = sample_count * 0.15`) and
# `data-commons/docs/wire-protocol.md` ("median of first 15%") specify.
R0_WINDOW_FRACTION = 0.15
# Floor: below ~5 samples the median is one or two readings and a single ADC LSB
# moves R0 by 10-20%. Ceiling: on a long recording an unbounded 15% would swallow
# the onset, so the baseline must stay inside the leading plateau.
R0_WINDOW_MIN_SAMPLES = 5
R0_WINDOW_MAX_SAMPLES = 30


def r0_window_samples(n_samples: int, declared: Optional[int] = None) -> int:
    """Number of leading samples forming the R0 baseline window.

    ``declared`` is a caller/manifest-declared window and is returned verbatim.
    ``None`` (and ``0``, which is not a meaningful window) means nothing was
    declared and the contract default applies:
    ``clamp(floor(0.15 * n_samples), 5, 30)``.

    **The default window is cadence-independent**; a fixed sample count is not.
    ``n_samples`` grows with the rate, so in the fraction region ``0.15 *
    n_samples`` spans ``0.15 * T`` seconds of recording whether it was sampled at
    1, 2, 10 or 100 Hz. The superseded fixed 15-sample default spanned 1.5 s at
    10 Hz and 15 s at 1 Hz, a 100x rescale across cadence. The clamps are the
    documented exception and are themselves sample counts, so they are
    cadence-*dependent*: the floor binds below 34 samples and the ceiling above
    200, and because those bounds are in samples the duration band they map to
    differs per cadence. Invariance is exact only for
    ``34 <= n_samples <= 200``; outside it, declare the window (rule 5).
    """
    if declared is None or declared <= 0:
        if n_samples <= 0:
            return R0_WINDOW_MIN_SAMPLES
        fraction = int(n_samples * R0_WINDOW_FRACTION)
        return min(R0_WINDOW_MAX_SAMPLES, max(R0_WINDOW_MIN_SAMPLES, fraction))
    return int(declared)

# Dynamic range is measured as a robust 5th-95th percentile span, minus this
# multiple of the channel's own noise standard deviation. Without the
# subtraction, an interference burst inflates max and min and *raises* the
# dynamic-range score for a recording that has got worse.
NOISE_SPAN_TOLERANCE = 3.0

# The time column is assumed to be milliseconds. When `samplingRateHz` is
# declared, the observed median gap should sit within these multiples of the
# expected period. A ratio far below 1 means the column is probably seconds or
# microseconds; far above 1 means it is probably nanoseconds or the declared
# rate is wrong. Both are unit faults rather than packet loss, so continuity is
# withheld instead of being reported as a real gap statistic.
MIN_TIME_UNIT_RATIO = 0.5
MAX_TIME_UNIT_RATIO = 2.0

# A dead sensing element is excluded from the live-channel mean because a
# constant channel contributes no span information. Excluding it must not read
# as an improvement, so each dead channel costs this much of the total.
DEAD_SENSOR_PENALTY = 12.5


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
class CalibrationDescriptor:
    a: float
    b: float
    reference_substance: Optional[str] = None
    reference_ppm: Optional[float] = None
    date: Optional[str] = None
    method: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in {
            "a": self.a,
            "b": self.b,
            "referenceSubstance": self.reference_substance,
            "referencePpm": self.reference_ppm,
            "date": self.date,
            "method": self.method,
        }.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationDescriptor":
        return cls(
            a=float(d["a"]),
            b=float(d["b"]),
            reference_substance=d.get("referenceSubstance"),
            reference_ppm=d.get("referencePpm"),
            date=d.get("date"),
            method=d.get("method"),
        )


@dataclass
class SensorDescriptor:
    sensor_type: str = "mox"
    channels: List[ChannelDescriptor] = field(default_factory=list)
    device: Optional[DeviceDescriptor] = None
    sampling_rate_hz: Optional[float] = None
    adc_bits: Optional[int] = None
    adc_max: Optional[int] = None
    time_column: str = "timestamp_ms"
    calibration: Optional[Dict[str, CalibrationDescriptor]] = None

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
        if self.calibration:
            d["calibration"] = {cid: c.to_dict() for cid, c in self.calibration.items()}
        d["timeColumn"] = self.time_column
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SensorDescriptor":
        calibration = d.get("calibration")
        return cls(
            sensor_type=d.get("sensorType", "mox"),
            channels=[ChannelDescriptor.from_dict(c) for c in d.get("channels", [])],
            device=DeviceDescriptor.from_dict(d.get("device")),
            sampling_rate_hz=d.get("samplingRateHz"),
            adc_bits=d.get("adcBits"),
            adc_max=d.get("adcMax"),
            time_column=d.get("timeColumn", "timestamp_ms"),
            calibration=({cid: CalibrationDescriptor.from_dict(c) for cid, c in calibration.items()}
                         if calibration else None),
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
    values: dict[str, Optional[float]]


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
    time_unit_mismatch: bool = False


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
