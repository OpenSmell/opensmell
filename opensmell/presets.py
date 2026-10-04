"""Use-case presets — the configuration layer for single-unit deployments.

A preset answers the questions a rig operator has *before* any data exists:
which sensors go in the array, how fast to sample, which feature groups are
worth computing at that cadence, how to establish R0, and which anomaly
thresholds to apply. It is deliberately a configuration object and not a model:
it says nothing about accuracy, and no field in it may be read as a performance
claim.

Three rules shape the whole module.

**Cadence is mandatory.** Every temporal feature divides a sample count by the
rate (`SAMPLING_CONTRACT.md` hard rule 2), so a preset without a rate produces
features whose units are unknown. There is no default here, on purpose: the SDK
already carries a default (`DEFAULT_SYNTHETIC_RATE_HZ = 10.0`) precisely so that
*recorded* data can be ingested with visible provenance, and a preset that
silently inherited it would launder an assumption into a configuration.

**Baseline is a duration, not a sample count.** A hardcoded 15 samples is 7.5 s
at the reference rig's 2 Hz and 1.5 s at 10 Hz (`r0_samples_for()` in
`interoperability/canonical_experiments/config.py`, hard rule 5). Presets store
`baseline.duration_s` and derive the sample count from the cadence, so the same
physical baseline applies on any rig.

**Unverified figures are marked, never dressed up.** Any number in a shipped
preset that has not been measured on this hardware is listed in that preset's
`provisional:` block, and `load_preset` refuses a provisional entry that does not
resolve to a real field — so the list cannot rot into a disclaimer that no longer
covers anything. Two classes of figure are treated this way:

- Cadences and protocol durations chosen by judgement (see each preset's
  `cadence_rationale`). Where a rate is documented by hardware
  (`HARDWARE.md`: 2 Hz production, 10 Hz kinetics, 1 Hz power-save) or already
  exercised by a dataset in this workspace (SmellNet and UCI-362 at ~1 Hz, the
  beef-spoilage series at one sample per minute), it is cited and not marked
  provisional. Where neither applies, it is marked.
- The false-alarm budgets. They are *policy targets chosen by judgement*, not
  measured operating points. The only false-alarm criterion with any evidence
  behind it in this workspace is `opensmell-rs/src/bin/mc_sweep.rs` — "false
  alarms <= 1/month at TPR >= 0.9" — and that is a synthetic-replay result with a
  stated caveat, not a field measurement.

R0 is deliberately under-specified. `HARDWARE.md` fixes the architecture but not
the element's resistance: R0 in ohms depends on the sensor lot, the load
resistor and the supply, so a shipped value would be fabricated. What *is*
knowable a priori is the clean-air resistance ratio Rs/R0, which the offline
constants table carries per model (`opensmell.constants.clean_air_ratio`) and
which `load_preset` cross-checks against, so a preset cannot claim a clean-air
ratio the SDK does not hold for that model.

No new dependency
-----------------
The preset files are YAML because YAML is what a human edits. Parsing them uses
a small block-YAML reader in this module (`_parse_simple_yaml`) rather than
PyYAML, because PyYAML is **not** a dependency of this package and the SDK
deliberately keeps its install list to numpy/pandas/scikit-learn/scipy. The
reader accepts only the subset the shipped presets use — block mappings, block
and flat-flow sequences, scalars, comments, and `|`/`>` block scalars — and
rejects everything else (flow mappings, nested flow, anchors, aliases, tags,
multiple documents, tab indentation) with a file-and-line error rather than
guessing. That is a smaller surface than a general YAML parser and a strictly
better failure mode for configuration: a preset that uses an unsupported
construct is a loud error, not a silently different value.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .constants import clean_air_ratio as _constants_clean_air_ratio
from .constants import is_power_law_sensor, sensor_gases
from .mox.timing import min_detectable_duration, timing_report
from .types import CONTEXT_COLUMN_HINTS, SENSOR_TYPES, ChannelDescriptor, SensorDescriptor

PRESET_SUFFIXES = (".yaml", ".yml")
#: Environment override for the preset search path (first candidate, if set).
PRESET_DIR_ENV = "OPENSMELL_PRESETS_DIR"

# Protocol durations from HARDWARE.md ("Sensor Heater Timing", "Baseline
# Collection"). These are documented for the reference ESP32 rig and are quoted
# rather than chosen.
DEFAULT_BASELINE_SECONDS = 1800.0
OPERATIONAL_SETTLING_SECONDS = 300.0
NEW_SENSOR_WARMUP_SECONDS = 172800.0

#: Feature groups a preset may request, in extractor order. Names match the
#: prefixes `opensmell.mox.features.feature_names` emits, so a request can be
#: checked against the real feature list rather than trusted.
FEATURE_GROUPS = (
    "device_agnostic",
    "absolute",
    "temporal",
    "health",
    "hardware",
    "saturation",
    "decay",
    "selectivity",
    "global",
)
FEATURE_GROUP_MARKERS = {
    "device_agnostic": "_da_",
    "absolute": "_abs_",
    "temporal": "_temp_",
    "health": "_health_",
    "hardware": "_hw_",
    "saturation": "_advanced_saturation_index",
    "decay": "_decay_",
    "selectivity": "sel_ratio_ch",
    "global": "global_",
}
# Groups whose values are integrals, slopes or crossing times. At a cadence whose
# resolution floor is longer than the event of interest these do not describe
# chemistry; they describe the shape of the sample sequence. A preset that asks
# for them at such a cadence gets an advisory, not an error.
CADENCE_SENSITIVE_GROUPS = ("device_agnostic", "temporal", "decay")

#: VerdictThresholds defaults, mirrored from
#: `opensmell-rs/src/anomaly/verdict.rs::VerdictThresholds::default`. A preset
#: that overrides any of them must record a `basis`, so a tuned threshold never
#: arrives without a stated reason.
VERDICT_THRESHOLD_DEFAULTS: Dict[str, float] = {
    "dead_noise_ceiling": 0.02,
    "drift_abs_floor": 1e-3,
    "drift_rel_ceiling": 0.01,
}
THRESHOLD_KEYS = tuple(VERDICT_THRESHOLD_DEFAULTS)

#: Why a channel is in the array. `primary` channels carry the decision,
#: `supporting` channels corroborate it, `cross_sensitivity` channels are there
#: because their cross-response to a *different* gas is the discriminating signal.
SENSOR_ROLES = ("primary", "supporting", "cross_sensitivity")

#: How R0 is established. Matches the firmware's `collectBaseline()`.
BASELINE_METHODS = ("median-of-leading-window",)

_ID_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


class PresetError(ValueError):
    """A preset is malformed, or a value in it is not usable.

    Raised by `load_preset` for every structural problem: a missing or
    non-positive cadence, an unknown feature group, a non-positive R0, a target
    gas the offline constants table does not hold for that sensor model, a
    threshold that was tuned without a stated basis. The message names the file
    and field so the fix is obvious.
    """


class PresetNotFoundError(PresetError, FileNotFoundError):
    """No shipped preset matches the requested name, or no file at the path.

    Also a `FileNotFoundError` so callers that only care that the input does not
    exist can catch the ordinary type.
    """


# ---------------------------------------------------------------------------
# Minimal block-YAML reader.
#
# Scope is exactly what the shipped presets use. Anything outside it raises with
# a file:line so a malformed preset never loads as a half-understood object.
# ---------------------------------------------------------------------------


@dataclass
class _Line:
    """One physical line: its indent, its raw text, and its 1-based number."""

    indent: int
    raw: str
    lineno: int


def _strip_comment(line: str) -> str:
    """Remove a trailing ``#`` comment, honouring quoted spans.

    A ``#`` only starts a comment at the start of the line or after whitespace,
    and never inside a quoted scalar, so ``"a # b"`` survives intact.
    """
    out: List[str] = []
    quote: Optional[str] = None
    for i, ch in enumerate(line):
        if quote is not None:
            out.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            out.append(ch)
            continue
        if ch == "#" and (i == 0 or line[i - 1] in " \t"):
            break
        out.append(ch)
    return "".join(out).rstrip()


def _content(line: _Line) -> str:
    """Comment-free, whitespace-trimmed text of a structural line."""
    return _strip_comment(line.raw).strip()


def _is_sequence_item(text: str) -> bool:
    return text == "-" or text.startswith("- ")


def _find_key_sep(text: str) -> int:
    """Index of the ``:`` that separates a mapping key, or -1 if there is none."""
    quote: Optional[str] = None
    for i, ch in enumerate(text):
        if quote is not None:
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            continue
        if ch == ":" and (i + 1 == len(text) or text[i + 1] in " \t"):
            return i
    return -1


def _split_key(text: str, where: str, lineno: int) -> Tuple[str, str]:
    sep = _find_key_sep(text)
    if sep < 0:
        raise PresetError(f"{where}:{lineno}: expected 'key: value', got {text!r}")
    key = text[:sep].strip()
    if not key:
        raise PresetError(f"{where}:{lineno}: empty mapping key")
    return key, text[sep + 1:].strip()


_INT_RE = re.compile(r"[+-]?\d+")
_FLOAT_RE = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?")
_NON_FINITE = (".inf", "-.inf", "+.inf", ".nan", "-.nan", "+.nan", "inf", "-inf", "nan")
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/"}


def _unescape(inner: str, where: str, lineno: int) -> str:
    out: List[str] = []
    i = 0
    while i < len(inner):
        ch = inner[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        if i + 1 >= len(inner):
            raise PresetError(f"{where}:{lineno}: string ends with a dangling backslash")
        nxt = inner[i + 1]
        if nxt not in _ESCAPES:
            raise PresetError(
                f"{where}:{lineno}: unsupported string escape '\\{nxt}'. "
                f"Supported: {''.join(sorted(_ESCAPES))}")
        out.append(_ESCAPES[nxt])
        i += 2
    return "".join(out)


def _split_flow_items(inner: str) -> List[str]:
    """Split a flow sequence body on commas that are not inside quotes."""
    parts: List[str] = []
    buf: List[str] = []
    quote: Optional[str] = None
    for ch in inner:
        if quote is not None:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            continue
        if ch == ",":
            parts.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    if quote is not None:
        raise ValueError("unterminated quoted scalar in flow sequence")
    parts.append("".join(buf))
    return parts


def _parse_flow_sequence(text: str, where: str, lineno: int) -> List[Any]:
    """Parse ``[a, b, "c"]`` — flat scalars only, one line only.

    Flow *mappings* and nested flow are refused: a preset that needs structure
    should use block style, where this reader's line numbers still apply.
    """
    if not text.endswith("]"):
        raise PresetError(f"{where}:{lineno}: unterminated flow sequence {text!r}")
    inner = text[1:-1].strip()
    if inner == "":
        return []
    try:
        parts = _split_flow_items(inner)
    except ValueError as exc:
        raise PresetError(f"{where}:{lineno}: {exc}") from None
    out: List[Any] = []
    for part in parts:
        item = part.strip()
        if item.startswith(("[", "{")):
            raise PresetError(
                f"{where}:{lineno}: nested flow collections are not supported; "
                f"use block style")
        out.append(_parse_scalar(item, where, lineno))
    return out


def _parse_scalar(text: str, where: str, lineno: int) -> Any:
    """Convert a scalar token to int / float / bool / None / str, or a flat list."""
    if text == "":
        return None
    first = text[0]
    if first == "[":
        return _parse_flow_sequence(text, where, lineno)
    if first == "{":
        raise PresetError(
            f"{where}:{lineno}: flow mappings are not supported by the preset reader; "
            f"use block style")
    if first in "&*!":
        raise PresetError(
            f"{where}:{lineno}: anchors, aliases and tags are not supported by the "
            f"preset reader")
    if first in ("'", '"'):
        if len(text) < 2 or text[-1] != first:
            raise PresetError(f"{where}:{lineno}: unterminated quoted scalar {text!r}")
        inner = text[1:-1]
        if first == "'":
            return inner.replace("''", "'")
        return _unescape(inner, where, lineno)
    low = text.lower()
    if low in ("null", "~"):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    if _INT_RE.fullmatch(text):
        return int(text)
    if _FLOAT_RE.fullmatch(text):
        return float(text)
    if low in _NON_FINITE:
        raise PresetError(
            f"{where}:{lineno}: non-finite numbers are not accepted in a preset "
            f"({text!r})")
    return text


def _tokenize(text: str, where: str) -> List[_Line]:
    lines: List[_Line] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            lines.append(_Line(-1, raw, lineno))
            continue
        indent = len(raw) - len(raw.lstrip(" \t"))
        if "\t" in raw[:indent]:
            raise PresetError(
                f"{where}:{lineno}: tab indentation is not supported; use spaces")
        lines.append(_Line(indent, raw, lineno))
    return lines


def _skip_blank(lines: List[_Line], i: int) -> int:
    while i < len(lines) and (lines[i].indent < 0 or not _content(lines[i])):
        i += 1
    return i


def _read_block_scalar(
    lines: List[_Line], i: int, parent_indent: int, style: str, where: str
) -> Tuple[str, int]:
    """Read a ``|`` (literal) or ``>`` (folded) block scalar.

    The raw text is used verbatim, so a ``#`` inside a rationale stays a ``#``.
    Both styles are read in clipped form (one trailing newline); strip markers
    are accepted and ignored, which keeps a preset's rendered text stable.
    """
    collected: List[str] = []
    while i < len(lines):
        line = lines[i]
        if line.indent < 0:
            collected.append("")
            i += 1
            continue
        if line.indent <= parent_indent:
            break
        collected.append(line.raw.rstrip())
        i += 1
    while collected and collected[-1] == "":
        collected.pop()
    if style.startswith("|"):
        # Literal style keeps line breaks, so the block's own indentation has to
        # come off every line -- otherwise the second line of a paragraph comes
        # back with four leading spaces that the first line does not have.
        widths = [len(line) - len(line.lstrip()) for line in collected if line.strip()]
        block_indent = min(widths) if widths else 0
        return "\n".join(line[block_indent:] for line in collected), i
    parts: List[str] = []
    buffer: List[str] = []
    for line in collected:
        if not line.strip():
            if buffer:
                parts.append(" ".join(buffer))
                buffer = []
            parts.append("")
            continue
        buffer.append(line.strip())
    if buffer:
        parts.append(" ".join(buffer))
    return "\n".join(parts), i


def _parse_mapping(
    lines: List[_Line], i: int, indent: int, where: str
) -> Tuple[Dict[str, Any], int]:
    out: Dict[str, Any] = {}
    while True:
        i = _skip_blank(lines, i)
        if i >= len(lines) or lines[i].indent < indent:
            return out, i
        if lines[i].indent > indent:
            raise PresetError(
                f"{where}:{lines[i].lineno}: unexpected indentation "
                f"(expected {indent} spaces, found {lines[i].indent})")
        text = _content(lines[i])
        if _is_sequence_item(text):
            return out, i
        key, rest = _split_key(text, where, lines[i].lineno)
        if key in out:
            raise PresetError(f"{where}:{lines[i].lineno}: duplicate key {key!r}")
        lineno = lines[i].lineno
        i += 1
        if rest == "":
            j = _skip_blank(lines, i)
            if j < len(lines) and lines[j].indent > indent:
                out[key], i = _parse_block(lines, j, lines[j].indent, where)
            elif j < len(lines) and lines[j].indent == indent and _is_sequence_item(_content(lines[j])):
                # A sequence may sit at its key's own indent.
                out[key], i = _parse_sequence(lines, j, indent, where)
            else:
                out[key] = None
        elif rest in ("|", ">", "|-", ">-"):
            out[key], i = _read_block_scalar(lines, i, indent, rest, where)
        else:
            out[key] = _parse_scalar(rest, where, lineno)


def _parse_sequence(
    lines: List[_Line], i: int, indent: int, where: str
) -> Tuple[List[Any], int]:
    items: List[Any] = []
    while True:
        i = _skip_blank(lines, i)
        if i >= len(lines) or lines[i].indent != indent:
            return items, i
        text = _content(lines[i])
        if not _is_sequence_item(text):
            return items, i
        after = text[1:]
        pad = len(after) - len(after.lstrip(" "))
        content = after.strip()
        lineno = lines[i].lineno
        if content == "":
            j = _skip_blank(lines, i + 1)
            if j < len(lines) and lines[j].indent > indent:
                value, i = _parse_block(lines, j, lines[j].indent, where)
            else:
                value, i = None, i + 1
        elif _find_key_sep(content) >= 0:
            # "- key: value" opens a mapping whose keys align at the content
            # column, so rewrite the dash line as a plain mapping line there.
            child_indent = indent + 1 + pad
            lines[i] = _Line(child_indent, " " * child_indent + content, lineno)
            value, i = _parse_block(lines, i, child_indent, where)
        else:
            value = _parse_scalar(content, where, lineno)
            i += 1
        items.append(value)


def _parse_block(
    lines: List[_Line], i: int, indent: int, where: str
) -> Tuple[Any, int]:
    text = _content(lines[i])
    if _is_sequence_item(text):
        return _parse_sequence(lines, i, indent, where)
    return _parse_mapping(lines, i, indent, where)


def _parse_simple_yaml(text: str, where: str) -> Dict[str, Any]:
    """Parse a single-document block-YAML mapping. Raises `PresetError` otherwise."""
    lines = _tokenize(text, where)
    i = _skip_blank(lines, 0)
    if i >= len(lines):
        return {}
    if _content(lines[i]) == "---":
        i = _skip_blank(lines, i + 1)
        if i >= len(lines):
            return {}
    value, i = _parse_block(lines, i, lines[i].indent, where)
    if not isinstance(value, dict):
        raise PresetError(f"{where}:1: a preset must be a mapping at the top level")
    i = _skip_blank(lines, i)
    if i < len(lines):
        marker = _content(lines[i])
        if marker == "---":
            raise PresetError(
                f"{where}:{lines[i].lineno}: multi-document YAML is not supported; "
                f"a preset is one document")
        raise PresetError(
            f"{where}:{lines[i].lineno}: unexpected content after the document body: "
            f"{marker!r}")
    return value


# ---------------------------------------------------------------------------
# Value coercion helpers. Every failure is a PresetError naming the field.
# ---------------------------------------------------------------------------


def _require_mapping(value: Any, where: str, key: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise PresetError(f"{where}: '{key}' must be a mapping, got {_typename(value)}")
    return value


def _require_list(value: Any, where: str, key: str) -> List[Any]:
    if not isinstance(value, list):
        raise PresetError(f"{where}: '{key}' must be a list, got {_typename(value)}")
    return value


def _require_text(value: Any, where: str, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PresetError(f"{where}: '{key}' must be a non-empty string, got {_typename(value)}")
    return value.strip()


def _require_number(
    value: Any,
    where: str,
    key: str,
    *,
    positive: bool = False,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PresetError(f"{where}: '{key}' must be a number, got {_typename(value)}")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise PresetError(f"{where}: '{key}' must be a finite number, got {value!r}")
    if positive and number <= 0:
        raise PresetError(f"{where}: '{key}' must be > 0, got {number!r}")
    if minimum is not None and number < minimum:
        raise PresetError(f"{where}: '{key}' must be >= {minimum}, got {number!r}")
    if maximum is not None and number > maximum:
        raise PresetError(f"{where}: '{key}' must be <= {maximum}, got {number!r}")
    return number


def _require_int(
    value: Any, where: str, key: str, *, minimum: Optional[int] = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PresetError(f"{where}: '{key}' must be an integer, got {_typename(value)}")
    if minimum is not None and value < minimum:
        raise PresetError(f"{where}: '{key}' must be >= {minimum}, got {value!r}")
    return int(value)


def _typename(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "mapping"
    return type(value).__name__


def _check_keys(data: Dict[str, Any], allowed: Tuple[str, ...], where: str) -> None:
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise PresetError(
            f"{where}: unknown field(s) {unknown}. Allowed: {sorted(allowed)}")


# ---------------------------------------------------------------------------
# Preset dataclasses.
# ---------------------------------------------------------------------------


@dataclass
class SensorSpec:
    """One channel in a preset's array.

    ``model`` is the part number, and when it is one of the models in
    `opensmell.constants.sensors.json` the tabulated response constants back the
    rest of the entry: ``targets`` must be gases that model is tabulated for, and
    ``clean_air_ratio`` must equal the tabulated Rs/R0. That is what keeps a
    preset's sensor list checkable rather than aspirational.

    ``r0`` is the measured clean-air baseline in the recording's own units
    (ohms, or ADC counts) and is deliberately absent from every shipped preset:
    R0 depends on the element lot, the load resistor and the supply, none of
    which the SDK pins, so shipping one would be a fabricated number. Record it
    here once the rig has been measured.
    """

    id: str
    model: str
    targets: List[str] = field(default_factory=list)
    role: str = "primary"
    clean_air_ratio: Optional[float] = None
    r0: Optional[float] = None
    r0_source: Optional[str] = None
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in {
            "id": self.id,
            "model": self.model,
            "targets": list(self.targets),
            "role": self.role,
            "clean_air_ratio": self.clean_air_ratio,
            "r0": self.r0,
            "r0_source": self.r0_source,
            "note": self.note,
        }.items() if v not in (None, [], "")}


@dataclass
class BaselineProtocol:
    """How R0 is established before any exposure is recorded.

    Stored as durations because that is what is physically meaningful across
    rigs; `r0_samples` is derived from ``duration_s * sampling_rate_hz`` unless a
    preset overrides it. ``warmup_s`` is the heater warm-up before the settling
    period: a new MOX element needs 24-48 h to stabilise, an already-running one
    about 5 min (HARDWARE.md), and using the wrong one silently poisons R0.
    """

    duration_s: float = DEFAULT_BASELINE_SECONDS
    settling_s: float = OPERATIONAL_SETTLING_SECONDS
    warmup_s: Optional[float] = None
    r0_samples: Optional[int] = None
    method: str = BASELINE_METHODS[0]

    def resolve_r0_samples(self, sampling_rate_hz: float) -> int:
        """Leading samples spanning `duration_s` at this cadence (>= 1)."""
        if self.r0_samples is not None:
            return int(self.r0_samples)
        return max(1, int(round(self.duration_s * float(sampling_rate_hz))))

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in {
            "duration_s": self.duration_s,
            "settling_s": self.settling_s,
            "warmup_s": self.warmup_s,
            "r0_samples": self.r0_samples,
            "method": self.method,
        }.items() if v is not None}


@dataclass
class AnomalyThresholds:
    """`VerdictThresholds` for one use case (opensmell-rs/src/anomaly/verdict.rs).

    - ``dead_noise_ceiling`` — dimensionless. A channel whose scatter has
      collapsed below this fraction of its own level is `Dead`, not quiet. The
      Rust default is 0.02 (~2% relative variation).
    - ``drift_abs_floor`` — absolute, in the units of the recorded series and per
      second. It is the one threshold that is *not* device-independent: a
      high-resistance element produces a larger absolute number for the same
      physical creep, which is why the policy also checks
      ``drift_rel_ceiling``. Ship it in the units your recorder writes and
      re-derive it per rig.
    - ``drift_rel_ceiling`` — dimensionless, ``|drift_rate| / R0`` per window.
      Above it, movement is too large to be ordinary creep and is not called
      correctable drift.

    `basis` is mandatory whenever any value departs from
    `VERDICT_THRESHOLD_DEFAULTS`: a tuned threshold without a stated reason is
    indistinguishable from a typo.
    """

    dead_noise_ceiling: float = VERDICT_THRESHOLD_DEFAULTS["dead_noise_ceiling"]
    drift_abs_floor: float = VERDICT_THRESHOLD_DEFAULTS["drift_abs_floor"]
    drift_rel_ceiling: float = VERDICT_THRESHOLD_DEFAULTS["drift_rel_ceiling"]
    basis: str = ""

    def is_sdk_default(self) -> bool:
        return all(
            getattr(self, key) == value for key, value in VERDICT_THRESHOLD_DEFAULTS.items()
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dead_noise_ceiling": self.dead_noise_ceiling,
            "drift_abs_floor": self.drift_abs_floor,
            "drift_rel_ceiling": self.drift_rel_ceiling,
            "basis": self.basis,
        }


@dataclass
class FalseAlarmBudget:
    """A false-alarm target for a continuously-monitoring deployment.

    Every shipped value is a judgement about alarm fatigue, not a measured
    operating point, and each one is listed in its preset's `provisional:`
    block. A budget is only meaningful where an alarm reaches a human on a
    schedule; a one-shot bench measurement has no per-day rate, and those presets
    leave it unset.
    """

    alarms_per_day: float
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {"alarms_per_day": self.alarms_per_day, "rationale": self.rationale}


@dataclass
class Preset:
    """One use case, configured end to end.

    Fields mirror the preset file one-for-one; `sampling_rate_hz` is populated
    whichever of the two mutually-redundant cadence forms the file used
    (`cadence_hz` or `cadence_period_s`), because a period expressed as "60" is
    exact where 1/60 is a rounded fraction.
    """

    id: str
    name: str
    description: str
    sampling_rate_hz: float
    sensors: List[SensorSpec] = field(default_factory=list)
    feature_groups: List[str] = field(default_factory=list)
    baseline: BaselineProtocol = field(default_factory=BaselineProtocol)
    thresholds: AnomalyThresholds = field(default_factory=AnomalyThresholds)
    sensor_type: str = "mox"
    context_columns: List[str] = field(default_factory=list)
    false_alarm_budget: Optional[FalseAlarmBudget] = None
    provisional: List[str] = field(default_factory=list)
    cadence_rationale: str = ""
    notes: str = ""
    source_path: Optional[str] = None

    # -- cadence ------------------------------------------------------------

    @property
    def cadence_period_s(self) -> float:
        """Sample period in seconds (``1 / sampling_rate_hz``)."""
        return 1.0 / float(self.sampling_rate_hz)

    @property
    def n_channels(self) -> int:
        return len(self.sensors)

    @property
    def min_detectable_duration_s(self) -> float:
        """Shortest event this cadence can resolve (``2 dt``).

        Delegated to `opensmell.mox.timing.min_detectable_duration` so a preset
        cannot disagree with the SDK about what its own cadence resolves.
        """
        return min_detectable_duration(self.sampling_rate_hz)

    @property
    def r0_samples(self) -> int:
        """Baseline length in samples at this preset's cadence."""
        return self.baseline.resolve_r0_samples(self.sampling_rate_hz)

    @property
    def timing(self):
        """`TimingReport` for this preset's cadence and channel count."""
        return timing_report(
            sampling_rate_hz=self.sampling_rate_hz, n_channels=self.n_channels
        )

    # -- feature selection --------------------------------------------------

    def requests(self, group: str) -> bool:
        return group in self.feature_groups

    def requests_cadence_sensitive(self) -> bool:
        return any(g in self.feature_groups for g in CADENCE_SENSITIVE_GROUPS)

    # -- advice, not validation --------------------------------------------

    def advisories(self) -> List[str]:
        """Non-fatal observations a user should read before deploying.

        Deliberately not errors: a 1-sample-per-minute spoilage monitor with the
        `temporal` group switched off is a *good* configuration, and a user who
        wants the group anyway is entitled to it. These are the things the preset
        cannot refuse but should not let pass unmentioned.
        """
        notes: List[str] = []
        for flag in self.timing.flags:
            notes.append(f"timing: {flag}")
        floor = self.min_detectable_duration_s
        if self.requests_cadence_sensitive() and floor > 60.0:
            notes.append(
                f"cadence: {', '.join(g for g in CADENCE_SENSITIVE_GROUPS if self.requests(g))} "
                f"are requested but the shortest resolvable event at "
                f"{self.sampling_rate_hz:g} Hz is {floor:g} s; those features will "
                f"describe the sample sequence, not the chemistry")
        if self.n_channels < 2 and self.requests("selectivity"):
            notes.append("selectivity: ratios need at least two channels")
        if self.r0_samples < 2:
            notes.append(
                f"baseline: {self.r0_samples} sample(s) at this cadence is not enough "
                f"for a stable R0; shorten the cadence or lengthen the baseline")
        if self.thresholds.dead_noise_ceiling >= 0.5:
            notes.append(
                "thresholds: dead_noise_ceiling at or above 0.5 would only condemn a "
                "channel that has stopped responding entirely")
        return notes

    # -- integration --------------------------------------------------------

    def to_sensor_descriptor(self) -> SensorDescriptor:
        """This preset's array as a `SensorDescriptor` for a `.osmell` manifest."""
        return SensorDescriptor(
            sensor_type=self.sensor_type,
            channels=[
                ChannelDescriptor(id=s.id, unit="adc" if self.sensor_type == "mox" else "")
                for s in self.sensors
            ],
            sampling_rate_hz=self.sampling_rate_hz,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "cadence_hz": self.sampling_rate_hz,
            "cadence_period_s": self.cadence_period_s,
            "cadence_rationale": self.cadence_rationale,
            "sensor_type": self.sensor_type,
            "sensors": [s.to_dict() for s in self.sensors],
            "context_columns": list(self.context_columns),
            "feature_groups": list(self.feature_groups),
            "baseline": self.baseline.to_dict(),
            "r0_samples": self.r0_samples,
            "thresholds": self.thresholds.to_dict(),
            "false_alarm_budget": (
                self.false_alarm_budget.to_dict() if self.false_alarm_budget else None
            ),
            "provisional": list(self.provisional),
            "notes": self.notes,
            "advisories": self.advisories(),
            "source_path": self.source_path,
        }

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return f"{self.id} ({self.name})"

    # -- validation ---------------------------------------------------------

    def validate(self) -> "Preset":
        """Check cross-field and semantic invariants. Returns self; raises otherwise.

        Type and range checks happen while the file is read, because a field with
        the wrong type must not reach a dataclass that claims it is a float.
        What is left is the part that needs the rest of the preset, the shipped
        constants table, and the real feature extractor:

        * the id matches the file it came from;
        * every requested feature group exists in the extractor's names for this
          channel count, and every sensor model is one the SDK holds;
        * every declared target gas is tabulated for its model, and every
          declared clean-air ratio equals the tabulated Rs/R0;
        * thresholds that depart from the SDK default carry a basis;
        * every `provisional:` entry resolves to a real field, so the
          not-yet-validated list cannot silently stop covering anything.
        """
        where = f"{self.source_path}" if self.source_path else f"preset {self.id!r}"

        if not _ID_RE.fullmatch(self.id):
            raise PresetError(
                f"{where}: id {self.id!r} must be lower-case snake_case "
                f"(letters, digits, single underscores)")
        if self.source_path is not None:
            stem = Path(self.source_path).stem
            if stem != self.id:
                raise PresetError(
                    f"{where}: id {self.id!r} does not match the file name {stem!r}; "
                    f"rename one so `opensmell-presets show {self.id}` finds this file")

        if self.sensor_type not in SENSOR_TYPES:
            raise PresetError(
                f"{where}: sensor_type {self.sensor_type!r} is not one of {list(SENSOR_TYPES)}")
        if not self.sensors:
            raise PresetError(f"{where}: 'sensors' must list at least one channel")
        if not self.feature_groups:
            raise PresetError(
                f"{where}: 'feature_groups' must list at least one group; "
                f"valid groups: {list(FEATURE_GROUPS)}")
        unknown_groups = [g for g in self.feature_groups if g not in FEATURE_GROUPS]
        if unknown_groups:
            raise PresetError(
                f"{where}: unknown feature group(s) {unknown_groups}; "
                f"valid groups: {list(FEATURE_GROUPS)}")
        duplicates = sorted({g for g in self.feature_groups if self.feature_groups.count(g) > 1})
        if duplicates:
            raise PresetError(f"{where}: duplicate feature group(s) {duplicates}")

        names = self.feature_names()
        for group in self.feature_groups:
            marker = FEATURE_GROUP_MARKERS[group]
            if not any(marker in n for n in names):
                raise PresetError(
                    f"{where}: feature group {group!r} produces no feature for a "
                    f"{self.n_channels}-channel array; the extractor emits {len(names)} "
                    f"names and none contains {marker!r}")

        seen: Dict[str, int] = {}
        for index, sensor in enumerate(self.sensors):
            label = f"{where}: sensors[{index}] ({sensor.id})"
            if not _ID_RE.fullmatch(sensor.id):
                raise PresetError(
                    f"{label}: channel id must be lower-case snake_case, got {sensor.id!r}")
            if sensor.id in seen:
                raise PresetError(
                    f"{label}: duplicate channel id (also at sensors[{seen[sensor.id]}])")
            seen[sensor.id] = index
            if sensor.role not in SENSOR_ROLES:
                raise PresetError(
                    f"{label}: role {sensor.role!r} is not one of {list(SENSOR_ROLES)}")
            if sensor.r0 is not None and sensor.r0 <= 0:
                raise PresetError(
                    f"{label}: r0 must be > 0 (a baseline resistance of "
                    f"{sensor.r0!r} is not physical), got {sensor.r0!r}")
            if sensor.r0 is None and sensor.r0_source is not None:
                raise PresetError(
                    f"{label}: r0_source is set but r0 is not; state the measured "
                    f"baseline or drop the source")
            self._verify_model(label, sensor)
            self._verify_targets(label, sensor)

        for column in self.context_columns:
            if column.lower() not in CONTEXT_COLUMN_HINTS:
                raise PresetError(
                    f"{where}: context column {column!r} is not recognised; "
                    f"known context columns: {list(CONTEXT_COLUMN_HINTS)}. Context "
                    f"columns are preserved as metadata and are never scored as "
                    f"sensor channels")

        if self.baseline.method not in BASELINE_METHODS:
            raise PresetError(
                f"{where}: baseline method {self.baseline.method!r} is not one of "
                f"{list(BASELINE_METHODS)}")
        if self.r0_samples < 2:
            raise PresetError(
                f"{where}: baseline of {self.baseline.duration_s:g} s at "
                f"{self.sampling_rate_hz:g} Hz is {self.r0_samples} sample(s); at "
                f"least 2 are needed to estimate a median and a spread")

        for key in THRESHOLD_KEYS:
            value = getattr(self.thresholds, key)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise PresetError(f"{where}: thresholds.{key} must be a number")
        if not 0 < self.thresholds.dead_noise_ceiling <= 1:
            raise PresetError(
                f"{where}: thresholds.dead_noise_ceiling must be in (0, 1] — it is a "
                f"scatter-to-level ratio, got {self.thresholds.dead_noise_ceiling!r}")
        if self.thresholds.drift_abs_floor < 0:
            raise PresetError(
                f"{where}: thresholds.drift_abs_floor must be >= 0, got "
                f"{self.thresholds.drift_abs_floor!r}")
        if self.thresholds.drift_rel_ceiling <= 0:
            raise PresetError(
                f"{where}: thresholds.drift_rel_ceiling must be > 0, got "
                f"{self.thresholds.drift_rel_ceiling!r}")
        if not self.thresholds.is_sdk_default() and not self.thresholds.basis.strip():
            raise PresetError(
                f"{where}: thresholds differ from the SDK defaults "
                f"({VERDICT_THRESHOLD_DEFAULTS}) but no 'basis' was given; a tuned "
                f"threshold must say why it was tuned")

        if self.false_alarm_budget is not None:
            if self.false_alarm_budget.alarms_per_day <= 0:
                raise PresetError(
                    f"{where}: false_alarm_budget.alarms_per_day must be > 0, got "
                    f"{self.false_alarm_budget.alarms_per_day!r}")
            if not self.false_alarm_budget.rationale.strip():
                raise PresetError(
                    f"{where}: false_alarm_budget.rationale must say who is paged and "
                    f"what a nuisance alarm costs")

        for entry in self.provisional:
            if not entry.strip():
                raise PresetError(f"{where}: 'provisional' contains an empty entry")
            try:
                resolve_field(self.to_dict(), entry)
            except KeyError as exc:
                raise PresetError(
                    f"{where}: provisional entry {entry!r} does not resolve to a field "
                    f"({exc}); the not-yet-validated list must name real fields or it "
                    f"stops being a disclaimer") from None
        return self

    def feature_names(self) -> List[str]:
        """The extractor's feature names for this preset's channel count."""
        from .mox.features import feature_names as _feature_names

        return _feature_names(n_channels=self.n_channels)

    # -- cross-checks against the shipped constants -------------------------

    @staticmethod
    def _verify_model(label: str, sensor: SensorSpec) -> None:
        if sensor.clean_air_ratio is not None and not is_power_law_sensor(sensor.model):
            raise PresetError(
                f"{label}: clean_air_ratio is declared for {sensor.model!r}, which has "
                f"no power-law table in opensmell.constants. There is no datasheet "
                f"Rs/R0 the SDK can check it against, so it cannot be presented as "
                f"verified; drop it or record r0 from a measurement instead")

    @staticmethod
    def _verify_targets(label: str, sensor: SensorSpec) -> None:
        if not sensor.targets:
            return
        try:
            tabulated = sensor_gases(sensor.model)
        except KeyError:
            return  # no gases table: nothing to check against, note carries the caveat
        unknown = [g for g in sensor.targets if g not in tabulated]
        if unknown:
            raise PresetError(
                f"{label}: target gas(s) {unknown} are not tabulated for "
                f"{sensor.model!r} in opensmell.constants. Tabulated for this model: "
                f"{tabulated}")


def resolve_field(data: Dict[str, Any], path: str) -> Any:
    """Resolve a dotted/indexed path such as ``sensors[0].r0_ratio``.

    Raises `KeyError` naming the first segment that does not exist, so a stale
    `provisional:` entry is reported against the field it meant to cover.
    """
    node: Any = data
    i = 0
    end = len(path)
    while i < end:
        if path[i] == ".":
            i += 1
            continue
        if path[i] == "[":
            close = path.find("]", i)
            if close < 0:
                raise KeyError(f"unclosed '[' (in {path!r})")
            index = path[i + 1:close]
            if not index.isdigit() or not isinstance(node, list):
                raise KeyError(f"{index!r} is not a list index (in {path!r})")
            position = int(index)
            if position >= len(node):
                raise KeyError(f"index {position} out of range (in {path!r})")
            node = node[position]
            i = close + 1
        else:
            stops = [p for p in (path.find(".", i), path.find("[", i)) if p >= 0]
            close = min(stops) if stops else end
            name = path[i:close]
            if not isinstance(node, dict) or name not in node:
                raise KeyError(f"{name!r} (in {path!r})")
            node = node[name]
            # Stop *at* a following "[" so the next pass reads it as an index.
            i = close if path[close:close + 1] == "[" else close + 1
    return node


# ---------------------------------------------------------------------------
# Reading.
# ---------------------------------------------------------------------------

_PRESET_KEYS = (
    "id", "name", "description", "cadence_hz", "cadence_period_s",
    "cadence_rationale", "sensor_type", "sensors", "context_columns",
    "feature_groups", "baseline", "thresholds", "false_alarm_budget",
    "provisional", "notes",
)
_SENSOR_KEYS = ("id", "model", "targets", "role", "clean_air_ratio", "r0", "r0_source", "note")
_BASELINE_KEYS = ("duration_s", "settling_s", "warmup_s", "r0_samples", "method")
_THRESHOLD_KEYS = THRESHOLD_KEYS + ("basis",)
_BUDGET_KEYS = ("alarms_per_day", "rationale")


def _build_sensor(raw: Any, index: int, where: str) -> SensorSpec:
    label = f"{where}: sensors[{index}]"
    data = _require_mapping(raw, label, "<entry>")
    _check_keys(data, _SENSOR_KEYS, label)
    if "id" not in data:
        raise PresetError(f"{label}: every sensor needs an 'id' (the channel name in the recording)")
    sensor_id = _require_text(data["id"], label, "id")
    if "model" not in data:
        raise PresetError(
            f"{label} ({sensor_id}): 'model' is required; a channel with no part "
            f"number cannot be checked against the SDK's sensor constants")
    model = _require_text(data["model"], label, "model")
    targets = [
        _require_text(t, label, "targets")
        for t in _require_list(data.get("targets", []), label, "targets")
    ]
    role = _require_text(data.get("role", "primary"), label, "role")
    clean = data.get("clean_air_ratio")
    clean_ratio = (
        _require_number(clean, label, "clean_air_ratio", positive=True) if clean is not None else None
    )
    if clean_ratio is None and is_power_law_sensor(model):
        # The SDK can check a declared ratio against the constants table, and it
        # can fall back to the tabulated value — but a preset that stays silent
        # has not said what Rs/R0 it assumes, and Rs/R0 is what every R0-derived
        # feature is divided by. State it; the loader then verifies it.
        raise PresetError(
            f"{label} ({sensor_id}): 'clean_air_ratio' is required for {model}, "
            f"which the SDK holds an Rs/R0 for. Declare it (expected "
            f"{_constants_clean_air_ratio(model):g}) so the value can be verified, or "
            f"use a model with no tabulated power law, which has nothing to declare")
    r0_raw = data.get("r0")
    r0 = _require_number(r0_raw, label, "r0", positive=True) if r0_raw is not None else None
    r0_source = data.get("r0_source")
    if r0_source is not None:
        r0_source = _require_text(r0_source, label, "r0_source")
    note = data.get("note")
    if note is not None:
        note = _require_text(note, label, "note")
    return SensorSpec(
        id=sensor_id, model=model, targets=targets, role=role,
        clean_air_ratio=clean_ratio, r0=r0, r0_source=r0_source, note=note,
    )


def _build_baseline(raw: Any, where: str) -> BaselineProtocol:
    label = f"{where}: baseline"
    data = _require_mapping(raw, label, "baseline")
    _check_keys(data, _BASELINE_KEYS, label)
    duration = _require_number(
        data.get("duration_s", DEFAULT_BASELINE_SECONDS), label, "duration_s", positive=True)
    settling = _require_number(
        data.get("settling_s", OPERATIONAL_SETTLING_SECONDS), label, "settling_s", positive=True)
    warmup_raw = data.get("warmup_s")
    warmup = (
        _require_number(warmup_raw, label, "warmup_s", positive=True)
        if warmup_raw is not None else None
    )
    r0_raw = data.get("r0_samples")
    r0_samples = (
        _require_int(r0_raw, label, "r0_samples", minimum=1) if r0_raw is not None else None
    )
    return BaselineProtocol(
        duration_s=duration,
        settling_s=settling,
        warmup_s=warmup,
        r0_samples=r0_samples,
        method=_require_text(data.get("method", BASELINE_METHODS[0]), label, "method"),
    )


def _build_thresholds(raw: Any, where: str) -> AnomalyThresholds:
    label = f"{where}: thresholds"
    data = _require_mapping(raw, label, "thresholds")
    _check_keys(data, _THRESHOLD_KEYS, label)
    values = {}
    for key in THRESHOLD_KEYS[:3]:
        if key not in data:
            raise PresetError(
                f"{label}: '{key}' is required. VerdictThresholds "
                f"(opensmell-rs/src/anomaly/verdict.rs) has all three fields and each "
                f"one changes what the verdict means: "
                f"{sorted(VERDICT_THRESHOLD_DEFAULTS)}")
        values[key] = _require_number(data[key], label, key)
    basis = data.get("basis", "")
    if basis is not None and str(basis).strip():
        basis = _require_text(basis, label, "basis")
    else:
        basis = ""
    return AnomalyThresholds(basis=basis, **values)


def _build_budget(raw: Any, where: str) -> FalseAlarmBudget:
    label = f"{where}: false_alarm_budget"
    data = _require_mapping(raw, label, "false_alarm_budget")
    _check_keys(data, _BUDGET_KEYS, label)
    if "alarms_per_day" not in data:
        raise PresetError(
            f"{label}: 'alarms_per_day' is required when a false-alarm budget is set")
    alarms = _require_number(data["alarms_per_day"], label, "alarms_per_day", positive=True)
    rationale = data.get("rationale", "")
    return FalseAlarmBudget(
        alarms_per_day=alarms,
        rationale=_require_text(rationale, label, "rationale"),
    )


def _build_preset(raw: Dict[str, Any], source: Optional[str]) -> Preset:
    where = str(source) if source else "<preset>"
    _check_keys(raw, _PRESET_KEYS, where)

    if "id" not in raw:
        raise PresetError(f"{where}: 'id' is required (it is the name users pass to load_preset)")
    preset_id = _require_text(raw["id"], where, "id")
    if "name" not in raw:
        raise PresetError(f"{where} ({preset_id}): 'name' is required")
    name = _require_text(raw["name"], where, "name")
    description = _require_text(raw.get("description", ""), where, "description")

    # --- cadence: mandatory, and never defaulted ---
    if "cadence_hz" not in raw and "cadence_period_s" not in raw:
        raise PresetError(
            f"{where} ({preset_id}): no sampling rate. Every temporal feature divides "
            f"a sample count by the rate (SAMPLING_CONTRACT.md hard rule 2), so a "
            f"preset without one produces features whose units are unknown. Declare "
            f"'cadence_hz' or 'cadence_period_s'.")
    period = raw.get("cadence_period_s")
    rate = raw.get("cadence_hz")
    if rate is not None:
        rate = _require_number(rate, where, "cadence_hz", positive=True)
    if period is not None:
        period = _require_number(period, where, "cadence_period_s", positive=True)
        derived = 1.0 / period
        if rate is not None and abs(rate - derived) > 1e-9:
            raise PresetError(
                f"{where} ({preset_id}): cadence_hz {rate!r} and cadence_period_s "
                f"{period!r} disagree (1/{period:g} = {derived:g} Hz)")
        rate = derived
    assert rate is not None  # one of the two was present and positive

    rationale = raw.get("cadence_rationale", "")
    if rationale is not None:
        rationale = _require_text(rationale, where, "cadence_rationale")

    sensor_type = _require_text(raw.get("sensor_type", "mox"), where, "sensor_type")
    sensors = [
        _build_sensor(entry, i, where)
        for i, entry in enumerate(_require_list(raw.get("sensors", []), where, "sensors"))
    ]
    context = [
        _require_text(c, where, "context_columns").lower()
        for c in _require_list(raw.get("context_columns", []), where, "context_columns")
    ]
    groups = [
        _require_text(g, where, "feature_groups")
        for g in _require_list(raw.get("feature_groups", []), where, "feature_groups")
    ]
    if "baseline" not in raw:
        raise PresetError(f"{where} ({preset_id}): 'baseline' is required (how R0 is established)")
    baseline = _build_baseline(raw["baseline"], where)
    if baseline.r0_samples is not None:
        # Not resolve_r0_samples(): that prefers the declared count, which is the
        # thing being checked here.
        implied = max(1, int(round(baseline.duration_s * float(rate))))
        if baseline.r0_samples != implied:
            raise PresetError(
                f"{where} ({preset_id}): baseline.r0_samples {baseline.r0_samples} "
                f"disagrees with duration_s {baseline.duration_s:g} at "
                f"{float(rate):g} Hz, which is {implied} samples. Declare one or the "
                f"other: a preset whose two statements disagree has no single R0")
    if "thresholds" not in raw:
        raise PresetError(f"{where} ({preset_id}): 'thresholds' is required")
    thresholds = _build_thresholds(raw["thresholds"], where)
    budget_raw = raw.get("false_alarm_budget")
    budget = _build_budget(budget_raw, where) if budget_raw is not None else None
    provisional = [
        _require_text(p, where, "provisional")
        for p in _require_list(raw.get("provisional", []), where, "provisional")
    ]
    notes = raw.get("notes", "")
    if notes is not None:
        notes = _require_text(notes, where, "notes")

    return Preset(
        id=preset_id,
        name=name,
        description=description,
        sampling_rate_hz=float(rate),
        sensors=sensors,
        feature_groups=groups,
        baseline=baseline,
        thresholds=thresholds,
        sensor_type=sensor_type,
        context_columns=context,
        false_alarm_budget=budget,
        provisional=provisional,
        cadence_rationale=rationale or "",
        notes=notes or "",
        source_path=str(source) if source else None,
    )


def _verify_clean_air_ratios(preset: Preset) -> None:
    """Cross-check declared Rs/R0 against the offline constants table."""
    where = preset.source_path or preset.id
    for index, sensor in enumerate(preset.sensors):
        if sensor.clean_air_ratio is None:
            continue
        try:
            tabulated = _constants_clean_air_ratio(sensor.model)
        except KeyError:
            continue  # not in the table; _verify_model has already refused a ratio here
        if abs(sensor.clean_air_ratio - tabulated) > 1e-6:
            raise PresetError(
                f"{where}: sensors[{index}] ({sensor.id}): clean_air_ratio "
                f"{sensor.clean_air_ratio!r} disagrees with the Rs/R0 the SDK holds "
                f"for {sensor.model!r} ({tabulated}). Either the value is wrong or the "
                f"constants table needs updating; do not ship a ratio that is not the "
                f"one the calibration machinery would use.")


def presets_dir() -> Path:
    """Directory holding the shipped presets.

    Searched in order: ``$OPENSMELL_PRESETS_DIR``, ``<repo>/presets`` (the
    source checkout and editable installs), ``opensmell/presets`` (vendored
    inside the package), ``<prefix>/share/opensmell/presets`` (installed
    layout). Raises `PresetNotFoundError` naming every candidate when none
    exists, because "no presets" and "presets somewhere else" need different
    fixes.
    """
    candidates: List[Path] = []
    env = os.environ.get(PRESET_DIR_ENV)
    if env:
        candidates.append(Path(env))
    here = Path(__file__).resolve()
    candidates.append(here.parents[1] / "presets")
    candidates.append(here.parent / "presets")
    candidates.append(Path(sys.prefix) / "share" / "opensmell" / "presets")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    searched = "\n  ".join(str(c) for c in candidates)
    raise PresetNotFoundError(
        f"No presets directory found. Searched:\n  {searched}\n"
        f"Set {PRESET_DIR_ENV} or pass an explicit path to load_preset().")


def preset_files() -> List[Path]:
    """Every shipped preset file, sorted by id."""
    directory = presets_dir()
    files = [
        p for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() in PRESET_SUFFIXES and not p.name.startswith("_")
    ]
    if not files:
        raise PresetNotFoundError(f"No preset files in {directory}")
    return files


def list_presets() -> List[str]:
    """Ids of every shipped preset, sorted.

    The id is the file stem, which is why `load_preset` can take either an id or
    a path and why `Preset.validate` insists the two agree.
    """
    return [p.stem for p in preset_files()]


def load_preset(name_or_path: str) -> Preset:
    """Load, validate and return one preset by id or by file path.

    ``name_or_path`` is treated as a path when it exists, when it ends in
    ``.yaml``/``.yml``, or when it contains a separator; otherwise it is an id
    looked up in `presets_dir`. Raises `PresetError` for a malformed preset
    (the message carries the file and the field) and `PresetNotFoundError` when
    nothing matches.
    """
    candidate = Path(name_or_path)
    separators = [os.sep] + ([os.altsep] if os.altsep else [])
    looks_like_path = (
        candidate.exists()
        or candidate.suffix.lower() in PRESET_SUFFIXES
        or any(sep and sep in name_or_path for sep in separators)
    )
    if looks_like_path:
        if not candidate.is_file():
            raise PresetNotFoundError(f"No preset file at {candidate}")
        return _load_file(candidate)

    directory = presets_dir()
    for suffix in PRESET_SUFFIXES:
        path = directory / f"{name_or_path}{suffix}"
        if path.is_file():
            return _load_file(path)
    try:
        available = ", ".join(list_presets()) or "(none)"
    except PresetNotFoundError:
        # The directory exists but holds no preset files. Say which name was
        # asked for: "no preset files" alone does not tell a user what to fix.
        available = "(none)"
    raise PresetNotFoundError(
        f"No preset named {name_or_path!r} in {directory}. "
        f"Available: {available}")


def load_all_presets() -> Dict[str, Preset]:
    """Every shipped preset, keyed by id. One bad file fails the whole call."""
    presets: Dict[str, Preset] = {}
    for path in preset_files():
        preset = _load_file(path)
        if preset.id in presets:
            raise PresetError(
                f"{path}: id {preset.id!r} is already defined by "
                f"{presets[preset.id].source_path}")
        presets[preset.id] = preset
    return dict(sorted(presets.items()))


def _load_file(path: Path) -> Preset:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PresetNotFoundError(f"Cannot read preset {path}: {exc}") from None
    raw = _parse_simple_yaml(text, str(path))
    if not raw:
        raise PresetError(f"{path}: file is empty")
    preset = _build_preset(raw, str(path))
    _verify_clean_air_ratios(preset)
    return preset.validate()


__all__ = [
    "Preset",
    "SensorSpec",
    "BaselineProtocol",
    "AnomalyThresholds",
    "FalseAlarmBudget",
    "PresetError",
    "PresetNotFoundError",
    "FEATURE_GROUPS",
    "FEATURE_GROUP_MARKERS",
    "CADENCE_SENSITIVE_GROUPS",
    "VERDICT_THRESHOLD_DEFAULTS",
    "THRESHOLD_KEYS",
    "SENSOR_ROLES",
    "DEFAULT_BASELINE_SECONDS",
    "OPERATIONAL_SETTLING_SECONDS",
    "NEW_SENSOR_WARMUP_SECONDS",
    "PRESET_SUFFIXES",
    "PRESET_DIR_ENV",
    "presets_dir",
    "preset_files",
    "list_presets",
    "load_preset",
    "load_all_presets",
    "resolve_field",
]
