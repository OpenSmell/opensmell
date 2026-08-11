"""CSV parsing for the .osmell data.csv member (and tolerant raw CSV import).

Mirrors `osmograph-web/lib/osmell/csv.ts` 1:1 so Python and TypeScript agree on
row filtering, quote handling, time-column detection, sampling-rate guessing and
column classification.

Import philosophy: parse never throws on structure — it adopts the file and
reports how it was interpreted. A missing time column becomes explicit
synthetic timing (`timeSource="synthetic"`, default rate), environmental columns
(Temperature, Pressure, Humidity, Gas_Resistance, Altitude…) are detected and
preserved as context rather than scored as sensor channels, and every
interpretation decision is surfaced in `warnings` so the UI can teach the user
the recommended structure instead of rejecting their data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from statistics import median as _py_median
from typing import List, Optional, Tuple

from .types import (
    CONTEXT_COLUMN_HINTS,
    DEFAULT_SYNTHETIC_RATE_HZ,
    TIME_COLUMNS,
    TIME_COLUMN_ALIASES,
    ParsedSample,
)

MOX_CHANNEL_IDS = ("VOC", "Alcohol", "LPG", "CO", "NO2", "C2H5OH")


@dataclass
class CsvParseResult:
    header: List[str]
    time_column: Optional[str]
    time_source: str  # "column" | "synthetic"
    synthetic_rate_hz: float  # assumed timing when time_source == "synthetic"
    samples: List[ParsedSample]
    row_count: int
    channel_ids: List[str]
    context_columns: List[str]
    unknown_columns: List[str]
    skipped_columns: List[str]
    guess_sampling_rate_hz: float
    non_finite: int
    unsorted: bool
    warnings: List[str] = field(default_factory=list)


def _parse_row(raw: str, delim: str = ",") -> List[str]:
    cells: List[str] = []
    current = ""
    in_quotes = False
    i = 0
    while i < len(raw):
        c = raw[i]
        if c == '"':
            if in_quotes and i + 1 < len(raw) and raw[i + 1] == '"':
                current += '"'
                i += 1
            else:
                in_quotes = not in_quotes
        elif c == delim and not in_quotes:
            cells.append(current)
            current = ""
        else:
            current += c
        i += 1
    cells.append(current)
    return cells


def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return _py_median(values)


def detect_time_column(header: List[str]) -> Optional[str]:
    """Find the time column by alias (case-insensitive), or None."""
    info = _detect_time_column_info(header)
    return info[0] if info else None


def _detect_time_column_info(header: List[str]) -> Optional[Tuple[str, str]]:
    """Return (name, unit) for a recognised time column, or None."""
    for raw in header:
        n = re.sub(r"[\(\)\[\]]", "", raw.lower().strip())
        n = re.sub(r"\s+", "", n)
        if re.match(r"^(timestamp|elapsed)(_ms)?$", n):
            return (raw, "ms")
        if re.match(r"^time(_ms)?$", n):
            return (raw, "ms")
        if re.match(r"^time(_s|s)$", n):
            return (raw, "s")
        if re.match(r"^synthetic_index$", n):
            return (raw, "ms")
    return None


def _detect_delimiter(sample_line: str) -> str:
    counts: List[Tuple[int, str]] = []
    for c in (",", ";", "\t", "|"):
        counts.append((sample_line.count(c), c))
    return max(counts)[1]


def _parse_time_value(raw: str, unit: str) -> Optional[float]:
    """Coerce a raw time cell to milliseconds.

    Handles plain numbers (ms), epoch seconds, ISO/datetime strings and
    HH:MM:SS[.mmm] stopwatch-style values.
    """
    s = raw.strip()
    if not s:
        return None

    n = _safe_float(s)
    if n is not None:
        return n * 1000.0 if unit == "s" else n

    iso = _iso_to_ms(s)
    if iso is not None:
        return iso

    clock = re.match(r"^(\d{1,3}):(\d{2}):(\d{2})(?:[.,](\d{1,6}))?$", s)
    if clock:
        h, m, sec = int(clock.group(1)), int(clock.group(2)), int(clock.group(3))
        frac_raw = clock.group(4) or ""
        if m < 60 and sec < 60:
            frac = float(frac_raw) / (10 ** len(frac_raw)) if frac_raw else 0.0
            return (h * 3600 + m * 60 + sec) * 1000.0 + frac * 1000.0

    return None


def _iso_to_ms(s: str) -> Optional[float]:
    """Parse ISO-8601 / common datetime strings to epoch ms (best effort)."""
    text = s.replace("Z", "+00:00")
    try:
        if "T" in text or " " in text:
            from datetime import datetime

            sep = "T" if "T" in text else " "
            date_part, _, time_part = text.partition(sep)
            time_part = time_part.split("+")[0].split("-")[0].rstrip() if time_part else ""
            if time_part:
                parsed = datetime.fromisoformat(f"{date_part}T{time_part}")
            else:
                parsed = datetime.fromisoformat(date_part)
            if parsed.tzinfo is None:
                return parsed.timestamp() * 1000.0
            return parsed.timestamp() * 1000.0
    except (ValueError, TypeError):
        return None
    return None


def is_context_column(name: str) -> bool:
    """True for environmental/metadata columns that are not sensor channels."""
    n = name.lower()
    return any(hint in n for hint in CONTEXT_COLUMN_HINTS)


def parse_csv(text: str) -> CsvParseResult:
    warnings: List[str] = []
    raw_rows = text.splitlines()
    rows = [r.strip() for r in raw_rows if r.strip() and not r.strip().startswith("#")]

    if not rows:
        raise ValueError("The CSV file is empty.")
    if len(rows) == 1:
        raise ValueError("The CSV has a header but no data rows.")

    delim = _detect_delimiter(rows[1])
    if delim != ",":
        warnings.append(
            f'Detected "{delim}"-delimited values; parsed accordingly. '
            f"Convert to comma-delimited CSV for widest tool compatibility."
        )

    header = [h.strip() for h in _parse_row(rows[0], delim)]
    if not header:
        raise ValueError("The CSV has no columns.")

    time_info = _detect_time_column_info(header)
    time_col: Optional[str] = time_info[0] if time_info else None
    time_unit: str = time_info[1] if time_info else "ms"
    time_source = "column" if time_col is not None else "synthetic"
    synthetic_rate_hz = DEFAULT_SYNTHETIC_RATE_HZ if time_source == "synthetic" else 0.0
    if time_source == "synthetic":
        warnings.append(
            "No time column found (expected timestamp_ms or elapsed_ms); "
            "synthesized 10 Hz timing from row index. Add a timestamp column "
            "for accurate time-based features."
        )

    time_idx = header.index(time_col) if time_col is not None else None
    candidate_cols = [h for i, h in enumerate(header) if i != time_idx]
    context_columns = [c for c in candidate_cols if is_context_column(c)]
    sensor_candidates = [c for c in candidate_cols if c not in context_columns]
    if context_columns:
        warnings.append(
            f"Detected context column(s) kept as metadata, not scored: "
            f"{', '.join(context_columns)}."
        )

    # First pass: decide which columns are numeric enough to be channels.
    numeric = {c: 0 for c in sensor_candidates}
    for r in range(1, len(rows)):
        cells = _parse_row(rows[r], delim)
        for c in sensor_candidates:
            idx = header.index(c)
            if idx < len(cells) and _safe_float(cells[idx]) is not None:
                numeric[c] += 1
    channel_ids = [c for c in sensor_candidates if numeric[c] > 0]
    skipped_columns = [c for c in sensor_candidates if numeric[c] == 0]
    if skipped_columns:
        warnings.append(
            f"Non-numeric column(s) skipped: {', '.join(skipped_columns)}."
        )
    unknown_columns = [c for c in channel_ids if c not in MOX_CHANNEL_IDS]
    if unknown_columns:
        warnings.append(
            f"Column(s) not in the MOX set treated as sensor channels: "
            f"{', '.join(unknown_columns)}."
        )

    # A detected-but-unreadable time column must not reject the whole file.
    # Pre-scan: if no time cell parses, drop back to synthetic timing so every
    # data row is still adopted (with a warning teaching the expected format).
    if time_source == "column":
        parsed = 0
        checked = 0
        for r in range(1, len(rows)):
            if checked >= 25:
                break
            cells = _parse_row(rows[r], delim)
            if len(cells) != len(header):
                continue
            checked += 1
            if _parse_time_value(cells[time_idx], time_unit) is not None:
                parsed += 1
        if parsed == 0:
            time_source = "synthetic"
            synthetic_rate_hz = DEFAULT_SYNTHETIC_RATE_HZ
            warnings.append(
                f'Column "{time_col}" was not readable as time (expected ms, '
                f"epoch seconds, ISO datetime or HH:MM:SS); synthesized 10 Hz "
                f"timing instead."
            )
            time_col = None

    samples: List[ParsedSample] = []
    non_finite = 0
    unsorted = False

    for r in range(1, len(rows)):
        cells = _parse_row(rows[r], delim)
        if len(cells) != len(header):
            continue

        if time_source == "column":
            raw_time = _parse_time_value(cells[time_idx], time_unit)
            if raw_time is None:
                non_finite += 1
                continue
        else:
            raw_time = float(len(samples) * 100)

        values: dict[str, float] = {}
        row_has_non_finite = False
        for ch in channel_ids:
            col_idx = header.index(ch)
            raw = _safe_float(cells[col_idx])
            if raw is None:
                non_finite += 1
                row_has_non_finite = True
                continue
            values[ch] = raw
        if row_has_non_finite:
            continue

        for col in context_columns:
            col_idx = header.index(col)
            values[col] = _safe_float(cells[col_idx])

        samples.append(ParsedSample(time=raw_time, values=values))

    # Numeric time that clearly sits in epoch-seconds range (≈2001–2036) is scaled to ms.
    if time_source == "column" and samples:
        times = sorted(s.time for s in samples)
        median_time = times[len(times) // 2]
        if 1_200_000_000 <= median_time <= 4_000_000_000:
            for s in samples:
                s.time = s.time * 1000.0
            warnings.append(
                "Time column read as epoch seconds and converted to milliseconds."
            )

    for i in range(1, len(samples)):
        if samples[i].time < samples[i - 1].time:
            unsorted = True
            break

    if unsorted:
        samples.sort(key=lambda s: s.time)

    gaps = [samples[i + 1].time - samples[i].time for i in range(len(samples) - 1)]
    positive = [g for g in gaps if g > 0]
    median_gap = _median(positive)
    guess_sampling_rate_hz = (
        (1000.0 / median_gap) if median_gap else DEFAULT_SYNTHETIC_RATE_HZ
    )
    if time_source == "synthetic":
        guess_sampling_rate_hz = DEFAULT_SYNTHETIC_RATE_HZ

    return CsvParseResult(
        header=header,
        time_column=time_col,
        time_source=time_source,
        synthetic_rate_hz=synthetic_rate_hz,
        samples=samples,
        row_count=len(samples),
        channel_ids=channel_ids,
        context_columns=context_columns,
        unknown_columns=unknown_columns,
        skipped_columns=skipped_columns,
        guess_sampling_rate_hz=guess_sampling_rate_hz,
        non_finite=non_finite,
        unsorted=unsorted,
        warnings=warnings,
    )


def _safe_float(raw: str) -> Optional[float]:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def guess_sensor_type(header: List[str]) -> str:
    """Guess the sensor family from CSV column names (web guessSensorType)."""
    hits = [h for h in header if h in MOX_CHANNEL_IDS]
    return "mox" if len(hits) >= 2 else "unknown"
