"""CSV parsing for the .osmell data.csv member.

Mirrors `osmograph-web/lib/osmell/csv.ts` 1:1 so Python and TypeScript agree on
row filtering, quote handling, time-column detection and sampling-rate guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median as _py_median
from typing import List, Optional

from .types import TIME_COLUMNS, ParsedSample

MOX_CHANNEL_IDS = ("VOC", "Alcohol", "LPG", "CO", "NO2", "C2H5OH")


@dataclass
class CsvParseResult:
    header: List[str]
    time_column: str
    samples: List[ParsedSample]
    row_count: int
    channel_ids: List[str]
    guess_sampling_rate_hz: float
    non_finite: int
    unsorted: bool


def _parse_row(raw: str) -> List[str]:
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
        elif c == "," and not in_quotes:
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


def parse_csv(text: str) -> CsvParseResult:
    raw_rows = text.splitlines()
    rows = [r.strip() for r in raw_rows if r.strip() and not r.strip().startswith("#")]

    if not rows:
        raise ValueError("The CSV file is empty.")

    header = [h.strip() for h in _parse_row(rows[0])]
    if not header:
        raise ValueError("The CSV has no columns.")

    time_col = next((h for h in header if h in TIME_COLUMNS), None)
    if time_col is None:
        raise ValueError(f"No time column found. Expected one of: {', '.join(TIME_COLUMNS)}.")

    time_idx = header.index(time_col)
    channel_ids = [h for i, h in enumerate(header) if i != time_idx]

    samples: List[ParsedSample] = []
    non_finite = 0
    unsorted = False

    for r in range(1, len(rows)):
        cells = _parse_row(rows[r])
        if len(cells) != len(header):
            continue

        raw_time = _safe_float(cells[time_idx])
        if raw_time is None:
            non_finite += 1
            continue

        values: dict[str, float] = {}
        row_has_non_finite = False
        for ch in channel_ids:
            col_idx = header.index(ch)
            if col_idx < 0:
                continue
            raw = _safe_float(cells[col_idx])
            if raw is None:
                non_finite += 1
                row_has_non_finite = True
                continue
            values[ch] = raw
        if row_has_non_finite:
            continue

        samples.append(ParsedSample(time=raw_time, values=values))

    for i in range(1, len(samples)):
        if samples[i].time < samples[i - 1].time:
            unsorted = True
            break

    if unsorted:
        samples.sort(key=lambda s: s.time)

    gaps = [samples[i + 1].time - samples[i].time for i in range(len(samples) - 1)]
    positive = [g for g in gaps if g > 0]
    median_gap = _median(positive)
    guess_sampling_rate_hz = (1000.0 / median_gap) if median_gap else 0.0

    return CsvParseResult(
        header=header,
        time_column=time_col,
        samples=samples,
        row_count=len(samples),
        channel_ids=channel_ids,
        guess_sampling_rate_hz=guess_sampling_rate_hz,
        non_finite=non_finite,
        unsorted=unsorted,
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
