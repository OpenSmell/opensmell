"""Bulk ingestion — folders of CSVs into labeled, scored collections.

Handles the two de-facto structures people actually keep:
  - folder = substance, one or more CSV sessions inside (SmellNet style), and
  - a loose pile of files (label taken from the file name).

Every file is adopted (never rejected on structure): columns are classified,
missing time becomes explicit synthetic timing, context columns (environmental
readings) are preserved as metadata, and each session carries a quality report
plus the parse warnings so a review UI can show exactly how the data was
interpreted and what to improve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .csv import parse_csv
from .features import run_processor
from .quality import compute_quality
from .types import (
    ChannelDescriptor,
    OsmellFile,
    OsmellManifest,
    QualityReport,
    SensorDescriptor,
    SessionDescriptor,
)

CSV_SUFFIXES = (".csv", ".txt")


@dataclass
class IngestedSession:
    source: str
    substance: str
    label: str
    ok: bool = False
    file: Optional[OsmellFile] = None
    report: Optional[QualityReport] = None
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def sensor_type(self) -> str:
        return self.file.manifest.sensor.sensor_type if self.file else "unknown"

    @property
    def time_source(self) -> str:
        extra = self.file.manifest.extra.get("ingest", {}) if self.file else {}
        return extra.get("timeSource", "column")


@dataclass
class IngestedCollection:
    substances: Dict[str, List[IngestedSession]] = field(default_factory=dict)

    def session_count(self) -> int:
        return sum(len(v) for v in self.substances.values())

    def ok_count(self) -> int:
        return sum(1 for s in self.iter_sessions() if s.ok)

    def iter_sessions(self):
        for substance in sorted(self.substances):
            for s in self.substances[substance]:
                yield s


def _sensor_type(parsed) -> str:
    from .csv import guess_sensor_type

    return guess_sensor_type(parsed.header)


def build_osmell_file(
    parsed,
    label: str,
    substance: str,
    source: str,
    role: str = "single",
) -> OsmellFile:
    """Normalize a parsed CSV into an OsmellFile with ingest provenance.

    Only sensor channels (MOX + unknown numeric columns) are declared as
    `sensor.channels` so features and quality never score environmental
    context. Context values are preserved losslessly in the manifest's ingest
    metadata block.
    """
    sensor_type = _sensor_type(parsed)
    manifest = OsmellManifest(
        osmell={"formatVersion": "1.0.0"},
        sensor=SensorDescriptor(
            sensor_type=sensor_type,
            channels=[ChannelDescriptor(id=cid, unit="adc") for cid in parsed.channel_ids],
            sampling_rate_hz=parsed.guess_sampling_rate_hz or None,
            time_column=parsed.time_column or "synthetic_index",
        ),
        session=SessionDescriptor(
            role=role,
            label=label,
            group_id=substance,
            duration_ms=int(
                (parsed.samples[-1].time - parsed.samples[0].time)
                if len(parsed.samples) > 1 else 0
            ),
            notes="; ".join(parsed.warnings) or None,
        ),
        software={"importer": "opensmell-ingest"},
        extra={
            "ingest": {
                "sourceFile": source,
                "timeSource": parsed.time_source,
                "syntheticRateHz": parsed.synthetic_rate_hz or None,
                "timeColumn": parsed.time_column,
                "contextColumns": parsed.context_columns,
                "unknownColumns": parsed.unknown_columns,
                "skippedColumns": parsed.skipped_columns,
                "warnings": parsed.warnings,
                "context": {
                    col: [s.values.get(col) for s in parsed.samples]
                    for col in parsed.context_columns
                },
            }
        },
    )
    data = {
        cid: [s.values.get(cid, float("nan")) for s in parsed.samples]
        for cid in parsed.channel_ids
    }
    return OsmellFile(manifest=manifest, time=[s.time for s in parsed.samples], data=data)


def ingest_file(path, substance: Optional[str] = None, role: str = "single") -> IngestedSession:
    """Ingest a single CSV/TXT file. Never raises for structure problems."""
    p = Path(path)
    label = p.stem
    substance = (substance or "").strip() or (p.parent.name if p.parent.name != "." else label)
    session = IngestedSession(source=str(p), substance=substance, label=label)
    try:
        parsed = parse_csv(p.read_text(encoding="utf-8", errors="replace"))
        if parsed.row_count == 0:
            session.error = "No usable data rows found."
            return session
        if not parsed.channel_ids:
            session.error = "No numeric sensor columns found."
            return session
        file = build_osmell_file(parsed, label=label, substance=substance, source=str(p), role=role)
        report = compute_quality(
            file,
            sample_count=parsed.row_count,
            guess_sampling_rate_hz=parsed.guess_sampling_rate_hz,
            unsorted=parsed.unsorted,
            non_finite=parsed.non_finite,
        )
        session.ok = True
        session.file = file
        session.report = report
        session.warnings = parsed.warnings
    except Exception as e:  # adopt-don't-reject: any failure is reported, not fatal
        session.error = str(e)
    return session


def ingest_folder(path, recurse: bool = True, label_from_dir: bool = True) -> IngestedCollection:
    """Ingest a folder of recordings, grouping by sub-folder = substance.

    Files directly in the root are grouped under the folder name. CSV suffix
    is case-insensitive.
    """
    root = Path(path)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a folder: {path}")

    groups: Dict[str, List[Path]] = {}
    if recurse:
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in CSV_SUFFIXES:
                rel = p.relative_to(root)
                parent = rel.parent
                substance = parent.name if label_from_dir and parent != Path(".") else root.name
                groups.setdefault(substance, []).append(p)
    else:
        for p in sorted(root.iterdir()):
            if p.is_file() and p.suffix.lower() in CSV_SUFFIXES:
                groups.setdefault(root.name, []).append(p)

    collection = IngestedCollection()
    for substance, paths in groups.items():
        collection.substances[substance] = [
            ingest_file(p, substance=substance) for p in paths
        ]
    return collection
