"""Read/write `.osmell` bundles (ZIP: manifest.json + data.csv + optional events.json).

Mirrors `osmograph-web/lib/osmell/io.ts` 1:1. The MIME type of the web bundle is
`application/vnd.opensmell.osmell`; here we operate on raw bytes via stdlib
`zipfile` so the Python SDK has no archive dependency.
"""

from __future__ import annotations

import io as _io
import zipfile
from pathlib import Path
from typing import Optional

from .csv import parse_csv
from .types import OsmellFile, OsmellManifest, SessionEvent

OSMELL_MIME_TYPE = "application/vnd.opensmell.osmell"


def parse_osmell(data: bytes) -> OsmellFile:
    """Parse an in-memory `.osmell` bundle into an OsmellFile."""
    with zipfile.ZipFile(_io.BytesIO(data)) as zf:
        names = set(zf.namelist())
        if "manifest.json" not in names or "data.csv" not in names:
            raise ValueError("Not a valid .osmell file: missing manifest.json or data.csv.")

        manifest = OsmellManifest.from_dict(_load_json(zf, "manifest.json"))
        csv = parse_csv(zf.read("data.csv").decode("utf-8"))

        if csv.row_count == 0:
            raise ValueError("The .osmell data.csv is empty.")

        expected = {c.id for c in manifest.sensor.channels}
        for cid in csv.channel_ids:
            if cid not in expected:
                raise ValueError(f'data.csv has column "{cid}" not declared in the manifest.')
        for c in manifest.sensor.channels:
            if c.id not in csv.channel_ids:
                raise ValueError(f'Manifest channel "{c.id}" is missing from data.csv.')

        time = [s.time for s in csv.samples]
        data = {cid: [s.values.get(cid, float("nan")) for s in csv.samples] for cid in csv.channel_ids}

        events: Optional[list[SessionEvent]] = None
        if "events.json" in names:
            raw = _load_json(zf, "events.json")
            events = [SessionEvent.from_dict(e) for e in raw]

    return OsmellFile(manifest=manifest, time=time, data=data, events=events)


def parse_osmell_file(path: str | Path) -> OsmellFile:
    """Load a `.osmell` bundle from disk."""
    return parse_osmell(Path(path).read_bytes())


def csv_from_file(file: OsmellFile) -> str:
    """Serialize the data channel of an OsmellFile back to CSV text."""
    channel_ids = [c.id for c in file.manifest.sensor.channels]
    time_column = file.manifest.sensor.time_column
    lines = [",".join([time_column, *channel_ids])]
    for i in range(len(file.time)):
        row = [str(file.time[i])]
        for cid in channel_ids:
            v = file.data.get(cid, [None])[i] if i < len(file.data.get(cid, [])) else None
            row.append(str(v) if v == v and v is not None else "")
        lines.append(",".join(row))
    return "\n".join(lines) + "\n"


def build_osmell(file: OsmellFile) -> bytes:
    """Serialize an OsmellFile to an in-memory `.osmell` bundle (DEFLATE)."""
    buf = _io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        _write_json(zf, "manifest.json", file.manifest.to_dict())
        zf.writestr("data.csv", csv_from_file(file))
        if file.events:
            _write_json(zf, "events.json", [e.to_dict() for e in file.events])
    return buf.getvalue()


def write_osmell(file: OsmellFile, path: str | Path) -> Path:
    """Write an OsmellFile to disk as a `.osmell` bundle."""
    p = Path(path)
    p.write_bytes(build_osmell(file))
    return p


def default_file_name(file: OsmellFile, role: Optional[str] = None) -> str:
    """Name suggestion: `<label>_<role>_<date>.osmell` (web defaultFileName)."""
    session = file.manifest.session
    if role is None:
        role = session.role if session else "single"
    label = (session.label or "recording") if session else "recording"
    import re

    label = re.sub(r"[^a-z0-9_\-]+", "-", label, flags=re.IGNORECASE)
    recorded = (session.recorded_at[:10]) if session and session.recorded_at else ""
    return f"{label}_{role}_{recorded}.osmell"


def _load_json(zf: zipfile.ZipFile, name: str):
    import json

    return json.loads(zf.read(name).decode("utf-8"))


def _write_json(zf: zipfile.ZipFile, name: str, obj) -> None:
    import json

    zf.writestr(name, json.dumps(obj, indent=2))
