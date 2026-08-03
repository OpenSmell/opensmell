"""`.osmell` bundle I/O: CSV parsing, zip round-trip, validation, naming."""

import io
import os
import sys
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import opensmell as o
from conftest import CHANNELS, make_file, make_manifest
from opensmell.types import (
    BaselineDescriptor,
    ChannelDescriptor,
    OsmellFile,
    OsmellManifest,
    SensorDescriptor,
    SessionDescriptor,
    SessionEvent,
)


def test_parse_csv_spec_compliant():
    text = "timestamp_ms,VOC,Alcohol\n0,100,200\n100,110,210\n"
    parsed = o.parse_csv(text)
    assert parsed.channel_ids == ["VOC", "Alcohol"]
    assert parsed.row_count == 2
    assert parsed.time_column == "timestamp_ms"
    assert parsed.guess_sampling_rate_hz == 10.0
    assert parsed.unsorted is False


def test_parse_csv_skips_comments_and_empty_lines():
    text = "# header comment\n\n# another\ntimestamp_ms,VOC\n0,100\n"
    parsed = o.parse_csv(text)
    assert parsed.channel_ids == ["VOC"]
    assert parsed.row_count == 1


def test_parse_csv_sorts_unsorted_rows():
    text = "timestamp_ms,VOC\n200,120\n100,110\n300,130\n"
    parsed = o.parse_csv(text)
    assert parsed.unsorted is True
    assert [s.time for s in parsed.samples] == [100, 200, 300]


def test_parse_csv_rejects_missing_time_column():
    with pytest.raises(ValueError, match="time column"):
        o.parse_csv("VOC,Alcohol\n100,200\n")


def test_parse_csv_quoted_fields():
    text = 'timestamp_ms,VOC,Alcohol\n0,"110",200\n100,"1,000",210\n'
    parsed = o.parse_csv(text)
    assert parsed.samples[0].values["VOC"] == 110.0
    assert parsed.row_count == 1, "comma-thousands is non-finite in JS parity and must be skipped"


def test_build_parse_roundtrip():
    f = make_file()
    blob = o.build_osmell(f)
    assert blob.startswith(b"PK"), "must be a zip"
    f2 = o.parse_osmell(blob)
    assert f2.time == f.time
    assert f2.data == f.data
    assert f2.manifest.session.role == "exposure"
    assert f2.manifest.baseline.source == "auto"


def test_events_roundtrip():
    f = make_file()
    f.events = [
        SessionEvent(label="baseline", start_ms=0, end_ms=30000),
        SessionEvent(label="exposure", start_ms=30000),
    ]
    f2 = o.parse_osmell(o.build_osmell(f))
    assert [e.label for e in f2.events] == ["baseline", "exposure"]
    assert f2.events[0].end_ms == 30000


def _raw_zip(manifest_dict, csv_text):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        import json

        zf.writestr("manifest.json", json.dumps(manifest_dict))
        zf.writestr("data.csv", csv_text)
    return buf.getvalue()


def test_parse_rejects_column_not_in_manifest():
    manifest = {
        "osmell": {"formatVersion": "1.0.0"},
        "sensor": {
            "sensorType": "mox",
            "timeColumn": "timestamp_ms",
            "channels": [{"id": "VOC", "unit": "adc"}, {"id": "Alcohol", "unit": "adc"}],
        },
        "session": {"role": "exposure", "label": "cinnamon"},
    }
    blob = _raw_zip(manifest, "timestamp_ms,VOC,Alcohol,EXTRA\n0,100,200,999\n")
    with pytest.raises(ValueError, match="not declared"):
        o.parse_osmell(blob)


def test_parse_rejects_manifest_channel_missing():
    manifest = {
        "osmell": {"formatVersion": "1.0.0"},
        "sensor": {
            "sensorType": "mox",
            "timeColumn": "timestamp_ms",
            "channels": [
                {"id": "VOC", "unit": "adc"},
                {"id": "Alcohol", "unit": "adc"},
                {"id": "GONE", "unit": "adc"},
            ],
        },
        "session": {"role": "exposure", "label": "cinnamon"},
    }
    blob = _raw_zip(manifest, "timestamp_ms,VOC,Alcohol\n0,100,200\n")
    with pytest.raises(ValueError, match="missing from data.csv"):
        o.parse_osmell(blob)


def test_parse_rejects_missing_members():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("manifest.json", "{}")
    with pytest.raises(ValueError, match="manifest.json or data.csv"):
        o.parse_osmell(buf.getvalue())


def test_csv_from_file_and_naming():
    f = make_file()
    text = o.csv_from_file(f)
    first = text.strip().split("\n")[0]
    assert first == "timestamp_ms," + ",".join(CHANNELS)
    name = o.default_file_name(f)
    assert name == "cinnamon_exposure_2026-01-15.osmell"


def test_guess_sensor_type():
    assert o.guess_sensor_type(["timestamp_ms", "VOC", "Alcohol", "CO"]) == "mox"
    assert o.guess_sensor_type(["timestamp_ms", "onlyone"]) == "unknown"
