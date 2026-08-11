"""Tolerant CSV import and bulk folder ingestion (adopt-don't-reject)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import opensmell as o
from opensmell.ingest import build_osmell_file

SMELLNET_CSV = (
    "NO2,C2H5OH,VOC,CO,Alcohol,LPG,Benzene,Temperature,Pressure,Humidity,"
    "Gas_Resistance,Altitude\n"
    "153,188,434,771,1,18,0,24.73,1007.1,29.79,290.6,82.45\n"
    "152,188,432,771,1,18,0,24.74,1007.1,29.9,291.37,82.45\n"
    "151,187,430,770,1,18,0,24.75,1007.1,29.95,291.9,82.45\n"
)


def test_smellnet_csv_ingests_without_time_column(tmp_path):
    p = tmp_path / "cinnamon" / "cinnamon.376616d4c817.csv.csv"
    p.parent.mkdir(parents=True)
    p.write_text(SMELLNET_CSV)

    session = o.ingest_file(p)
    assert session.ok, session.error
    assert session.substance == "cinnamon"
    assert session.sensor_type == "mox"
    assert session.time_source == "synthetic"

    f = session.file
    assert f.manifest.sensor.time_column == "synthetic_index"
    assert [c.id for c in f.manifest.sensor.channels] == [
        "NO2", "C2H5OH", "VOC", "CO", "Alcohol", "LPG", "Benzene",
    ]
    # Context columns preserved but never scored as channels.
    ingest = f.manifest.extra["ingest"]
    assert ingest["contextColumns"] == [
        "Temperature", "Pressure", "Humidity", "Gas_Resistance", "Altitude",
    ]
    assert "Temperature" not in f.data
    assert len(ingest["context"]["Temperature"]) == 3
    assert ingest["context"]["Temperature"] == [24.73, 24.74, 24.75]
    assert ingest["context"]["Humidity"][0] == 29.79

    assert session.report is not None
    assert session.report.total is not None
    assert session.report.badge in ("Excellent", "Good", "Fair", "Poor")
    # Synthetic timing must be visible in the manifest + quality notes.
    assert ingest["timeSource"] == "synthetic"
    assert session.report.notes


def test_ingest_folder_groups_by_substance(tmp_path):
    for substance in ("cinnamon", "garlic"):
        d = tmp_path / substance
        d.mkdir()
        for i in range(2):
            (d / f"{substance}_{i}.csv").write_text(SMELLNET_CSV)
    # A loose file in the root.
    (tmp_path / "loose.csv").write_text(SMELLNET_CSV)

    collection = o.ingest_folder(tmp_path)
    assert set(collection.substances) == {"cinnamon", "garlic", tmp_path.name}
    assert collection.session_count() == 5
    assert collection.ok_count() == 5
    for session in collection.iter_sessions():
        assert session.ok


def test_ingest_file_reports_failure_not_exception(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("VOC,Alcohol\n")
    session = o.ingest_file(p)
    assert not session.ok
    assert session.error is not None


def test_ingest_osmell_roundtrip_preserves_provenance(tmp_path):
    p = tmp_path / "cinnamon.csv"
    p.write_text(SMELLNET_CSV)
    session = o.ingest_file(p)
    blob = o.build_osmell(session.file)
    f2 = o.parse_osmell(blob)
    assert f2.manifest.extra["ingest"]["timeSource"] == "synthetic"
    assert f2.manifest.extra["ingest"]["contextColumns"]
    assert f2.manifest.software["importer"] == "opensmell-ingest"
    assert f2.time == session.file.time
    assert f2.data == session.file.data


def test_build_osmell_file_role_and_provenance(tmp_path):
    parsed = o.parse_csv("timestamp_ms,VOC,Alcohol\n0,100,200\n")
    f = build_osmell_file(parsed, label="a", substance="b", source="x.csv", role="exposure")
    assert f.manifest.session.role == "exposure"
    assert f.manifest.session.group_id == "b"
    assert f.manifest.sensor.time_column == "timestamp_ms"


def test_mq_array_adopted_with_unknown_sensor_type(tmp_path):
    """An MQ-series array isn't a MOX set but must still be adopted and scored."""
    p = tmp_path / "mq.csv"
    p.write_text(
        "timestamp_ms,MQ135,MQ3,MQ6,MQ7,MQ4,MQ8\n"
        "0,19789,19789,19789,19789,19789,19789\n"
        "100,20334,20334,20334,20334,20334,20334\n"
        "200,20727,20727,20727,20727,20727,20727\n"
        "300,21240,21240,21240,21240,21240,21240\n"
    )
    session = o.ingest_file(p)
    assert session.ok, session.error
    assert session.sensor_type == "unknown"
    assert session.time_source == "column"
    assert session.report is not None
    assert session.report.total is not None
    assert [c.id for c in session.file.manifest.sensor.channels] == [
        "MQ135", "MQ3", "MQ6", "MQ7", "MQ4", "MQ8",
    ]
