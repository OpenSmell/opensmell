"""Sensor profile detection and modality boundaries."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

import opensmell as o
from conftest import make_file


def _write_mq_csv(path):
    path.write_text(
        "timestamp_ms,MQ135,MQ3,MQ6\n"
        "0,100,200,300\n"
        "100,110,210,310\n"
        "200,120,220,320\n"
    )


def _write_unknown_six_csv(path):
    names = [f"S{i}" for i in range(1, 7)]
    path.write_text(
        "time," + ",".join(names) + "\n"
        + "\n".join(
            f"{i * 100}," + ",".join(str(1000 + i + j) for j in range(6))
            for i in range(120)
        )
        + "\n"
    )


def test_inspect_csv_resolves_mq_models(tmp_path):
    path = tmp_path / "mq.csv"
    _write_mq_csv(path)
    profile = o.inspect_source(path)
    assert profile.sensor_type == "mox"
    assert profile.feature_family == "mox"
    assert profile.supported
    assert profile.models["MQ135"] == "MQ-135"
    assert profile.models["MQ3"] == "MQ-3"
    assert profile.unmodeled_channels == []


def test_inspect_unknown_array_is_mox_compatible(tmp_path):
    path = tmp_path / "six.csv"
    _write_unknown_six_csv(path)
    profile = o.inspect_source(path)
    assert profile.sensor_type == "unknown"
    assert profile.feature_family == "mox-compatible"
    assert profile.supported
    assert profile.inferred
    assert any("MOX-compatible" in w for w in profile.warnings)

    features, names = o.extract_features(path)
    assert features.shape[1] == 187 == len(names)


def test_inspect_channel_selection(tmp_path):
    path = tmp_path / "six.csv"
    _write_unknown_six_csv(path)
    profile = o.inspect_source(path, channels={"S1": "a", "S3": "b"})
    assert profile.channel_ids == ["S1", "S3"]
    assert profile.unmodeled_channels == ["S1", "S3"]

    with pytest.raises(ValueError, match="Requested channel"):
        o.inspect_source(path, channels=["NOPE"])


def test_inspect_ingest_session(tmp_path):
    path = tmp_path / "mq.csv"
    _write_mq_csv(path)
    session = o.ingest_file(path)
    assert session.ok
    profile = o.inspect_source(session)
    assert profile.sensor_type == "mox"
    assert profile.source == str(path)


def test_declared_electrochemical_is_unsupported():
    file = make_file()
    file.manifest.sensor.sensor_type = "electrochemical"
    profile = o.inspect_source(file)
    assert profile.sensor_type == "electrochemical"
    assert profile.feature_family == "unsupported"
    assert not profile.supported

    # Raw normalization is still valid; inheriting MOX features is not.
    assert o.load_recording(file).shape == (600, 6)
    with pytest.raises(o.UnsupportedSensorTypeError, match="electrochemical"):
        o.extract_features(file)
    with pytest.raises(o.UnsupportedSensorTypeError, match="electrochemical"):
        o.process(file)


def test_electrochemical_csv_is_refused(tmp_path):
    path = tmp_path / "ec.csv"
    path.write_text(
        "timestamp_ms,working_electrode_ua,reference_electrode_mv\n"
        "0,1.0,0.4\n"
        "100,1.1,0.4\n"
        "200,1.2,0.4\n"
    )
    profile = o.inspect_source(path)
    assert profile.sensor_type == "electrochemical"
    assert profile.feature_family == "unsupported"
    with pytest.raises(o.UnsupportedSensorTypeError):
        o.extract_features(path)


def test_profile_serializes_camel_case():
    profile = o.inspect_source(make_file())
    data = profile.to_dict()
    assert data["sensorType"] == "mox"
    assert data["featureFamily"] == "mox"
    assert data["channelIds"] == ["VOC", "Alcohol", "LPG", "CO", "NO2", "C2H5OH"]
    assert isinstance(data["warnings"], list)


def test_unknown_array_features_finite(tmp_path):
    path = tmp_path / "six.csv"
    _write_unknown_six_csv(path)
    features, _ = o.extract_features(path)
    assert np.all(np.isfinite(features))
