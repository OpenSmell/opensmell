"""MOX quality scoring on the new .osmell path."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import opensmell as o
from conftest import make_file
from opensmell.types import OsmellFile, OsmellManifest


def _quality(f, **kw):
    sample_count = kw.pop("sample_count", len(f.time))
    return o.compute_quality(
        f,
        sample_count=sample_count,
        guess_sampling_rate_hz=10.0,
        **kw,
    )


def test_good_exposure_scores_high():
    q = _quality(make_file())
    assert q.format == "opensmell-quality"
    assert q.badge == "Excellent"
    assert q.total >= 90


def test_dead_sensor_flagged():
    f = make_file(dead_channels=["NO2"])
    q = _quality(f)
    assert "NO2" in q.flags.dead_sensors
    assert any("Dead sensors" in n for n in q.notes)


def test_single_role_skips_signal_strength():
    f = make_file(role="single")
    q = _quality(f)
    assert q.subscores["signalStrength"].value is None
    assert q.total is not None, "total must be renormalized over available sub-scores"


def test_no_baseline_scores_zero_stability():
    f = make_file()
    f.manifest.baseline.source = "none"
    q = _quality(f)
    assert q.subscores["baselineStability"].value == 0
    assert q.subscores["baselineStability"].reason == "no_baseline"


def test_short_duration_penalized():
    q = _quality(make_file(n=60))
    assert q.subscores["durationAdequacy"].reason == "too_short"


def test_miris_routing_not_implemented():
    f = make_file()
    f.manifest.sensor.sensor_type = "miris"
    with pytest.raises(NotImplementedError, match="miris"):
        _quality(f)


def test_unknown_sensor_type_not_implemented():
    f = make_file()
    f.manifest.sensor.sensor_type = "alien"
    with pytest.raises(NotImplementedError, match="alien"):
        _quality(f)
