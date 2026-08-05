"""R0-contract tests (master doc §10.2/§10.3) and View A/B consumers.

Covers: explicit-baseline provenance, auto-R0 (median of first finite samples),
degenerate-channel guards, dead-channel detection (cv < 0.001), View A sharing a
single per-channel R0, View B parity fields, and the `sensor.calibration`
manifest contract (§10.10).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from conftest import make_file
from opensmell.mox.features import (
    _r0_from_contract,
    calibration_for_channel,
    compute_channel_device_agnostic,
    compute_channel_absolute,
    extract_all_framework_features,
    process_mox,
)
from opensmell.types import (
    BaselineDescriptor,
    CalibrationDescriptor,
    ChannelDescriptor,
    OsmellFile,
    OsmellManifest,
    SensorDescriptor,
    SessionDescriptor,
)


def make_auto_file(series_by_channel, r0_samples=15):
    ids = list(series_by_channel.keys())
    n = len(series_by_channel[ids[0]])
    manifest = OsmellManifest(
        osmell={"formatVersion": "1.0.0"},
        sensor=SensorDescriptor(
            sensor_type="mox",
            channels=[ChannelDescriptor(id=c, unit="adc") for c in ids],
            time_column="timestamp_ms",
        ),
        session=SessionDescriptor(role="exposure", label="contract"),
        baseline=BaselineDescriptor(source="auto", r0_samples=r0_samples),
    )
    return OsmellFile(manifest=manifest, time=[i * 100 for i in range(n)], data=series_by_channel)


def make_explicit_file(series_by_channel):
    ids = list(series_by_channel.keys())
    n = len(series_by_channel[ids[0]])
    manifest = OsmellManifest(
        osmell={"formatVersion": "1.0.0"},
        sensor=SensorDescriptor(
            sensor_type="mox",
            channels=[ChannelDescriptor(id=c, unit="adc") for c in ids],
            time_column="timestamp_ms",
        ),
        session=SessionDescriptor(role="baseline", label="contract-baseline"),
        baseline=BaselineDescriptor(source="explicit", file="baseline.csv"),
    )
    return OsmellFile(manifest=manifest, time=[i * 100 for i in range(n)], data=series_by_channel)


def test_auto_r0_median_of_first_finite_samples():
    series = [float("nan")] * 3 + [1000.0] * 40
    f = make_auto_file({"VOC": series})
    r0 = process_mox(f)["features"][0].r0
    assert r0 == pytest.approx(1000.0)


def test_auto_r0_ignores_leading_nan_via_contract_helper():
    series = np.array([float("nan")] * 3 + [1000.0] * 30)
    assert _r0_from_contract(series, 15) == pytest.approx(1000.0)
    assert _r0_from_contract(series, 15, r0=777.0) == pytest.approx(777.0)


def test_explicit_baseline_uses_entire_channel():
    series = [1000.0] * 10 + [3000.0] * 20
    f = make_explicit_file({"VOC": series})
    r0 = process_mox(f)["features"][0].r0
    assert r0 == pytest.approx(3000.0), "explicit R0 must be the median of the whole channel"


def test_auto_vs_explicit_r0_differ():
    series = [1000.0] * 10 + [3000.0] * 20
    auto = process_mox(make_auto_file({"VOC": series}))["features"][0].r0
    explicit = process_mox(make_explicit_file({"VOC": series}))["features"][0].r0
    assert auto == pytest.approx(1000.0)
    assert explicit == pytest.approx(3000.0)


def test_guard_median_window_all_negative_uses_one():
    series = np.array([-100.0] * 30)
    assert _r0_from_contract(series, 15) == pytest.approx(1.0)


def test_guard_median_zero_falls_back_to_mean_positive():
    series = np.array([0.0] * 10 + [100.0] * 20)
    assert _r0_from_contract(series, 15) == pytest.approx(100.0)


def test_guard_empty_channel_uses_one():
    assert _r0_from_contract(np.array([]), 15) == pytest.approx(1.0)


def test_dead_channel_cv_threshold():
    flat = [1000.0] * 60
    da = compute_channel_device_agnostic(np.asarray(flat), r0_samples=15)
    assert da["is_dead"] is True
    assert da["relative_amplitude"] == 0.0


def test_dead_channel_fewer_than_two_finite():
    da = compute_channel_device_agnostic(np.asarray([float("nan")] * 30), r0_samples=15)
    assert da["is_dead"] is True


def test_view_a_shares_single_r0_per_channel():
    base = np.array([1000.0] * 20 + [3000.0] * 20)
    data = np.column_stack([base, base + 100.0])
    feats = extract_all_framework_features(data, r0_samples=15)
    assert feats["ch0_abs_baseline_resistance"] == pytest.approx(1000.0)
    assert feats["ch1_abs_baseline_resistance"] == pytest.approx(1100.0)


def test_view_a_honors_explicit_r0():
    data = np.array([1000.0] * 20 + [3000.0] * 20).reshape(-1, 1)
    auto = extract_all_framework_features(data, r0_samples=15)
    explicit = extract_all_framework_features(data, r0_samples=15, r0_per_channel={0: 2000.0})
    assert auto["ch0_abs_baseline_resistance"] == pytest.approx(1000.0)
    assert explicit["ch0_abs_baseline_resistance"] == pytest.approx(2000.0)
    assert explicit["ch0_da_relative_amplitude"] == pytest.approx(0.5)
    assert explicit["ch0_abs_calibrated_concentration"] != auto["ch0_abs_calibrated_concentration"]


def test_view_b_parity_fields_computed():
    f = make_file()
    res = process_mox(f)
    for feat in res["features"]:
        assert feat.dead is False
        assert feat.decay_time_ms is not None, "View B decay_time_ms must be wired (§10.9)"
        assert feat.endpoint_delta == pytest.approx(0.0, abs=1e-6)
        assert 0.0 <= feat.saturation_index <= 1.0


def test_view_b_flat_channel_zeroes_parity_fields():
    f = make_auto_file({"VOC": [1000.0] * 60})
    feat = process_mox(f)["features"][0]
    assert feat.dead is True
    assert feat.relative_amplitude == 0.0
    assert feat.decay_time_ms is None
    assert feat.endpoint_delta == 0.0
    assert feat.saturation_index == 0.0


def test_calibration_roundtrip_and_consumer():
    manifest = OsmellManifest(
        osmell={"formatVersion": "1.0.0"},
        sensor=SensorDescriptor(
            sensor_type="mox",
            channels=[ChannelDescriptor(id="VOC", unit="adc"),
                      ChannelDescriptor(id="CO", unit="adc")],
            time_column="timestamp_ms",
            calibration={"VOC": CalibrationDescriptor(
                a=0.5, b=-0.6, reference_substance="ethanol",
                reference_ppm=50.0, method="two-point",
            )},
        ),
        session=SessionDescriptor(role="exposure", label="cal"),
        baseline=BaselineDescriptor(source="auto", r0_samples=15),
    )
    import opensmell as o
    blob = o.build_osmell(OsmellFile(manifest=manifest, time=[0.0], data={"VOC": [1000.0], "CO": [1000.0]}))
    parsed = o.parse_osmell(blob)
    cal = parsed.manifest.sensor.calibration
    assert cal is not None
    assert cal["VOC"].a == pytest.approx(0.5)
    assert cal["VOC"].reference_substance == "ethanol"
    assert cal["VOC"].method == "two-point"
    assert "CO" not in cal

    a, b, src = calibration_for_channel(parsed.manifest, "VOC")
    assert (a, b, src) == (0.5, -0.6, "manifest")
    a, b, src = calibration_for_channel(parsed.manifest, "CO")
    assert (a, b, src) == (1.0, -0.5, "nominal-default")


def test_calibration_changes_concentration():
    series = np.array([1000.0] * 20 + [1200.0] * 10)
    r0 = 1000.0
    nominal = compute_channel_absolute(series, r0=r0)
    manifest = compute_channel_absolute(series, r0=r0, a_const=0.5, b_const=-0.6)
    assert nominal["calibrated_concentration"] != manifest["calibrated_concentration"]
    assert nominal["baseline_resistance"] == manifest["baseline_resistance"]


def test_calibration_wired_through_view_a():
    data = np.array([1000.0] * 20 + [1200.0] * 10).reshape(-1, 1)
    nominal = extract_all_framework_features(data, r0_samples=15)
    calibrated = extract_all_framework_features(
        data, r0_samples=15, calibration={0: {"a": 0.5, "b": -0.6}}
    )
    assert nominal["ch0_abs_calibrated_concentration"] != calibrated["ch0_abs_calibrated_concentration"]
