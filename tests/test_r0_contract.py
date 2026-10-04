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
    compute_channel_absolute,
    compute_channel_device_agnostic,
    compute_channel_health,
    extract_all_framework_features,
    process_mox,
)
from opensmell.types import (
    R0_WINDOW_FRACTION,
    R0_WINDOW_MAX_SAMPLES,
    R0_WINDOW_MIN_SAMPLES,
    BaselineDescriptor,
    CalibrationDescriptor,
    ChannelDescriptor,
    OsmellFile,
    OsmellManifest,
    SensorDescriptor,
    SessionDescriptor,
    r0_window_samples,
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


# --- R0 window contract (SAMPLING_CONTRACT.md, "The R0 window contract") ---
#
# `r0_samples=None` (or 0) means "nothing declared" and resolves to
# `clamp(floor(0.15 * n), 5, 30)`. These tests pin that resolution, its cadence
# behaviour, and the boundaries where cadence invariance provably stops. The same
# facts are asserted in the JS and Rust suites so the SDKs cannot drift again.

CADENCES = (1.0, 2.0, 10.0, 100.0)
# 0.15 * n is inside [5, 30] exactly for 34 <= n <= 200.
FRACTION_REGION = (40, 80, 160)


def _exposure(fs, plateau_s=12.0, duration_s=60.0):
    """A flat clean-air plateau then a monotone exposure, sampled at `fs` Hz.

    The plateau is 12 s so that the default window lies wholly inside it at every
    cadence under test (the widest is 9 samples at 1 Hz). Deterministic and
    finite, so the three SDKs can be handed identical samples without sharing a
    random seed.
    """
    n = int(round(duration_s * fs)) + 1
    t = np.arange(n) / fs
    return 1000.0 + 50.0 * np.clip((t - plateau_s) / 10.0, 0.0, 1.0)


def test_r0_window_is_a_floored_capped_fraction():
    # Floor: a 20-sample window would be 3 samples at 15% — too few for a stable
    # median — so the floor of 5 binds.
    assert r0_window_samples(1) == R0_WINDOW_MIN_SAMPLES
    assert r0_window_samples(20) == R0_WINDOW_MIN_SAMPLES
    # Fraction region, 34 <= n <= 200.
    assert r0_window_samples(60) == 9
    assert r0_window_samples(100) == 15  # the canonical DEFAULT_WINDOW_SIZE
    assert r0_window_samples(200) == 30
    # Ceiling: an unbounded 15% of 600 would be 90 samples and would swallow the
    # onset on a long recording.
    assert r0_window_samples(600) == R0_WINDOW_MAX_SAMPLES
    assert r0_window_samples(18000) == R0_WINDOW_MAX_SAMPLES


def test_declared_window_wins_verbatim():
    assert r0_window_samples(600, 15) == 15
    assert r0_window_samples(60, 180) == 180
    # 0 is not a meaningful window: it is the "not declared" sentinel, shared with
    # the Rust `R0_WINDOW_DEFAULT`, so the three SDKs agree on what 0 means.
    assert r0_window_samples(100, 0) == 15
    assert r0_window_samples(100, None) == 15


def test_default_window_spans_15_percent_of_seconds_at_every_cadence():
    """In the fraction region the window covers 0.15 * T seconds at any rate.

    A fixed 15-sample default failed this: 1.5 s at 10 Hz against 15 s at 1 Hz.
    The fraction holds because the sample count grows with the rate, so the
    baseline covers the same physical share of the recording either way.
    """
    for fs in CADENCES:
        for n in FRACTION_REGION:
            window = r0_window_samples(n)
            duration_s = n / fs
            assert window / fs == pytest.approx(0.15 * duration_s, rel=1e-9), (
                f"{n} samples at {fs:g} Hz: window covers {window / fs:g} s of a "
                f"{duration_s:g} s recording, expected {0.15 * duration_s:g} s"
            )


def test_default_window_is_length_dependent_not_cadence_dependent():
    """More samples at the same cadence means a wider window; that is the point.

    The old fixed-15 default had neither property: the window tracked neither the
    recording length nor the recording duration.
    """
    for fs in CADENCES:
        widths = [r0_window_samples(n) / fs for n in FRACTION_REGION]
        assert widths == sorted(widths), f"{fs:g} Hz: {widths}"
        assert len(set(widths)) == len(widths)


def test_clamps_are_documented_cadence_dependence():
    """The clamps are sample counts, so they break invariance outside 34..200.

    60 s at 1 Hz is 61 samples: the fraction binds and the window covers 9 s. The
    same 60 s at 100 Hz is 6001 samples: the ceiling binds and it covers 0.3 s.
    This is the contract's stated limitation, not an invariant, and it is why a
    caller who needs the same baseline duration at every cadence must declare it.
    """
    slow = r0_window_samples(61) / 1.0
    fast = r0_window_samples(6001) / 100.0
    assert slow == pytest.approx(9.0)
    assert fast == pytest.approx(0.3)
    assert slow / fast == pytest.approx(30.0)


def test_declared_window_restores_cadence_invariance():
    """Rule 5's path: the declaring party converts a duration to a count."""
    duration_s = 60.0
    spans = {fs: r0_window_samples(int(round(duration_s * fs)) + 1,
                                   int(round(0.15 * duration_s * fs))) / fs
             for fs in CADENCES}
    for fs, span in spans.items():
        assert span == pytest.approx(0.15 * duration_s, rel=1e-9), f"{fs:g} Hz -> {span} s"


def test_r0_median_tracks_the_plateau_not_the_window():
    """R0 must be the clean-air level at 1, 2, 10 and 100 Hz alike.

    The four cadences resolve four different windows (9, 18, 30, 30 samples) and
    all four must read the same physical baseline. Under a fixed 15-sample window
    the 1 Hz and 100 Hz cases would have averaged over 15 s and 0.15 s of the
    same 60 s recording and disagreed by the response amplitude.
    """
    for fs in CADENCES:
        series = _exposure(fs)
        assert _r0_from_contract(series, None) == pytest.approx(1000.0)


def test_fixed_15_would_have_been_cadence_dependent():
    """The superseded default, kept as the regression this contract exists for."""
    spans = {fs: 15.0 / fs for fs in CADENCES}
    assert spans[10.0] == pytest.approx(1.5)
    assert spans[1.0] == pytest.approx(15.0)
    assert spans[1.0] / spans[100.0] == pytest.approx(100.0), "100x rescale across cadence"


def test_short_plateau_needs_a_declared_window():
    """The limit the fraction does *not* fix, stated rather than papered over.

    With a 1 s plateau in a 60 s recording the plateau is shorter than 15% of the
    recording, so no fraction-based window can find it: the 1 Hz window reaches
    9 s and lands on the ramp. A declared window — rule 5, ``round(duration_s *
    sr)`` — is the only thing that recovers it, and it recovers it at every
    cadence. This is the same limitation the SmellNet audit records: auto-R0
    cannot invent a baseline that is not there.
    """
    assert _r0_from_contract(_exposure(1.0, plateau_s=1.0), 15) == pytest.approx(1030.0)
    assert _r0_from_contract(_exposure(100.0, plateau_s=1.0), 15) == pytest.approx(1000.0)
    assert _r0_from_contract(_exposure(1.0, plateau_s=1.0), None) == pytest.approx(1015.0)
    for fs in CADENCES:
        declared = int(round(1.0 * fs))
        assert _r0_from_contract(_exposure(fs, plateau_s=1.0), declared) == pytest.approx(1000.0)


def test_declared_window_still_wins_end_to_end():
    """A declared window is used verbatim by every block that reduces R0.

    Samples 0-6 sit at 1000 and samples 7-14 at 2000, so the median and the
    spread both move when the window crosses index 7. That distinguishes a window
    of 7 from a window of 15, and from the undeclared default at n=101 (15).
    """
    series = np.array([1000.0] * 7 + [2000.0] * 8 + [1500.0] * 86)
    assert r0_window_samples(len(series), 7) == 7
    assert r0_window_samples(len(series), 15) == 15
    assert r0_window_samples(len(series), None) == 15

    assert _r0_from_contract(series, 7) == pytest.approx(1000.0)
    # 15 samples is 7x1000 then 8x2000, so the median is the 8th value, 2000.
    assert _r0_from_contract(series, 15) == pytest.approx(2000.0)

    narrow = compute_channel_health(series, r0_samples=7)
    wide = compute_channel_health(series, r0_samples=15)
    # R0 *and* noise_floor are taken over the same resolved window: a 7-sample
    # window sees a perfectly flat 1000 and so has zero spread.
    assert narrow["noise_floor"] == pytest.approx(0.0)
    assert wide["noise_floor"] > 0
    assert compute_channel_health(series)["noise_floor"] == pytest.approx(wide["noise_floor"])


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
