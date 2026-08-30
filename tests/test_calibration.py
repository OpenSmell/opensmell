"""Reference-point calibration (§4.6, §10.10): power-law fit, inversion,
two-point exact recovery, LOOCV falsification, and the manifest payload.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

import opensmell
from opensmell.calibration import (
    CalibrationError,
    build_calibration_payload,
    calibrate_precise,
    calibrate_quick,
    concentration_series,
    fit_power_law,
    invert_concentration,
    loocv_power_law,
    normed_to_rr,
    two_point_calibration,
)
from opensmell.constants import (
    UnknownSensorError,
    all_power_laws,
    clean_air_ratio,
    is_power_law_sensor,
    power_law,
    sensor_gases,
    sensor_models,
    sensor_sources,
)
from opensmell.types import CalibrationDescriptor


def _clean_powerlaw(a, b, c, sigma=0.0, seed=0):
    """rr = a·C^b with optional multiplicative log-normal noise."""
    rng = np.random.default_rng(seed)
    rr = a * np.asarray(c, dtype=np.float64) ** b
    if sigma > 0:
        rr = rr * np.exp(rng.normal(0.0, sigma, size=rr.shape))
    return rr


def test_invert_roundtrip():
    a, b = 2.0, -0.5
    C = np.array([5.0, 20.0, 100.0])
    rr = a * C ** b
    assert np.allclose(invert_concentration(rr, a, b), C)


def test_invert_guards_return_nan():
    assert np.isnan(invert_concentration(-1.0, 2.0, -0.5))
    assert np.isnan(invert_concentration(1.0, 0.0, -0.5))
    assert np.isnan(invert_concentration(1.0, 2.0, 0.0))


def test_two_point_recovers_exact():
    a, b = 2.0, -0.7
    c1, c2 = 3.0, 50.0
    rr1, rr2 = a * c1 ** b, a * c2 ** b
    a_hat, b_hat = two_point_calibration(rr1, c1, rr2, c2)
    assert (a_hat, b_hat) == pytest.approx((a, b))


def test_two_point_guards():
    with pytest.raises(CalibrationError):
        two_point_calibration(1.0, 5.0, 2.0, 5.0)  # same concentration
    with pytest.raises(CalibrationError):
        two_point_calibration(1.0, 5.0, 1.0, 20.0)  # same response
    with pytest.raises(CalibrationError):
        two_point_calibration(-1.0, 5.0, 1.0, 20.0)  # non-positive


def test_fit_power_law_recovery_under_noise():
    c = np.logspace(0.0, 2.0, 6)
    rr = _clean_powerlaw(2.0, -0.6, c, sigma=0.03, seed=1)
    fit = fit_power_law(rr, c)
    assert fit["b"] == pytest.approx(-0.6, rel=0.05)
    assert fit["a"] == pytest.approx(2.0, rel=0.10)
    assert fit["r2"] > 0.98
    assert fit["n_points"] == 6
    assert fit["decades"] == pytest.approx(2.0)
    assert fit["method"] == "multi-point-loglog"


def test_fit_power_law_filters_invalid_and_guards():
    c = np.array([1.0, 10.0, 100.0])
    rr = _clean_powerlaw(2.0, -0.6, c)
    rr[1] = np.nan  # invalid point is dropped, not fatal
    fit = fit_power_law(rr, c)
    assert fit["n_points"] == 2
    with pytest.raises(CalibrationError):
        fit_power_law(np.array([1.0, np.nan]), np.array([1.0, 10.0]))  # 1 valid


def test_loocv_errors_stay_bounded_under_noise():
    c = np.logspace(0.0, 2.0, 6)
    rr = _clean_powerlaw(2.0, -0.6, c, sigma=0.05, seed=2)
    res = loocv_power_law(rr, c)
    assert res is not None
    assert res["n_folds"] == 6
    assert res["median_abs_pct_error"] < 20.0
    assert res["max_abs_pct_error"] < 60.0


def test_loocv_requires_three_points():
    c = np.array([1.0, 10.0])
    rr = _clean_powerlaw(2.0, -0.6, c)
    assert loocv_power_law(rr, c) is None


def test_build_payload_roundtrips_through_descriptor():
    c = np.logspace(1.0, 2.0, 4)
    rr0 = _clean_powerlaw(2.0, -0.6, c)
    rr1 = _clean_powerlaw(3.0, -0.4, c)
    payload = build_calibration_payload(
        {"VOC": fit_power_law(rr0, c), "CO": fit_power_law(rr1, c)},
        reference_substance="ethanol",
        calibration_date="2026-08-06",
    )
    desc = {ch: CalibrationDescriptor.from_dict(d) for ch, d in payload.items()}
    assert desc["VOC"].a == pytest.approx(2.0, rel=0.05)
    assert desc["VOC"].b == pytest.approx(-0.6, rel=0.05)
    assert desc["CO"].method == "multi-point-loglog"
    assert desc["VOC"].reference_substance == "ethanol"
    # round-trip through descriptor to_dict is stable
    again = {ch: CalibrationDescriptor.from_dict(dd.to_dict())
             for ch, dd in desc.items()}
    assert again["VOC"].a == desc["VOC"].a


def test_concentration_series_uses_normed_input():
    normed = np.array([[0.0, 0.5, 0.0], [0.1, 0.9, 0.0]])  # (n, 3)
    out = concentration_series(normed, {1: (2.0, -0.5)})
    # channel 1: rr = 1 + normed -> C = (rr/2)^(1/-0.5) = (rr/2)^-2
    expected = np.array([(1.5 / 2.0) ** -2, (1.9 / 2.0) ** -2])
    assert np.allclose(out[:, 1], expected)
    assert np.isnan(out[:, 0]).all()
    assert np.isnan(out[:, 2]).all()
    assert np.allclose(normed_to_rr(normed), 1.0 + normed)


def test_module_exports():
    for name in ("two_point_calibration", "fit_power_law",
                 "invert_concentration", "loocv_power_law",
                 "build_calibration_payload", "concentration_series",
                 "calibrate_quick", "calibrate_precise",
                 "CalibrationError"):
        assert hasattr(opensmell, name), f"missing top-level export {name}"


# --- constants table ---


def test_sensor_models_cover_common_mq():
    models = sensor_models()
    for m in ("MQ-2", "MQ-3", "MQ-4", "MQ-5", "MQ-6", "MQ-7", "MQ-8",
              "MQ-9", "MQ-135", "MQ-136", "MQ-131", "MQ-303A", "MQ-309A"):
        assert m in models, f"missing {m}"


def test_clean_air_ratio_present_and_positive():
    for m in sensor_models():
        if is_power_law_sensor(m):
            assert clean_air_ratio(m) > 0, m


def test_power_law_convention_roundtrips():
    # SDK: rr = a*C^b, C = (rr/a)^(1/b). Datasheet (MQUnified): C = a_u*rr^b_u.
    # Verify conversion consistency at a representative response.
    for m in sensor_models():
        if not is_power_law_sensor(m):
            continue  # relative-response MEMS channels have no gases table
        for gas, ab in all_power_laws(m).items():
            a, b = ab["a"], ab["b"]
            assert a > 0, (m, gas)
            assert b != 0, (m, gas)
            # pick a mid-range rr and confirm invert returns a finite C
            rr = max(2.0, a * 100.0 ** b)
            C = invert_concentration(rr, a, b)
            assert np.isfinite(C)
            assert np.allclose(a * C ** b, rr, rtol=1e-3), (m, gas)


def test_power_law_unknown_sensor():
    with pytest.raises(UnknownSensorError):
        power_law("MQ-999", "CO")
    with pytest.raises(UnknownSensorError):
        sensor_gases("NOPE")


def test_power_law_unknown_gas_keyerror():
    assert "CO" in sensor_gases("MQ-135")
    with pytest.raises(KeyError):
        power_law("MQ-136", "nonsense_gas")


def test_mq2_lpg_matches_known_datapoint():
    # Sanity anchor: converted MQ-2 LPG constants verified visually earlier.
    ab = power_law("MQ-2", "LPG")
    assert ab["a"] == pytest.approx(17.447, rel=1e-3)
    assert ab["b"] == pytest.approx(-0.450, rel=1e-2)


# --- sources & non-power-law (MEMS) entries ---


def test_sources_present_for_all_mq_sensors():
    for model in sensor_models():
        src = sensor_sources(model)
        assert isinstance(src, list), model
        # every entry must carry at least one verification link
        assert len(src) >= 1, model
        assert all(s.startswith("http") for s in src), model


def test_mq_sources_point_to_mqsensorslib():
    # the constants originate from MQUnifiedSensorsLib example .ino tables
    for model in sensor_models():
        if is_power_law_sensor(model):
            joined = " ".join(sensor_sources(model))
            assert "MQSensorsLib" in joined, model


def test_mems_entries_are_relative_response_not_powerlaw():
    # TGS8100 / SGP30 / SGP40 / BME680 are documented relative-response VOC
    # channels with NO authoritative power-law constants -- they must NOT be
    # quantifiable via a tabulated (a, b) pair.
    for model in ["TGS8100", "SGP30", "SGP40", "BME680"]:
        assert model in sensor_models(), model
        assert is_power_law_sensor(model) is False, model
        # requesting constants/gases must fail loudly (not fabricate values)
        with pytest.raises(UnknownSensorError):
            sensor_gases(model)
        with pytest.raises(UnknownSensorError):
            power_law(model, "VOC")


def test_calibrate_quick_rejects_non_powerlaw_mems():
    # calibrating a relative-response MEMS channel from offline constants is
    # not allowed -- there is no (a, b) to use.
    with pytest.raises(UnknownSensorError):
        calibrate_quick("SGP30", "VOC")


# --- calibrate_quick ---


def test_calibrate_quick_from_datasheet():
    q = calibrate_quick("MQ-2", "LPG", channel="alcohol_ch", reference_ppm=1000.0)
    assert "alcohol_ch" in q
    d = q["alcohol_ch"]
    assert d["method"] == "datasheet"
    assert d["referenceSubstance"] == "LPG"
    assert d["referencePpm"] == 1000.0
    assert d["a"] == pytest.approx(power_law("MQ-2", "LPG")["a"])
    assert d["b"] == pytest.approx(power_law("MQ-2", "LPG")["b"])
    # produced payload round-trips through the manifest descriptor
    desc = CalibrationDescriptor.from_dict(d)
    assert desc.a == d["a"]
    assert desc.b == d["b"]


def test_calibrate_quick_override():
    q = calibrate_quick("MQ-2", "CO", override={"a": 5.0, "b": -0.3})
    d = q["ch0"]
    assert d["a"] == 5.0
    assert d["b"] == -0.3


def test_calibrate_quick_unknown():
    with pytest.raises(UnknownSensorError):
        calibrate_quick("MQ-999", "CO")


def test_calibrate_quick_degenerate_override():
    with pytest.raises(CalibrationError):
        calibrate_quick("MQ-2", "LPG", override={"b": 0.0})
    with pytest.raises(CalibrationError):
        calibrate_quick("MQ-2", "LPG", override={"a": -1.0})


# --- calibrate_precise ---


def test_calibrate_precise_recovers_known_fit():
    c = np.logspace(1.0, 3.0, 6)
    a, b = 17.44, -0.450
    rr = a * c ** b
    res = calibrate_precise("MQ-2", "LPG", rr, c, channel="chLab")
    assert res["channel"] == "chLab"
    assert res["calibration"]["chLab"]["method"] == "multi-point-loglog"
    assert res["diagnostics"]["a"] == pytest.approx(a, rel=1e-2)
    assert res["diagnostics"]["b"] == pytest.approx(b, rel=1e-2)
    assert res["diagnostics"]["r2"] > 0.99
    assert res["loocv"]["n_folds"] == 6


def test_calibrate_precise_needs_two_points():
    with pytest.raises(CalibrationError):
        calibrate_precise("MQ-2", "LPG", [1.0], [10.0])
