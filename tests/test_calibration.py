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
    concentration_series,
    fit_power_law,
    invert_concentration,
    loocv_power_law,
    normed_to_rr,
    two_point_calibration,
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
                 "CalibrationError"):
        assert hasattr(opensmell, name), f"missing top-level export {name}"
