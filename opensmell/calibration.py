"""Reference-point calibration (§4.6, §10.10).

The only sanctioned path to quantification: measured resistance ratio
``rr = R/R0`` at KNOWN concentrations ``C`` on a specific rig, per channel.
The sensor power law is

    rr = a · C^b          (b < 0 for the classic reducing-gas response)

so a channel's ``(a, b)`` are fitted in log-log space and concentration is
inverted as

    C = (rr / a) ^ (1 / b)

These are per-rig, per-channel, per-substance quantities. They do not transfer
across rigs (§4.6 proof), they are valid only for the reference substance they
were measured with, and they must be re-measured as the rig drifts. The
``sensor.calibration`` manifest contract (§10.10) stores them.

Design principles:

- *Warn, never silently interpolate.* Every fit reports R², residual spread,
  coverage (ppm span) and a leave-one-out concentration error.
- *Falsifiable.* ``loocv_power_law`` gives the held-out % error you should
  expect at the fitted concentrations; extrapolation outside the calibrated
  ppm range is explicitly penalized (see ``research/calibration-experiments/
  reference-point-calibration/`` for the quantified numbers).
"""

from __future__ import annotations

import math
from datetime import date
from typing import Dict, List, Optional

import numpy as np


class CalibrationError(ValueError):
    """A calibration fit is impossible with the given data (e.g. < 2 points,
    non-positive values, or a degenerate reference pair)."""


def normed_to_rr(normed) -> np.ndarray:
    """Convert normalized response ``(R - R0)/R0`` to the ratio ``R/R0``.

    The calibration power law uses ``rr = R/R0``; ``load_recording`` returns the
    normalized form, so ``rr = 1 + normed``.
    """
    arr = np.asarray(normed, dtype=np.float64)
    return 1.0 + arr


def invert_concentration(rr, a: float, b: float) -> np.ndarray:
    """Invert ``rr = a·C^b`` to concentration ``C = (rr/a)^(1/b)``.

    Returns NaN where ``rr``, ``a``, or ``b`` make the inversion undefined
    (``rr <= 0``, ``a <= 0``, ``b == 0``).
    """
    arr = np.asarray(rr, dtype=np.float64)
    scalar = arr.ndim == 0
    arr = np.atleast_1d(arr)
    a = float(a)
    b = float(b)
    if a <= 0 or b == 0:
        out = np.full(arr.shape, np.nan)
    else:
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            out = (arr / a) ** (1.0 / b)
        out = np.where(arr > 0, out, np.nan)
    return float(out[0]) if scalar else out


def two_point_calibration(rr1: float, c1: float, rr2: float, c2: float):
    """Exact (a, b) from two measured (rr, C) points (§4.6).

    ``b = log(rr1/rr2) / log(C1/C2)``, ``a = rr1 / C1^b``. Raises
    ``CalibrationError`` on degenerate inputs (equal concentrations, equal
    responses, non-positive values).
    """
    if not (rr1 > 0 and rr2 > 0 and c1 > 0 and c2 > 0):
        raise CalibrationError(
            f"Reference points must be positive: (rr1, c1, rr2, c2) = "
            f"({rr1}, {c1}, {rr2}, {c2}).")
    if c1 == c2:
        raise CalibrationError("Two-point calibration needs distinct concentrations.")
    if rr1 == rr2:
        raise CalibrationError("Two-point calibration needs distinct responses.")
    b = math.log(rr1 / rr2) / math.log(c1 / c2)
    a = rr1 / (c1 ** b)
    return float(a), float(b)


def _valid_mask(rr, c) -> np.ndarray:
    rr = np.asarray(rr, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    return np.isfinite(rr) & np.isfinite(c) & (rr > 0) & (c > 0)


def fit_power_law(rr, c) -> dict:
    """Multi-point power-law fit in log-log space (recommended over two-point).

    OLS of ``log(rr) = log(a) + b·log(C)``. Returns a dict with ``a``, ``b``,
    ``r2``, ``rmse_log_rr`` (residual spread in log-response, a noise proxy),
    ``n_points``, ppm coverage, and ``residuals`` (log domain).

    Requires at least 2 valid (positive, finite) points; raises
    ``CalibrationError`` otherwise.
    """
    rr = np.asarray(rr, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    if rr.ndim == 0:
        rr = rr.reshape(1)
        c = np.asarray(c, dtype=np.float64).reshape(1)
    mask = _valid_mask(rr, c)
    if mask.sum() < 2:
        raise CalibrationError(
            f"Need >= 2 valid (positive) reference points, got {int(mask.sum())}.")
    log_rr = np.log(rr[mask])
    log_c = np.log(c[mask])
    b, loga = np.polyfit(log_c, log_rr, 1)
    a = float(math.exp(loga))
    resid = log_rr - (b * log_c + loga)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((log_rr - log_rr.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return {
        "a": float(a),
        "b": float(b),
        "r2": r2,
        "rmse_log_rr": rmse,
        "n_points": int(mask.sum()),
        "min_ppm": float(np.min(c[mask])),
        "max_ppm": float(np.max(c[mask])),
        "decades": float(np.log10(np.max(c[mask]) / np.min(c[mask]))),
        "residuals": resid.tolist(),
        "method": "multi-point-loglog",
    }


def loocv_power_law(rr, c) -> Optional[dict]:
    """Leave-one-concentration-out falsification of the power-law fit.

    For each reference point, fit ``(a, b)`` on the other points and predict
    the held-out concentration; report relative error. Returns ``None`` when
    fewer than 3 valid points exist (a 2-point fit has no independent holdout).
    """
    rr = np.asarray(rr, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    mask = _valid_mask(rr, c)
    rr_v, c_v = rr[mask], c[mask]
    if len(rr_v) < 3:
        return None

    folds = []
    for i in range(len(rr_v)):
        tr_rr = np.concatenate([rr_v[:i], rr_v[i + 1:]])
        tr_c = np.concatenate([c_v[:i], c_v[i + 1:]])
        fit = fit_power_law(tr_rr, tr_c)
        pred = invert_concentration(rr_v[i], fit["a"], fit["b"])
        err = (float(pred) - float(c_v[i])) / float(c_v[i])
        folds.append({
            "heldout_ppm": float(c_v[i]),
            "pred_ppm": float(pred),
            "rel_error": float(err),
            "abs_pct_error": abs(err) * 100.0,
        })
    abs_pct = np.array([f["abs_pct_error"] for f in folds])
    rel = np.array([f["rel_error"] for f in folds])
    return {
        "n_folds": len(folds),
        "mean_abs_pct_error": float(np.mean(abs_pct)),
        "median_abs_pct_error": float(np.median(abs_pct)),
        "max_abs_pct_error": float(np.max(abs_pct)),
        "bias_pct": float(np.mean(rel) * 100.0),
        "folds": folds,
    }


def build_calibration_payload(
    fits: Dict[str, dict],
    reference_substance: str,
    calibration_date: Optional[str] = None,
    method: str = "multi-point-loglog",
) -> Dict[str, dict]:
    """Build a ``sensor.calibration`` manifest payload (§10.10 contract).

    ``fits`` maps channel id -> ``fit_power_law`` result. The returned dict is
    directly consumable by ``CalibrationDescriptor.from_dict`` and round-trips
    through ``.osmell``. ``reference_ppm`` is the geometric-mean concentration
    of the calibration range (the contract stores a single scalar).
    """
    if calibration_date is None:
        calibration_date = date.today().isoformat()
    payload: Dict[str, dict] = {}
    for ch, fit in fits.items():
        a = fit.get("a")
        b = fit.get("b")
        if a is None or b is None:
            continue
        ref_ppm = math.sqrt(max(fit.get("min_ppm", 1.0), 1.0)
                            * max(fit.get("max_ppm", 1.0), 1.0))
        payload[ch] = {
            "a": float(a),
            "b": float(b),
            "referenceSubstance": reference_substance,
            "referencePpm": float(ref_ppm),
            "date": calibration_date,
            "method": method,
        }
    return payload


def concentration_series(normed, params: Dict[int, tuple]) -> np.ndarray:
    """Per-channel concentration time series from a normalized recording.

    ``normed`` is ``(R - R0)/R0`` per channel (``load_recording`` output);
    ``params`` maps channel index -> ``(a, b)``. Uncalibrated channels yield
    NaN. Use with the ``sensor.calibration`` manifest contract via
    ``extract_all_framework_features(calibration=...)`` for the scalar feature
    path.
    """
    normed = np.asarray(normed, dtype=np.float64)
    if normed.ndim == 1:
        normed = normed.reshape(-1, 1)
    rr = normed_to_rr(normed)
    out = np.full_like(rr, np.nan)
    for ch, (a, b) in params.items():
        if 0 <= ch < rr.shape[1]:
            out[:, ch] = invert_concentration(rr[:, ch], a, b)
    return out
