import warnings
import numpy as np
from scipy import signal as sp_signal, optimize as sp_optimize

warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.optimize")
warnings.filterwarnings("ignore", message="Covariance of the parameters could not be estimated")

N_CHANNELS = 6


def _trapz(y, x=None):
    """Composite trapezoidal rule, version-robust across numpy.

    ``np.trapezoid`` (numpy >= 2.0) and ``np.trapz`` (numpy < 2.0) are the same
    function under different names; neither exists across both. Sending the
    ``trapezoid`` attribute through getattr keeps the AUC/hysteresis code working
    on any supported numpy without depending on a pinned major.
    """
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    if trapz is None:
        raise RuntimeError("numpy provides neither np.trapezoid nor np.trapz")
    if x is None:
        return trapz(y)
    return trapz(y, x)

NOMINAL_CALIBRATION = (1.0, -0.5)


def _r0_from_contract(series, r0_samples, r0=None):
    """R0 from the binding contract (§10.2): explicit R0, else median of the
    first ``r0_samples`` finite samples; guard falls back to mean of positive
    samples, then 1.0. Never returns NaN or a non-positive value."""
    series = np.asarray(series, dtype=np.float64)
    finite = series[np.isfinite(series)]
    if r0 is None:
        window = finite[:r0_samples] if r0_samples else finite
        r0 = float(np.median(window)) if len(window) else 0.0
    if not np.isfinite(r0) or r0 <= 0:
        positive = finite[finite > 0]
        r0 = float(np.mean(positive)) if len(positive) else 1.0
    return r0 if r0 > 0 else 1.0


def _exp_decay(t, a, tau, c):
    return a * np.exp(-t / tau) + c


def _bi_exp_decay(t, a1, tau1, a2, tau2, c):
    return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2) + c


def _tri_exp_decay(t, a1, tau1, a2, tau2, a3, tau3, c):
    return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2) + a3 * np.exp(-t / tau3) + c


def compute_channel_device_agnostic(series, r0_samples=15, sr=10, r0=None):
    series = np.asarray(series, dtype=np.float64)
    if len(series) < r0_samples + 2:
        return {k: -1.0 for k in ["relative_amplitude", "direction", "rise_time",
                                   "decay_time", "auc", "endpoint_delta"]}
    R0 = _r0_from_contract(series, r0_samples, r0)

    finite = series[np.isfinite(series)]
    std_ratio = float(np.std(finite) / R0) if len(finite) else float("inf")
    dead = (len(finite) < 2) or (std_ratio < 0.001)
    if dead:
        return {"relative_amplitude": 0.0, "direction": 0,
                "rise_time": -1.0, "decay_time": -1.0,
                "auc": 0.0, "endpoint_delta": 0.0,
                "R0": float(R0), "is_dead": True, "std_ratio": std_ratio}

    norm = (series - R0) / R0
    max_val, min_val = np.max(series), np.min(series)
    delta_max, delta_min = max_val - R0, min_val - R0
    if abs(delta_max) >= abs(delta_min):
        peak_raw, delta_raw = max_val, delta_max
        direction = 1
    else:
        peak_raw, delta_raw = min_val, delta_min
        direction = -1

    relative_amplitude = abs(delta_raw) / R0
    full_span = abs(delta_raw)

    threshold_10 = R0 + 0.1 * full_span * direction
    threshold_90 = R0 + 0.9 * full_span * direction
    threshold_10 = np.clip(threshold_10, min_val, max_val)
    threshold_90 = np.clip(threshold_90, min_val, max_val)

    def first_cross(series, thresh, dir_):
        if dir_ > 0:
            cond = series >= thresh
        else:
            cond = series <= thresh
        idx = np.where(cond)[0]
        if len(idx) == 0 or idx[0] >= len(series) - 1:
            return None
        return idx[0]

    rise_time = -1.0
    idx_10 = first_cross(series, threshold_10, direction)
    idx_90 = first_cross(series, threshold_90, direction)
    if idx_10 is not None and idx_90 is not None:
        rise_time = abs(idx_90 - idx_10) / sr

    decay_time = -1.0
    peak_idx = np.argmax(abs(series - R0))
    post_peak = series[peak_idx:]
    if len(post_peak) > 2:
        desorb_dir = -direction
        if desorb_dir > 0:
            cond_start = post_peak >= threshold_90
            cond_end = post_peak >= threshold_10
        else:
            cond_start = post_peak <= threshold_90
            cond_end = post_peak <= threshold_10
        start_idx = np.where(cond_start)[0]
        if len(start_idx) > 0:
            si = start_idx[0]
            end_candidates = np.where(cond_end[si:])[0]
            if len(end_candidates) > 0:
                decay_time = (si + end_candidates[0]) / sr

    auc = float(_trapz(np.abs(norm)))
    endpoint_delta = float((series[-1] - R0) / R0)

    return {
        "relative_amplitude": float(relative_amplitude),
        "direction": int(direction),
        "rise_time": float(rise_time),
        "decay_time": float(decay_time),
        "auc": float(auc),
        "endpoint_delta": float(endpoint_delta),
        "series_norm": norm,
        "series_raw": series,
        "R0": float(R0),
        "peak_idx": int(peak_idx),
        "is_dead": bool(dead),
        "std_ratio": float(std_ratio),
    }


def compute_channel_absolute(series, r0=None, a_const=1.0, b_const=-0.5):
    series = np.asarray(series, dtype=np.float64)
    if r0 is None or not (np.isfinite(r0) and r0 > 0):
        r0 = _r0_from_contract(series, 15, r0)
    raw_resistance = float(np.mean(series[-10:])) if len(series) >= 10 else float(np.mean(series))
    baseline_resistance = float(r0)
    voltage = float(raw_resistance)
    rr_ratio = raw_resistance / r0 if r0 > 0 else 0.0
    if b_const != 0 and rr_ratio > 0:
        calib_conc = ((rr_ratio / max(a_const, 0.001)) ** (1.0 / b_const))
    else:
        calib_conc = 0.0
    return {
        "raw_resistance": raw_resistance,
        "baseline_resistance": baseline_resistance,
        "voltage": voltage,
        "calibrated_concentration": float(calib_conc),
    }


def compute_channel_temporal(series, sr=10):
    series = np.asarray(series, dtype=np.float64)
    if len(series) < 5:
        return {"hf_transient": 0.0, "oscillation_freq": 0.0,
                "oscillation_amp": 0.0, "response_latency": -1.0}

    diffs = np.diff(series)
    hf_transient = float(np.mean(np.abs(diffs))) if len(diffs) > 0 else 0.0

    detrended = sp_signal.detrend(series)
    if len(detrended) > 20:
        freqs, psd = sp_signal.periodogram(detrended, fs=sr)
        peak_idx = np.argmax(psd[1:]) + 1 if len(psd) > 2 else 0
        osc_freq = float(freqs[peak_idx]) if peak_idx > 0 else 0.0
        osc_amp = float(np.sqrt(psd[peak_idx])) if peak_idx > 0 else 0.0
    else:
        osc_freq, osc_amp = 0.0, 0.0

    response_latency = -1.0
    threshold = np.std(series[:10]) * 3 if len(series) >= 10 else np.std(series) * 3
    baseline_mean = np.mean(series[:10]) if len(series) >= 10 else series[0]
    for i in range(10, len(series)):
        if abs(series[i] - baseline_mean) > threshold:
            response_latency = i / sr
            break

    return {
        "hf_transient": hf_transient,
        "oscillation_freq": osc_freq,
        "oscillation_amp": osc_amp,
        "response_latency": response_latency,
    }


def compute_channel_health(series, r0_samples=15, r0=None):
    series = np.asarray(series, dtype=np.float64)
    if len(series) < r0_samples + 5:
        return {"drift_rate": 0.0, "sensitivity_decay": 0.0,
                "noise_floor": 0.0, "hysteresis": 0.0}

    r0 = _r0_from_contract(series, r0_samples, r0)

    drift_rate = float((np.mean(series[-10:]) - r0) / r0) if len(series) >= 10 else 0.0
    sensitivity_decay = 0.0
    noise_floor = float(np.std(series[:r0_samples]) / r0) if r0 > 0 else 0.0

    peak_idx = np.argmax(np.abs(series - r0))
    if peak_idx < len(series) - 5 and peak_idx > 5:
        ads_curve = series[:peak_idx + 1]
        des_curve = series[peak_idx:]
        ads_path = _trapz(np.abs(ads_curve - r0))
        des_path = _trapz(np.abs(des_curve - r0))
        hysteresis = float(abs(ads_path - des_path) / max(ads_path, 1e-10))
    else:
        hysteresis = 0.0

    return {
        "drift_rate": drift_rate,
        "sensitivity_decay": sensitivity_decay,
        "noise_floor": noise_floor,
        "hysteresis": hysteresis,
    }


def compute_multi_exp_decay(series, peak_idx=None, sr=10, n_components=2, r0=None):
    """Fit multi-exponential decay constants to the recovery phase.

    The default model is bi-exponential (fast + slow components):

        y(t) = a1·exp(-t/tau1) + a2·exp(-t/tau2) + c

    Physical meaning: Different binding sites on the MOX surface have different
    desorption activation energies. Fast sites (tau1 ~1-3s) correspond to weakly
    adsorbed molecules; slow sites (tau2 ~10-30s) correspond to strongly bound
    species. The ratio tau2/tau1 indicates surface heterogeneity.

    A tri-exponential model (tau1, tau2, tau3) is over-parameterized for MOX
    recovery data: the three components are not uniquely identifiable, which
    makes the fit unreliable (it can alternate between near-equal-cost local
    minima). It is therefore opt-in only via ``n_components=3`` for edge cases
    (e.g. unusual surface chemistry), and may be nondeterministic.

    Returns dict with tau1, tau2, tau3 and amplitudes a1, a2, a3 (tau3/a3 are
    -1.0 for the default bi-exponential fit), plus ``cost`` (residual sum of
    squares of the final fit). Returns -1 for all if fitting fails.

    Note: results may vary slightly across runs for fits where the optimizer
    converges to near-equal-cost local minima. This is a known limitation of
    scipy's MINPACK optimizer on equal-cost solutions, not of the model itself.
    """
    if n_components >= 3:
        n_components = 3

    series = np.asarray(series, dtype=np.float64)
    if len(series) < 20:
        return {"tau1": -1.0, "tau2": -1.0, "tau3": -1.0,
                "a1": -1.0, "a2": -1.0, "a3": -1.0, "cost": -1.0}

    if peak_idx is None:
        R0_est = _r0_from_contract(series, min(15, len(series) // 3), r0)
        peak_idx = np.argmax(np.abs(series - R0_est))
    peak_idx = max(5, min(peak_idx, len(series) - 10))

    recovery = series[peak_idx:]
    if len(recovery) < 10:
        return {"tau1": -1.0, "tau2": -1.0, "tau3": -1.0,
                "a1": -1.0, "a2": -1.0, "a3": -1.0, "cost": -1.0}

    t = np.arange(len(recovery), dtype=np.float64) / sr
    y = recovery - recovery[-1]  # zero-end
    y = y[:len(t)]

    if np.all(y == 0) or np.std(y) < 1e-8:
        return {"tau1": -1.0, "tau2": -1.0, "tau3": -1.0,
                "a1": -1.0, "a2": -1.0, "a3": -1.0, "cost": -1.0}

    a0 = y[0]
    results = {"tau1": -1.0, "tau2": -1.0, "tau3": -1.0,
               "a1": -1.0, "a2": -1.0, "a3": -1.0, "cost": -1.0}

    try:
        popt, _ = sp_optimize.curve_fit(
            _exp_decay, t, y,
            p0=[a0, 3.0, 0],
            maxfev=2000,
        )
        results["tau1"] = float(abs(popt[1]))
        results["a1"] = float(popt[0])
        results["cost"] = float(np.sum((_exp_decay(t, *popt) - y) ** 2))
    except (RuntimeError, ValueError):
        pass

    try:
        popt, _ = sp_optimize.curve_fit(
            _bi_exp_decay, t, y,
            p0=[a0 * 0.7, 2.0, a0 * 0.3, 10.0, 0],
            maxfev=5000,
        )
        results["tau1"] = float(abs(popt[1]))
        results["tau2"] = float(abs(popt[3]))
        results["a1"] = float(popt[0])
        results["a2"] = float(popt[2])
        results["cost"] = float(np.sum((_bi_exp_decay(t, *popt) - y) ** 2))
    except (RuntimeError, ValueError):
        pass

    if n_components == 3 and len(recovery) >= 30:
        try:
            popt, _ = sp_optimize.curve_fit(
                _tri_exp_decay, t, y,
                p0=[a0 * 0.5, 1.5, a0 * 0.3, 5.0, a0 * 0.2, 20.0, 0],
                maxfev=10000,
            )
            results["tau1"] = float(abs(popt[1]))
            results["tau2"] = float(abs(popt[3]))
            results["tau3"] = float(abs(popt[5]))
            results["a1"] = float(popt[0])
            results["a2"] = float(popt[2])
            results["a3"] = float(popt[4])
            results["cost"] = float(np.sum((_tri_exp_decay(t, *popt) - y) ** 2))
        except (RuntimeError, ValueError):
            pass

    return results


def compute_saturation_index(series, r0_samples=15, r0=None):
    """Compute how close the sensor response is to its estimated saturation capacity.

    Saturation index = observed response / estimated saturation response.
    0.0 = no response, 1.0 = fully saturated.

    Physical meaning: At high concentrations, all surface sites are occupied
    and the sensor cannot respond further. The saturation index tracks how
    close to this limit the current measurement is, derived from the Langmuir
    adsorption isotherm approximation.

    Uses a simple empirical estimator: ratio of current response to the
    maximum ever observed response in the series, scaled by noise floor.
    """
    series = np.asarray(series, dtype=np.float64)
    if len(series) < r0_samples + 5:
        return 0.0

    R0 = _r0_from_contract(series, r0_samples, r0)

    norm = np.abs(series - R0) / R0
    current_response = float(np.max(norm))
    noise_floor = float(np.std(norm[:r0_samples]))

    if current_response < noise_floor * 2:
        return 0.0

    denom = current_response + noise_floor * 10
    saturation = min(1.0, current_response / denom) if denom > 0 else 0.0
    return float(saturation)


def compute_channel_hardware(series):
    series = np.asarray(series, dtype=np.float64)
    if len(series) < 2:
        return {"circuit_response": 0.0, "thermal_profile": 0.0,
                "adc_noise": 0.0}

    circuit_response = float(np.mean(series))
    thermal_profile = float(np.std(series))

    adc_noise = 0.0
    if len(series) >= 10:
        smooth = np.convolve(series, np.ones(5) / 5, mode="valid")
        residuals = series[2:2 + len(smooth)] - smooth
        adc_noise = float(np.std(residuals)) if len(residuals) > 0 else 0.0

    return {
        "circuit_response": circuit_response,
        "thermal_profile": thermal_profile,
        "adc_noise": adc_noise,
    }


def extract_all_framework_features(data, r0_samples=15, sr=10, r0_per_channel=None, calibration=None):
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n_ch = data.shape[1]

    features = {}

    device_agnostic_results = []
    absolute_results = []
    temporal_results = []
    health_results = []
    hardware_results = []
    decay_results = []

    for ch in range(n_ch):
        series = data[:, ch]

        R0 = _r0_from_contract(series, r0_samples, (r0_per_channel or {}).get(ch))
        da = compute_channel_device_agnostic(series, r0_samples, sr, R0)
        device_agnostic_results.append(da)
        R0 = da.get("R0", R0)

        a_c, b_c = NOMINAL_CALIBRATION
        if calibration:
            cal = calibration.get(ch)
            if cal:
                a_c = cal.get("a", NOMINAL_CALIBRATION[0])
                b_c = cal.get("b", NOMINAL_CALIBRATION[1])
        ab = compute_channel_absolute(series, r0=R0, a_const=a_c, b_const=b_c)
        absolute_results.append(ab)

        te = compute_channel_temporal(series, sr)
        temporal_results.append(te)

        he = compute_channel_health(series, r0_samples, R0)
        health_results.append(he)

        ha = compute_channel_hardware(series)
        hardware_results.append(ha)

        pk = da.get("peak_idx", len(series) // 2)
        dec = compute_multi_exp_decay(series, peak_idx=pk, sr=sr, r0=R0)
        decay_results.append(dec)

        sat = compute_saturation_index(series, r0_samples, R0)
        features[f"ch{ch}_advanced_saturation_index"] = sat

        for feat_name in ["relative_amplitude", "direction", "rise_time",
                           "decay_time", "auc", "endpoint_delta"]:
            features[f"ch{ch}_da_{feat_name}"] = da.get(feat_name, -1.0)

        for feat_name in ["raw_resistance", "baseline_resistance",
                           "voltage", "calibrated_concentration"]:
            features[f"ch{ch}_abs_{feat_name}"] = ab.get(feat_name, 0.0)

        for feat_name in ["hf_transient", "oscillation_freq",
                           "oscillation_amp", "response_latency"]:
            features[f"ch{ch}_temp_{feat_name}"] = te.get(feat_name, 0.0)

        for feat_name in ["drift_rate", "sensitivity_decay",
                           "noise_floor", "hysteresis"]:
            features[f"ch{ch}_health_{feat_name}"] = he.get(feat_name, 0.0)

        for feat_name in ["circuit_response", "thermal_profile", "adc_noise"]:
            features[f"ch{ch}_hw_{feat_name}"] = ha.get(feat_name, 0.0)

        for feat_name in ["tau1", "tau2", "tau3", "a1", "a2", "a3"]:
            features[f"ch{ch}_decay_{feat_name}"] = dec.get(feat_name, -1.0)

    active = [i for i, r in enumerate(device_agnostic_results)
              if not r.get("is_dead", True) and r.get("relative_amplitude", 0) > 0]

    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            ci, cj = active[i], active[j]
            dr_i = device_agnostic_results[ci].get("relative_amplitude", 0)
            dr_j = device_agnostic_results[cj].get("relative_amplitude", 0)
            ratio = dr_i / dr_j if dr_j > 0 else 0
            features[f"sel_ratio_ch{ci}_ch{cj}"] = ratio

    for ch_i in range(n_ch):
        for ch_j in range(n_ch):
            if ch_i >= ch_j:
                continue
            key = f"sel_ratio_ch{ch_i}_ch{ch_j}"
            if key not in features:
                features[key] = 0.0

    active_dr = [device_agnostic_results[i].get("relative_amplitude", 0)
                 for i in range(n_ch)
                 if not device_agnostic_results[i].get("is_dead", True)]
    features["global_max_delta_ratio"] = float(max(active_dr)) if active_dr else 0.0
    features["global_mean_delta_ratio"] = float(np.mean(active_dr)) if active_dr else 0.0
    features["global_n_active_channels"] = len(active_dr)
    features["global_total_auc"] = float(
        sum(device_agnostic_results[i].get("auc", 0)
            for i in range(n_ch)
            if not device_agnostic_results[i].get("is_dead", True))
    )

    return features


def feature_names(n_channels=N_CHANNELS):
    """Ordered framework feature names for ``n_channels``.

    The framework length is a function of channel count —
    ``28·c + c(c−1)/2 + 4`` (28 per channel, one selectivity ratio per pair, 4
    global) — so names are generated for the given ``n_channels``. This mirrors
    the extractor, which derives the channel count from the data shape, and the
    Rust SDK's ``framework_feature_len(c)``. Defaults to the canonical 6.
    """
    n_ch = N_CHANNELS if n_channels is None else n_channels
    if n_ch < 1:
        return []
    names = []
    for ch in range(n_ch):
        for fn in ["relative_amplitude", "direction", "rise_time", "decay_time",
                    "auc", "endpoint_delta"]:
            names.append(f"ch{ch}_da_{fn}")
    for ch in range(n_ch):
        for fn in ["raw_resistance", "baseline_resistance", "voltage",
                    "calibrated_concentration"]:
            names.append(f"ch{ch}_abs_{fn}")
    for ch in range(n_ch):
        for fn in ["hf_transient", "oscillation_freq", "oscillation_amp",
                    "response_latency"]:
            names.append(f"ch{ch}_temp_{fn}")
    for ch in range(n_ch):
        for fn in ["drift_rate", "sensitivity_decay", "noise_floor", "hysteresis"]:
            names.append(f"ch{ch}_health_{fn}")
    for ch in range(n_ch):
        for fn in ["circuit_response", "thermal_profile", "adc_noise"]:
            names.append(f"ch{ch}_hw_{fn}")
    for ch in range(n_ch):
        names.append(f"ch{ch}_advanced_saturation_index")
        for fn in ["tau1", "tau2", "tau3", "a1", "a2", "a3"]:
            names.append(f"ch{ch}_decay_{fn}")
    for ch_i in range(n_ch):
        for ch_j in range(ch_i + 1, n_ch):
            names.append(f"sel_ratio_ch{ch_i}_ch{ch_j}")
    names.extend(["global_max_delta_ratio", "global_mean_delta_ratio",
                   "global_n_active_channels", "global_total_auc"])
    return names


# ---------------------------------------------------------------------------
# Ported kinetic features (web lib/osmell/processors.ts -> processMox).
# Per-channel, array-driven: one feature row per declared channel.
# ---------------------------------------------------------------------------

from dataclasses import dataclass
from typing import List, Optional

from ..types import DEFAULT_R0_SAMPLES
from .normalize import baseline_for_channel, channel_stats, normalized_series, std


def calibration_for_channel(manifest, channel_id):
    """Return (a, b, source) for a channel from the manifest `sensor.calibration`
    contract (§10.10). Falls back to nominal defaults with source
    ``"nominal-default"`` — never silently: callers can see which path was used."""
    sensor = getattr(manifest, "sensor", None)
    cal = getattr(sensor, "calibration", None) or {}
    entry = cal.get(channel_id)
    if entry is not None and entry.a > 0 and entry.b != 0:
        return entry.a, entry.b, "manifest"
    return NOMINAL_CALIBRATION[0], NOMINAL_CALIBRATION[1], "nominal-default"


@dataclass
class MoxFeatures:
    channel: str
    relative_amplitude: float
    direction: int
    rise_time_ms: Optional[float]
    decay_time_ms: Optional[float]
    auc: float
    r0: float
    dead: bool
    endpoint_delta: float
    saturation_index: float


def _first_cross_time(time, norm, threshold):
    for i in range(len(norm)):
        if norm[i] >= threshold:
            return time[i]
    return None


def _argmax_abs(norm):
    best = 0
    for i in range(1, len(norm)):
        if abs(norm[i]) > abs(norm[best]):
            best = i
    return best


def _decay_time_ms_after(norm, time, peak_idx):
    if len(norm) - peak_idx <= 2:
        return None
    pk = norm[peak_idx]
    if not (pk == pk) or pk == 0:
        return None
    t90 = 0.9 * pk
    t10 = 0.1 * pk
    near_peak = (lambda v: v >= t90) if pk >= 0 else (lambda v: v <= t90)
    near_baseline = (lambda v: v <= t10) if pk >= 0 else (lambda v: v >= t10)
    si = -1
    for i in range(peak_idx, len(norm)):
        if near_peak(norm[i]):
            si = i
            break
    if si < 0:
        return None
    for i in range(si, len(norm)):
        if near_baseline(norm[i]):
            return time[i] - time[peak_idx]
    return None


def _saturation_index_for(norm, r0_samples):
    if len(norm) < r0_samples + 5:
        return 0.0
    r0_norm = norm[:r0_samples]
    current_response = max((abs(v) for v in norm), default=0.0)
    noise_floor = std(r0_norm)
    if noise_floor != noise_floor or current_response < noise_floor * 2:
        return 0.0
    return min(1.0, current_response / (current_response + noise_floor * 10))


def process_mox(file) -> dict:
    """Compute per-channel MOX kinetic features (web processMox parity)."""
    channels = file.manifest.sensor.channels
    baseline = file.manifest.baseline
    r0_samples = (baseline.r0_samples if baseline and baseline.r0_samples
                  else DEFAULT_R0_SAMPLES)
    features: List[MoxFeatures] = []
    normalized = {}

    for ch in channels:
        cid = ch.id
        values = file.data.get(cid, [])
        r0 = baseline_for_channel(file, cid, values)[0]
        stats = channel_stats(values, r0)
        norm = normalized_series(values, r0)

        finite_norm = [v for v in norm if v == v]
        relative_amplitude = 0.0
        direction = 1
        auc = 0.0
        rise_time_ms = None
        decay_time_ms = None
        endpoint_delta = 0.0
        saturation_index = 0.0

        if not stats.dead and finite_norm:
            max_val = max(finite_norm)
            min_val = min(finite_norm)
            peak = max_val if abs(max_val) >= abs(min_val) else min_val
            direction = 1 if peak >= 0 else -1
            relative_amplitude = abs(peak)

            span = max_val - min_val
            t10 = _first_cross_time(file.time, norm, min_val + 0.1 * span)
            t90 = _first_cross_time(file.time, norm, min_val + 0.9 * span)
            if t10 is not None and t90 is not None:
                rise_time_ms = t90 - t10

            prev = norm[0]
            for i in range(1, len(norm)):
                dt = file.time[i] - file.time[i - 1]
                if dt > 0:
                    auc += (norm[i] + prev) * dt * 0.5
                prev = norm[i]

            peak_idx = _argmax_abs(norm)
            decay_time_ms = _decay_time_ms_after(norm, file.time, peak_idx)
            endpoint_delta = norm[-1]
            saturation_index = _saturation_index_for(norm, r0_samples)

        normalized[cid] = norm
        features.append(MoxFeatures(
            channel=cid,
            relative_amplitude=relative_amplitude,
            direction=direction,
            rise_time_ms=rise_time_ms,
            decay_time_ms=decay_time_ms,
            auc=auc,
            r0=r0,
            dead=stats.dead,
            endpoint_delta=endpoint_delta,
            saturation_index=saturation_index,
        ))

    return {"sensor_type": "mox", "features": features, "normalized": normalized}
