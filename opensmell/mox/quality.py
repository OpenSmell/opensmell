"""MOX quality scoring — spec-compliant 7-factor implementation.

Port of OSMELL_FORMAT_SPEC.md §7. Factors and weights:

    C continuity             0.15
    D dynamic range          0.10
    S saturation-free        0.10
    B baseline stability     0.20
    G signal strength / SNR  0.20
    R recovery completeness  0.15
    T duration adequacy      0.10

G and R are `null` (excluded from the total) for any role other than exposure.
When `baseline.source == "auto"`, B is capped at 50 (an auto-R0 cannot earn full
baseline credit). When `adcMax` is undeclared, upper-rail clipping is not
detectable and only the lower rail (`<= 0`) counts toward saturation. When
`samplingRateHz` is undeclared, continuity uses the median gap as the nominal
schedule.
"""

from __future__ import annotations

from typing import List, Optional

from ..normalize import mean, median
from ..types import (
    DEFAULT_ADC_MAX,
    FULL_SCORE_DURATION_S,
    GAP_TOLERANCE,
    MIN_SPAN_FRACTION,
    NOISE_CV_LIMIT,
    SNR_TARGET,
    ChannelStats,
    OsmellFile,
    QualityFlags,
    QualityReport,
    SubScore,
)
from .normalize import baseline_for_channel, channel_stats, normalized_series

WEIGHTS = {
    "continuity": 0.15,
    "dynamicRange": 0.10,
    "saturationFree": 0.10,
    "baselineStability": 0.20,
    "signalStrength": 0.20,
    "recoveryCompleteness": 0.15,
    "durationAdequacy": 0.10,
}


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def compute_quality_mox(
    file: OsmellFile,
    sample_count: int,
    guess_sampling_rate_hz: float,
    unsorted: bool,
    non_finite: int,
) -> QualityReport:
    sensor = file.manifest.sensor
    adc_declared = sensor.adc_max is not None
    adc_max = sensor.adc_max if adc_declared else DEFAULT_ADC_MAX
    rate_declared = sensor.sampling_rate_hz is not None
    sampling_rate_hz = sensor.sampling_rate_hz if rate_declared else guess_sampling_rate_hz
    channel_ids = [c.id for c in sensor.channels]
    role = file.manifest.session.role if file.manifest.session else "single"
    baseline_source = file.manifest.baseline.source if file.manifest.baseline else "none"

    flags = QualityFlags(
        dead_sensors=[],
        unsorted_rows=unsorted,
        non_finite_samples=non_finite,
        used_default_adc_max=not adc_declared,
        used_median_sampling_rate=not rate_declared,
        no_baseline=baseline_source == "none",
        empty_recording=sample_count == 0,
    )
    reasons: dict[str, str] = {}
    notes: List[str] = []

    # --- Continuity C (spec 7.1.1) ---
    gaps = [file.time[i + 1] - file.time[i] for i in range(len(file.time) - 1)]
    positive_gaps = [g for g in gaps if g > 0]
    if sample_count < 2:
        continuity = SubScore(value=100.0, reason="ok")
    else:
        if rate_declared:
            nominal = 1000.0 / sampling_rate_hz if sampling_rate_hz and sampling_rate_hz > 0 else None
        else:
            nominal = median(positive_gaps) if positive_gaps else None
            if nominal is not None:
                notes.append("samplingRateHz not declared; nominal period taken as the median gap.")
            flags.used_median_sampling_rate = True
        if nominal is not None and nominal > 0:
            tol = GAP_TOLERANCE * nominal
            regular = sum(1 for g in gaps if abs(g - nominal) <= tol)
            total = len(gaps)
            continuity = SubScore(
                value=100.0 if total == 0 else (regular / total) * 100.0,
                reason="irregular_gaps" if regular < total else "ok",
            )
        else:
            continuity = SubScore(value=50.0, reason="irregular_gaps")

    # --- Per-channel stats with R0 ---
    stats: List[ChannelStats] = []
    for cid in channel_ids:
        values = file.data.get(cid, [])
        r0 = baseline_for_channel(file, cid, values)[0]
        st = channel_stats(values, r0)
        st.id = cid
        if st.dead:
            flags.dead_sensors.append(cid)
        stats.append(st)

    live = [s for s in stats if not s.dead]

    # --- Dynamic range D (spec 7.1.2) ---
    dynamic_value = 0.0 if not live else 100.0 * mean(
        [_clamp((s.span / adc_max) * (1.0 / MIN_SPAN_FRACTION), 0.0, 1.0) for s in live]
    )
    dynamic_range = SubScore(
        value=dynamic_value,
        reason="low_span" if dynamic_value < 50 else "ok",
    )
    if dynamic_range.reason == "low_span":
        reasons["dynamicRange"] = "channel_span_below_10_percent_of_adc_range"

    # --- Saturation-free S (spec 7.1.3) ---
    sat_scores = []
    for s in stats:
        values = file.data.get(s.id, [])
        if adc_declared:
            clipped = sum(1 for v in values if v >= adc_max or v <= 0)
        else:
            clipped = sum(1 for v in values if v <= 0)
        s.clipped = clipped
        sat_scores.append(100.0 if len(values) == 0 else 100.0 * (1.0 - clipped / len(values)))
    saturation_free = SubScore(value=mean(sat_scores), reason="ok")

    # --- Baseline stability B (spec 7.1.4) ---
    if baseline_source == "none":
        baseline_stability = SubScore(value=0.0, reason="no_baseline")
    else:
        cvs = []
        for s in stats:
            values = file.data.get(s.id, [])
            cvs.append(baseline_for_channel(file, s.id, values)[2])
        finite_cvs = [c for c in cvs if _is_finite(c)]
        cv_window = mean(finite_cvs) if finite_cvs else float("nan")
        raw_b = 100.0 * _clamp(1.0 - cv_window / NOISE_CV_LIMIT, 0.0, 1.0)
        if baseline_source == "auto":
            baseline_stability = SubScore(value=min(raw_b, 50.0), reason="auto_r0")
        else:
            baseline_stability = SubScore(
                value=raw_b,
                reason="r0_window_cv_too_high" if cv_window >= NOISE_CV_LIMIT else "ok",
            )

    # --- Signal strength G + Recovery completeness R (spec 7.1.5 / 7.1.6) ---
    exposure_with_r0 = role == "exposure" and baseline_source != "none"
    if not exposure_with_r0:
        signal_strength = SubScore(value=None, reason="no_exposure_signal")
        recovery = SubScore(value=None, reason="no_exposure_signal")
    else:
        best_g: List[float] = []
        recovery_scores: List[float] = []
        for s in live:
            values = file.data.get(s.id, [])
            r0 = baseline_for_channel(file, s.id, values)[0]
            norm = [v for v in normalized_series(values, r0) if _is_finite(v)]
            base = baseline_for_channel(file, s.id, values)
            noise = max(base[2], 1e-6)
            if not norm:
                best_g.append(0.0)
                recovery_scores.append(0.0)
                continue
            peak = max(abs(v) for v in norm)
            best_g.append(_clamp(peak / noise / SNR_TARGET, 0.0, 1.0) * 100.0)
            final_win = median(norm[-15:]) if norm else 0.0
            recovered = 1.0 - _clamp(abs(final_win) / max(peak, 1e-6), 0.0, 1.0)
            recovery_scores.append(100.0 * recovered)
        signal_strength = SubScore(value=max(best_g) if best_g else 0.0, reason="ok")
        recovery = SubScore(value=mean(recovery_scores) if recovery_scores else 0.0, reason="ok")

    # --- Duration adequacy T (spec 7.1.7) ---
    t_seconds = ((sample_count - 1) / sampling_rate_hz) if sampling_rate_hz and sampling_rate_hz > 0 else 0.0
    duration_adequacy = SubScore(
        value=100.0 * _clamp(t_seconds / FULL_SCORE_DURATION_S, 0.0, 1.0),
        reason="too_short" if t_seconds < FULL_SCORE_DURATION_S else "ok",
    )

    subs = {
        "continuity": continuity,
        "dynamicRange": dynamic_range,
        "saturationFree": saturation_free,
        "baselineStability": baseline_stability,
        "signalStrength": signal_strength,
        "recoveryCompleteness": recovery,
        "durationAdequacy": duration_adequacy,
    }

    weighted = 0.0
    sum_w = 0.0
    for k, sub in subs.items():
        if sub.value is None:
            continue
        weighted += WEIGHTS[k] * sub.value
        sum_w += WEIGHTS[k]

    total = round(weighted / sum_w) if sum_w > 0 else None
    if total is None:
        badge = "Unknown"
    elif total >= 90:
        badge = "Excellent"
    elif total >= 75:
        badge = "Good"
    elif total >= 50:
        badge = "Fair"
    else:
        badge = "Poor"

    if flags.dead_sensors:
        notes.append(f"Dead sensors (cv < 0.001): {', '.join(flags.dead_sensors)}")
    if flags.non_finite_samples:
        notes.append(f"{flags.non_finite_samples} non-finite values skipped.")
    if flags.unsorted_rows:
        notes.append("Rows were out of order and were sorted.")
    if not rate_declared:
        notes.append("Sampling rate inferred from median gap; verify against hardware.")
    if not adc_declared:
        notes.append("adcMax not declared; upper-rail clipping not checked (lower rail only).")
    if flags.no_baseline:
        notes.append("No baseline; auto-R0 applied and baseline stability scores zero.")

    return QualityReport(
        format="opensmell-quality",
        version="1",
        computed_at=_utc_now_iso(),
        total=total,
        badge=badge,
        subscores=subs,
        flags=flags,
        reasons=reasons,
        notes=notes,
    )


def _is_finite(v: float) -> bool:
    return v == v and v not in (float("inf"), float("-inf"))


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
