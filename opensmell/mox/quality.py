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

PROVISIONAL WEIGHTS
-------------------
The values in `WEIGHTS` are **provisional and deliberately not yet calibrated**.
They encode the authors' judgement about which failure modes matter most, not a
measured optimum. Do not cite them as an optimum, and do not tune them against a
single corpus.

`tools/derive_quality_weights.py` and `tools/stress_quality_corpus.py` implement
the calibration study and both currently report that calibration is impossible
with the available data:

* SmellNet-Base (120 sessions) exercises none of the failure modes. Four of the
  seven subscores are never computable there and three are constant across every
  file, so they carry no between-recording information.
* Injected-defect studies show the scorer is metadata-dependent. It declines to
  evaluate `baselineStability` unless the manifest declares a baseline, and can
  only detect upper-rail saturation against a correctly declared `adcMax`.
* Split-half reliability is unmeasurable or negative for most subscores on
  protocol-shaped synthetic recordings, because clean recordings score nearly
  identically and reliability requires between-recording variance.

Five concrete defects were identified. Four were logic bugs and are now fixed; the
fifth needs the corpus and is the remaining blocker.

The four fixed defects come in two shapes, and both are reasons to distrust a
quality score that has not been shown monotone in every input. Three of them —
noise, saturation, and a dead channel — **raised** the score: worse data scored
better. The fourth, a mislabelled time column, did the opposite and is the more
instructive of the two. It did not reward bad data; it made a clean recording
score identically to a broken one, which destroys the subscore's ability to
report on anything at all. A score that improves for worse data is wrong, and a
score that reports the same number for two opposite faults is not measuring
anything.

1. FIXED. The time column is assumed to be **milliseconds**. When the observed
   median gap is far from the period implied by `samplingRateHz`, continuity is
   now withheld with reason `time_unit_mismatch` and flagged via
   `flags.time_unit_mismatch`, rather than silently reporting 0 for
   `irregular_gaps`. Real jitter inside the 0.5x-2.0x band is still scored.
2. FIXED. `dynamicRange` was **noise-rewarding**, because raw `span / adcMax` is
   inflated by interference; a burst raised the score. Span is now a robust
   5th-95th percentile range minus three times the channel's noise floor, with
   the floor estimated from successive differences rather than the overall
   standard deviation (which includes the exposure and would drive clean
   recordings to zero). Span is also measured over unclipped samples only.
3. FIXED. **Saturation rewarded itself.** A sample pinned at the converter rail
   carries no amplitude information -- the true peak is unknown and lies above
   full scale -- yet rail samples were counted both in the percentile span and as
   the peak amplitude of `signalStrength`. A saturated channel therefore scored a
   wider dynamic range and a stronger signal than the same channel read in range,
   and the total rose. Both now measure over unclipped samples; `saturationFree`
   still reports the clipping, so no information is lost. Requires a manifest
   declaring `adcMax`.
4. FIXED. **Dead sensors raised `dynamicRange`**, because dead channels are
   excluded from the live-channel mean, so losing hardware lifted the average.
   Each dead channel now costs `DEAD_SENSOR_PENALTY` off the total.
5. OPEN, needs the corpus. Subscores are **not on a common scale**.
   `continuity` and `durationAdequacy` are per-recording and move by 80+ points,
   while `saturationFree` is a per-channel mean, so saturating one of six channels
   fully moves it by at most 16.7. Summing them under a shared weight vector is
   unprincipled until they are placed on one scale. This is the single remaining
   blocker for calibration.

Calibration still requires a corpus with genuine between-recording variance (varied
noise, baseline quality and SNR) *and* known ground-truth defects. See
`docs/quality-weight-calibration.md` for the full study, the fixes, the remaining
blocker, and the data needed to close it.
"""

from __future__ import annotations

from typing import List, Optional

from ..normalize import mean, median
from ..types import (
    DEFAULT_ADC_MAX,
    DEAD_SENSOR_PENALTY,
    FULL_SCORE_DURATION_S,
    GAP_TOLERANCE,
    MAX_TIME_UNIT_RATIO,
    MIN_SPAN_FRACTION,
    MIN_TIME_UNIT_RATIO,
    NOISE_CV_LIMIT,
    NOISE_SPAN_TOLERANCE,
    SNR_TARGET,
    ChannelStats,
    OsmellFile,
    QualityFlags,
    QualityReport,
    SubScore,
)
from .normalize import baseline_for_channel, channel_stats, normalized_series

# PROVISIONAL: not calibrated against data. See the module docstring and
# docs/quality-weight-calibration.md. Do not tune against a single corpus.
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

    # The time column is assumed to be milliseconds, matching the
    # `timestamp_ms` contract. If a caller supplies seconds the nominal period
    # below would be wrong by 1000x and continuity would collapse to 0 with
    # reason "irregular_gaps", which looks identical to real packet loss.
    # Detect the unit mismatch explicitly instead of reporting a plausible-
    # looking score for a recording whose timestamps were never in the
    # documented unit.
    observed_median = median(positive_gaps) if positive_gaps else None
    time_unit_mismatch = False
    if observed_median is not None and rate_declared:
        expected = 1000.0 / sampling_rate_hz if sampling_rate_hz and sampling_rate_hz > 0 else None
        if expected is not None and expected > 0:
            ratio = observed_median / expected
            if ratio < MIN_TIME_UNIT_RATIO or ratio > MAX_TIME_UNIT_RATIO:
                time_unit_mismatch = True
                flags.time_unit_mismatch = True
                notes.append(
                    f"median gap {observed_median:g} ms is {ratio:.4g}x the expected "
                    f"{expected:g} ms for {sampling_rate_hz:g} Hz; the time column is "
                    "probably in seconds or microseconds rather than milliseconds. "
                    "Continuity is not reported."
                )

    if sample_count < 2:
        continuity = SubScore(value=100.0, reason="ok")
    elif time_unit_mismatch:
        continuity = SubScore(value=None, reason="time_unit_mismatch")
    else:
        if rate_declared:
            nominal = 1000.0 / sampling_rate_hz if sampling_rate_hz and sampling_rate_hz > 0 else None
        else:
            nominal = observed_median
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
    # Span is measured robustly (5th-95th percentile) and then the noise
    # contribution is subtracted before scaling. A raw max-min span rewards
    # interference: an interference burst inflates max and min, so span grows
    # and the score rises for a recording that got worse. Subtracting a
    # multiple of the channel's own noise floor means span only earns credit
    # for variation that exceeds the noise.
    def _net_span(s: ChannelStats) -> float:
        values = file.data.get(s.id, [])
        finite = [v for v in values if _is_finite(v)]
        # Samples sitting on the converter rail are not measurements of the
        # chemistry. Including them inflates the span, so a channel driven into
        # saturation scores a *wider* dynamic range than the same channel read
        # below the rail -- the same reward-for-worse-data failure the noise
        # correction below fixes. Span is therefore measured over the unclipped
        # samples only; saturation itself is scored separately by
        # `saturationFree`, so discarding these samples here loses no signal.
        unclipped = [v for v in finite if not (adc_declared and (v >= adc_max or v <= 0))]
        if len(unclipped) < 5:
            return 0.0
        ordered = sorted(unclipped)
        k = max(1, int(0.05 * (len(ordered) - 1)))
        robust = ordered[-1 - k] - ordered[k]
        # The noise floor must be estimated from sample-to-sample variation, not
        # from the overall standard deviation. The overall std includes the
        # exposure itself, so subtracting a multiple of it would subtract the
        # signal we are trying to measure and drive the score to zero for every
        # clean recording. The median absolute successive difference divided by
        # sqrt(2) estimates the noise of a random walk while largely ignoring
        # the slow chemical response.
        diffs = [abs(unclipped[i + 1] - unclipped[i]) for i in range(len(unclipped) - 1)]
        noise = median(diffs) / 1.4142135623730951 if diffs else 0.0
        return max(0.0, robust - NOISE_SPAN_TOLERANCE * noise)

    dynamic_value = 0.0 if not live else 100.0 * mean(
        [_clamp((_net_span(s) / adc_max) * (1.0 / MIN_SPAN_FRACTION), 0.0, 1.0) for s in live]
    )
    dynamic_range = SubScore(
        value=dynamic_value,
        reason="low_span" if dynamic_value < 50 else "ok",
    )
    if dynamic_range.reason == "low_span":
        reasons["dynamicRange"] = "channel_span_below_10_percent_of_adc_range_after_noise_correction"

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
            # A sample pinned at the converter rail carries no amplitude
            # information: the true peak is unknown and lies somewhere above
            # full scale. Counting it as the peak would let saturation *raise*
            # the SNR score, so rail samples are excluded from the normalised
            # series used for peak and recovery. `saturationFree` reports the
            # clipping itself.
            usable = (
                [v for v in values if not (adc_declared and (v >= adc_max or v <= 0))]
                if adc_declared
                else list(values)
            )
            norm = [v for v in normalized_series(usable, r0) if _is_finite(v)]
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
    # A dead sensing element is excluded from the live-channel means because a
    # constant channel carries no span, recovery or signal information. That
    # exclusion would otherwise *raise* the total, because dropping a
    # zero-span channel lifts the mean over the remaining ones. Each dead
    # channel therefore costs a fixed penalty, so losing hardware reads as the
    # degradation it is.
    if total is not None and flags.dead_sensors:
        n_dead = len(flags.dead_sensors)
        penalty = min(100.0, DEAD_SENSOR_PENALTY * n_dead)
        total = round(max(0.0, total - penalty))
        notes.append(
            f"{n_dead} dead channel(s) ({', '.join(flags.dead_sensors)}) "
            f"reduced the total by {penalty:g}."
        )

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
