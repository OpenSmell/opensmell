"""Timing discipline for sequentially-sampled channels.

An ESP32 reads its ADC1 channels in a loop, so the six values in one reported
frame were not acquired at the same instant. The frame is timestamped when it is
*reported*, so every channel in it is credited with the frame time and none is
credited with the microseconds it actually lagged by.

Two consequences matter for anything that reasons about time:

1. Cross-channel comparisons during a sharp common-mode event are biased by the
   sweep. `de_skew` puts the channels back on a common time base.
2. A reported cadence is a Nyquist statement about what is resolvable at all.
   `min_detectable_duration` reports the floor, so a caller can decline to claim
   an event shorter than the sampling can express.

Measured ESP32 sweep is roughly 1-12 ms against a 500 ms sample period. That is
negligible for second-scale MOX chemistry, so de-skewing is optional and off by
default: it costs a tail of NaNs, and applying it unconditionally would discard
real samples in exchange for a correction two orders of magnitude below the noise
floor. It matters when a consumer wants channel coherence during a fast transient,
and for multi-channel arrays whose sweep is long relative to their sample period.

The source of truth for the underlying measurements is
`electronic-nose/SAMPLING_CONTRACT.md` ("Sequential ADC skew").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import numpy as np

# De-skewing beyond this fraction of the sample period is not a rounding fix, it
# is a structural claim that the reported cadence does not describe the data.
MAX_SKEW_FRACTION_OF_PERIOD = 0.5

# Below this many samples per channel there is not enough data to estimate a sweep
# or to say anything about cadence variability.
MIN_SAMPLES_FOR_TIMING = 3


def scan_offsets(n_channels: int, sweep_s: float) -> np.ndarray:
    """Acquisition offset per channel, in seconds from the frame timestamp.

    Channel 0 is read first and so carries no lag; channel k is read k sweeps
    after the frame was stamped. Assumes a fixed sweep, which is what a bare
    `analogRead` loop gives.
    """
    if n_channels < 1:
        raise ValueError(f"n_channels must be >= 1, got {n_channels}")
    if sweep_s < 0:
        raise ValueError(f"sweep_s must be >= 0, got {sweep_s!r}")
    return np.arange(n_channels, dtype=float) * float(sweep_s)


def de_skew(
    series: np.ndarray,
    sweep_s: float,
    sample_period_s: float,
) -> np.ndarray:
    """Shift each channel onto a common time base.

    `series` is `(n_samples, n_channels)`, indexed by frame. Channel k's sample
    at frame `i` was really acquired at `t_i + k * sweep_s`, so its value belongs
    at common-time index `i - shift_k`. The shift is applied by dropping a
    leading slice of each channel and leaving the tail as NaN, because the data
    for those frames does not exist yet -- extrapolating them would invent the
    samples that a cross-channel comparison is most sensitive to.

    `sample_period_s` is required rather than inferred: the sample index carries
    no time information, and guessing a period from a frame count is exactly the
    kind of assumption this module exists to avoid.

    The residual sub-sample part of the sweep is not corrected. With a 12 ms
    sweep at 2 Hz that is 2.4% of a sample, below the ADC's own quantisation;
    `timing_report` reports it so the caller can confirm that rather than assume it.
    """
    arr = np.asarray(series, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"series must be 2-D (samples, channels), got shape {arr.shape}")
    n_samples, n_channels = arr.shape
    if n_channels < 1:
        raise ValueError("series must have at least one channel")
    if sample_period_s <= 0:
        raise ValueError(f"sample_period_s must be > 0, got {sample_period_s!r}")

    shifts = np.floor(scan_offsets(n_channels, sweep_s) / float(sample_period_s)).astype(int)
    shifts = np.clip(shifts, 0, max(n_samples - 1, 0))

    out = np.full_like(arr, np.nan)
    for ch, shift in enumerate(shifts):
        if shift == 0:
            out[:, ch] = arr[:, ch]
        else:
            out[: n_samples - shift, ch] = arr[shift:, ch]
    return out


def min_detectable_duration(sampling_rate_hz: float) -> float:
    """Shortest event the sample sequence can resolve, in seconds.

    An event occupying fewer than about two samples cannot be separated from the
    sampling itself: it may fall entirely between two frames and leave no trace,
    or be split across two frames in a way that is indistinguishable from noise.
    The threshold is `2 dt`, not `dt`, because a one-sample feature has no way to
    distinguish "the event happened between samples" from "the event did not
    happen".

    This is a floor on resolvability, not on detectability. A longer event can
    still be undetectable if its amplitude is below the noise floor; that
    requires the channel's noise level and is a separate question.
    """
    if sampling_rate_hz <= 0:
        raise ValueError(f"sampling_rate_hz must be > 0, got {sampling_rate_hz!r}")
    return 2.0 / float(sampling_rate_hz)


def is_resolvable(duration_s: float, sampling_rate_hz: float) -> bool:
    """Whether an event of `duration_s` is expressible at this cadence."""
    if duration_s < 0:
        raise ValueError(f"duration_s must be >= 0, got {duration_s!r}")
    return duration_s >= min_detectable_duration(sampling_rate_hz)


def measure_period(time: Sequence[float]) -> float:
    """Median inter-sample gap, the cadence the timestamps actually show.

    Measured beats declared: a firmware build constant can be wrong, and the
    timestamps are what the analysis will divide by. Non-positive steps are
    dropped rather than included, so a duplicated frame does not deflate the
    estimate.
    """
    t = np.asarray(list(time), dtype=float)
    if t.size < 2:
        raise ValueError(f"need at least 2 timestamps, got {t.size}")
    gaps = np.diff(t)
    positive = gaps[gaps > 0]
    if positive.size == 0:
        raise ValueError("timestamps contain no positive step")
    return float(np.median(positive))


@dataclass
class TimingReport:
    """What the sample sequence can and cannot support."""

    n_samples: int
    n_channels: int
    sampling_rate_hz: Optional[float]
    sample_period_s: Optional[float]
    rate_source: str
    min_detectable_duration_s: Optional[float]
    sweep_s: Optional[float] = None
    max_skew_s: Optional[float] = None
    max_skew_fraction_of_period: Optional[float] = None
    residual_sub_sample_skew_s: Optional[float] = None
    flags: List[str] = field(default_factory=list)

    def resolvable(self, duration_s: float) -> bool:
        """Whether an event of this duration can be expressed at this cadence."""
        if self.min_detectable_duration_s is None:
            return False
        return duration_s >= self.min_detectable_duration_s

    def needs_de_skew(self) -> bool:
        """Whether cross-channel coherence here requires de-skewing.

        False when the sweep is unknown or small against the sample period. The
        threshold is half a period: below that the skew cannot move a channel by
        a full sample, so it cannot reorder or misalign a comparison.
        """
        if self.max_skew_fraction_of_period is None:
            return False
        return self.max_skew_fraction_of_period > MAX_SKEW_FRACTION_OF_PERIOD


def timing_report(
    time: Optional[Sequence[float]] = None,
    sampling_rate_hz: Optional[float] = None,
    n_channels: Optional[int] = None,
    sweep_s: Optional[float] = None,
) -> TimingReport:
    """Describe the timing limits of a recording.

    Cadence is taken from the timestamps when they are available and from
    `sampling_rate_hz` otherwise, recording which in `rate_source`. Neither
    source is trusted over the other when both exist: the cross-check belongs to
    the ingestion gate in `electronic-nose/SAMPLING_CONTRACT.md`, which flags a
    disagreement of more than 2x as a likely time-unit error.
    """
    flags: List[str] = []

    n_obs = 0 if time is None else len(time)
    if n_obs and n_obs < MIN_SAMPLES_FOR_TIMING:
        flags.append("too_few_samples_to_characterise_timing")

    period_s: Optional[float] = None
    rate: Optional[float] = None
    if time is not None and n_obs >= 2:
        try:
            period_s = measure_period(time)
            rate = 1.0 / period_s
            source = "timestamps"
        except ValueError:
            flags.append("timestamps_contain_no_positive_step")
            source = "unavailable"
    elif time is not None:
        source = "unavailable"
    else:
        source = "declared" if sampling_rate_hz is not None else "unavailable"

    if rate is None and sampling_rate_hz is not None and sampling_rate_hz > 0:
        rate = float(sampling_rate_hz)
        period_s = 1.0 / rate
        if source == "unavailable":
            source = "declared"

    if source == "unavailable":
        flags.append("cadence_unknown")

    if sampling_rate_hz is not None and sampling_rate_hz <= 0:
        flags.append("declared_rate_not_positive")
    elif rate is not None and rate < 1.0:
        flags.append("cadence_below_1hz")

    channels = int(n_channels) if n_channels is not None else 0

    max_skew = None
    skew_fraction = None
    residual = None
    if sweep_s is not None and channels > 1:
        max_skew = sweep_s * (channels - 1)
        if period_s is not None and period_s > 0:
            skew_fraction = max_skew / period_s
            shifts = scan_offsets(channels, sweep_s) / period_s
            residual = float(np.max(np.abs(shifts - np.round(shifts)))) * period_s
            if skew_fraction > MAX_SKEW_FRACTION_OF_PERIOD:
                flags.append("channel_sweep_exceeds_half_sample_period")
    elif sweep_s is not None and channels <= 1:
        flags.append("single_channel_so_sweep_is_irrelevant")

    floor = min_detectable_duration(rate) if rate is not None else None

    return TimingReport(
        n_samples=n_obs,
        n_channels=channels,
        sampling_rate_hz=rate,
        sample_period_s=period_s,
        rate_source=source,
        min_detectable_duration_s=floor,
        sweep_s=sweep_s,
        max_skew_s=max_skew,
        max_skew_fraction_of_period=skew_fraction,
        residual_sub_sample_skew_s=residual,
        flags=flags,
    )


def restrict_to_resolvable(
    duration_s: float, sampling_rate_hz: float
) -> Union[float, None]:
    """Return `duration_s` if it is resolvable, else None.

    For callers that must not report a sub-resolution duration: a returned None
    is a refusal to claim the event, which is safer than passing on a number the
    sample sequence cannot support.
    """
    return duration_s if is_resolvable(duration_s, sampling_rate_hz) else None