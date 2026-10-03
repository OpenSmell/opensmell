"""Tests for the sequential-sampling timing module."""

import numpy as np
import pytest

from opensmell.mox.timing import (
    MAX_SKEW_FRACTION_OF_PERIOD,
    de_skew,
    is_resolvable,
    measure_period,
    min_detectable_duration,
    restrict_to_resolvable,
    scan_offsets,
    timing_report,
)


class TestMinDetectableDuration:
    def test_two_sample_periods(self):
        assert min_detectable_duration(2.0) == pytest.approx(1.0)
        assert min_detectable_duration(1.0) == pytest.approx(2.0)
        assert min_detectable_duration(10.0) == pytest.approx(0.2)

    def test_rejects_non_positive_rate(self):
        for bad in (0.0, -1.0, -0.5):
            with pytest.raises(ValueError, match="must be > 0"):
                min_detectable_duration(bad)

    def test_one_sample_is_not_resolvable(self):
        # A single-sample feature cannot distinguish "happened between samples"
        # from "did not happen", which is why the floor is 2 dt and not dt.
        assert not is_resolvable(0.5, 2.0)
        assert is_resolvable(1.0, 2.0)

    def test_boundary_is_inclusive(self):
        assert is_resolvable(1.0, 2.0)
        assert is_resolvable(1.0000001, 2.0)
        assert not is_resolvable(0.9999999, 2.0)

    def test_rejects_negative_duration(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            is_resolvable(-1.0, 2.0)

    def test_restrict_returns_none_below_floor(self):
        assert restrict_to_resolvable(0.4, 2.0) is None
        assert restrict_to_resolvable(1.5, 2.0) == 1.5


class TestScanOffsets:
    def test_first_channel_has_no_lag(self):
        assert scan_offsets(6, 0.002)[0] == 0.0

    def test_offsets_are_evenly_spaced(self):
        off = scan_offsets(6, 0.002)
        assert off.tolist() == pytest.approx([0.0, 0.002, 0.004, 0.006, 0.008, 0.010])

    def test_zero_sweep_is_all_zero(self):
        assert scan_offsets(6, 0.0).tolist() == [0.0] * 6

    def test_rejects_bad_arguments(self):
        with pytest.raises(ValueError, match="n_channels"):
            scan_offsets(0, 0.001)
        with pytest.raises(ValueError, match="sweep_s"):
            scan_offsets(4, -0.001)


class TestDeSkew:
    def _ramp(self):
        # channel k carries a ramp offset by k so a shift is detectable
        n = 10
        return np.column_stack(
            [np.arange(n, dtype=float) + k for k in range(3)]
        )

    def test_shifts_later_channels_earlier(self):
        out = de_skew(self._ramp(), sweep_s=0.5, sample_period_s=0.5)
        # sweep == one period, so each channel shifts by exactly one frame
        # channel k holds (frame + k), so its frame-k sample lands at index 0
        assert out[0, 0] == pytest.approx(0.0)
        assert out[0, 1] == pytest.approx(1.0 + 1)
        assert out[0, 2] == pytest.approx(2.0 + 2)

    def test_tail_is_nan_not_extrapolated(self):
        out = de_skew(self._ramp(), sweep_s=0.5, sample_period_s=0.5)
        assert np.isnan(out[-1, 1])
        assert np.isnan(out[-1, 2])
        assert np.isnan(out[-2, 2])
        assert not np.isnan(out[-1, 0])

    def test_zero_sweep_is_identity(self):
        arr = self._ramp()
        out = de_skew(arr, sweep_s=0.0, sample_period_s=0.5)
        assert np.allclose(out, arr)

    def test_sub_period_sweep_rounds_to_no_shift(self):
        # 12 ms against a 500 ms period is 2.4% of a sample: below one frame, so
        # no shift is applied rather than an interpolated one.
        arr = self._ramp()
        out = de_skew(arr, sweep_s=0.012, sample_period_s=0.5)
        assert np.allclose(out, arr)

    def test_shift_never_exceeds_frame_count(self):
        # A sweep far larger than the recording would otherwise index out of range.
        # The shift is clamped to n-1, so the last channel keeps at most one real
        # sample and the rest of its column is NaN rather than an out-of-bounds read.
        arr = self._ramp()
        out = de_skew(arr, sweep_s=100.0, sample_period_s=0.5)
        assert out.shape == arr.shape
        assert not np.isnan(out[:, 0]).any()
        assert np.isnan(out[1:, 2]).all()
        assert np.isnan(out[1:, 1]).all()

    def test_rejects_bad_shapes(self):
        with pytest.raises(ValueError, match="2-D"):
            de_skew(np.arange(10.0), sweep_s=0.0, sample_period_s=0.5)
        with pytest.raises(ValueError, match="at least one channel"):
            de_skew(np.zeros((10, 0)), sweep_s=0.0, sample_period_s=0.5)
        with pytest.raises(ValueError, match="sample_period_s"):
            de_skew(self._ramp(), sweep_s=0.0, sample_period_s=0.0)


class TestMeasurePeriod:
    def test_median_gap(self):
        assert measure_period([0.0, 0.5, 1.0, 1.5]) == pytest.approx(0.5)

    def test_duplicate_frames_do_not_deflate_estimate(self):
        # A repeated timestamp carries no timing information and must not count.
        assert measure_period([0.0, 0.0, 0.5, 1.0, 1.5]) == pytest.approx(0.5)

    def test_jitter_uses_median_not_mean(self):
        gaps = [0.5] * 9 + [0.9]
        time = np.concatenate([[0.0], np.cumsum(gaps)])
        assert measure_period(time) == pytest.approx(0.5)

    def test_rejects_insufficient_and_non_advancing(self):
        with pytest.raises(ValueError, match="at least 2"):
            measure_period([0.0])
        with pytest.raises(ValueError, match="no positive step"):
            measure_period([1.0, 1.0, 1.0])


class TestTimingReport:
    def test_prefers_measured_cadence_over_declared(self):
        rep = timing_report(time=[0.0, 0.5, 1.0, 1.5], sampling_rate_hz=1.0, n_channels=6)
        assert rep.rate_source == "timestamps"
        assert rep.sampling_rate_hz == pytest.approx(2.0)

    def test_falls_back_to_declared_rate(self):
        rep = timing_report(time=None, sampling_rate_hz=2.0, n_channels=6)
        assert rep.rate_source == "declared"
        assert rep.sampling_rate_hz == pytest.approx(2.0)
        assert rep.min_detectable_duration_s == pytest.approx(1.0)

    def test_unknown_cadence_is_flagged_not_guessed(self):
        rep = timing_report(time=None, n_channels=6)
        assert rep.rate_source == "unavailable"
        assert rep.sampling_rate_hz is None
        assert rep.min_detectable_duration_s is None
        assert "cadence_unknown" in rep.flags
        assert not rep.resolvable(1000.0)

    def test_tiny_recording_is_flagged(self):
        rep = timing_report(time=[0.0, 0.5], n_channels=2)
        assert "too_few_samples_to_characterise_timing" in rep.flags

    def test_non_positive_declared_rate_is_flagged(self):
        rep = timing_report(time=None, sampling_rate_hz=0.0, n_channels=2)
        assert "declared_rate_not_positive" in rep.flags

    def test_sub_1hz_cadence_is_flagged(self):
        rep = timing_report(time=[0.0, 3.0, 6.0], n_channels=2)
        assert "cadence_below_1hz" in rep.flags

    def test_measured_sweep_is_reported(self):
        rep = timing_report(
            time=[i * 0.5 for i in range(10)], n_channels=6, sweep_s=0.012
        )
        assert rep.max_skew_s == pytest.approx(0.012 * 5)
        assert rep.max_skew_fraction_of_period == pytest.approx(0.012 * 5 / 0.5)
        assert rep.residual_sub_sample_skew_s == pytest.approx(0.012 * 5 % 0.5)

    def test_small_sweep_does_not_require_de_skew(self):
        # 60 ms total skew against a 500 ms period is well under half a frame,
        # so it cannot reorder or misalign a cross-channel comparison.
        rep = timing_report(time=[i * 0.5 for i in range(10)], n_channels=6, sweep_s=0.012)
        assert not rep.needs_de_skew()
        assert "channel_sweep_exceeds_half_sample_period" not in rep.flags

    def test_sweep_over_half_a_period_requires_de_skew(self):
        rep = timing_report(time=[i * 0.5 for i in range(10)], n_channels=6, sweep_s=0.06)
        assert rep.needs_de_skew()
        assert "channel_sweep_exceeds_half_sample_period" in rep.flags

    def test_single_channel_sweep_is_irrelevant(self):
        rep = timing_report(time=[i * 0.5 for i in range(10)], n_channels=1, sweep_s=0.5)
        assert "single_channel_so_sweep_is_irrelevant" in rep.flags
        assert not rep.needs_de_skew()

    def test_unknown_sweep_does_not_force_de_skew(self):
        rep = timing_report(time=[i * 0.5 for i in range(10)], n_channels=6, sweep_s=None)
        assert not rep.needs_de_skew()
        assert rep.max_skew_s is None


class TestContractConsistency:
    def test_threshold_matches_sampling_contract(self):
        # The SAMPLING_CONTRACT rule is "an event shorter than about 2 dt cannot
        # be resolved". This pins the implementation to that wording.
        assert min_detectable_duration(2.0) == pytest.approx(2 * (1 / 2.0))

    def test_measured_esp32_sweep_is_below_threshold(self):
        # Worst measured sweep (12 ms x 5 gaps = 60 ms) at the nominal 2 Hz.
        rep = timing_report(time=[i * 0.5 for i in range(100)], n_channels=6, sweep_s=0.012)
        assert rep.max_skew_s == pytest.approx(0.060)
        assert rep.max_skew_fraction_of_period < MAX_SKEW_FRACTION_OF_PERIOD