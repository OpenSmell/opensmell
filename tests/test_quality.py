"""MOX quality scoring on the new .osmell path."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import opensmell as o
from conftest import CHANNELS, make_file
from opensmell.types import OsmellFile, OsmellManifest


def _quality(f, **kw):
    sample_count = kw.pop("sample_count", len(f.time))
    return o.compute_quality(
        f,
        sample_count=sample_count,
        guess_sampling_rate_hz=10.0,
        **kw,
    )


def test_good_exposure_scores_high():
    q = _quality(make_file())
    assert q.format == "opensmell-quality"
    assert q.badge == "Excellent"
    assert q.total >= 90


def test_dead_sensor_flagged():
    f = make_file(dead_channels=["NO2"])
    q = _quality(f)
    assert "NO2" in q.flags.dead_sensors
    assert any("Dead sensors" in n for n in q.notes)


def test_single_role_skips_signal_strength():
    f = make_file(role="single")
    q = _quality(f)
    assert q.subscores["signalStrength"].value is None
    assert q.total is not None, "total must be renormalized over available sub-scores"


def test_no_baseline_scores_zero_stability():
    f = make_file()
    f.manifest.baseline.source = "none"
    q = _quality(f)
    assert q.subscores["baselineStability"].value == 0
    assert q.subscores["baselineStability"].reason == "no_baseline"


def test_short_duration_penalized():
    q = _quality(make_file(n=60))
    assert q.subscores["durationAdequacy"].reason == "too_short"


def test_miris_routing_not_implemented():
    f = make_file()
    f.manifest.sensor.sensor_type = "miris"
    with pytest.raises(NotImplementedError, match="miris"):
        _quality(f)


def test_unknown_sensor_type_not_implemented():
    f = make_file()
    f.manifest.sensor.sensor_type = "alien"
    with pytest.raises(NotImplementedError, match="alien"):
        _quality(f)


# --- Regressions for the fixed defects found in the calibration study ---
# See docs/quality-weight-calibration.md. Each test asserts the corrected
# behaviour, so a regression is caught without needing the calibration corpus.


def test_time_column_in_seconds_is_flagged_not_scored_as_packet_loss():
    """Defect 1: a seconds-valued time column must not read as irregular gaps.

    Continuity used to collapse to 0 with reason "irregular_gaps", which is
    indistinguishable from real packet loss.
    """
    f = make_file()
    f.time = [t / 1000.0 for t in f.time]  # seconds, not milliseconds
    f.manifest.sensor.sampling_rate_hz = 10.0

    q = _quality(f)

    assert q.subscores["continuity"].value is None
    assert q.subscores["continuity"].reason == "time_unit_mismatch"
    assert q.flags.time_unit_mismatch is True
    assert any("milliseconds" in n for n in q.notes)


def test_millisecond_time_column_is_not_flagged():
    """The unit check must not fire on a correctly-scaled recording."""
    f = make_file()
    f.manifest.sensor.sampling_rate_hz = 10.0

    q = _quality(f)

    assert q.flags.time_unit_mismatch is False
    assert q.subscores["continuity"].reason == "ok"
    assert q.subscores["continuity"].value == 100.0


def test_gap_variability_within_tolerance_is_not_a_unit_mismatch():
    """Real jitter must still be scored, not mistaken for a unit fault."""
    f = make_file()
    f.manifest.sensor.sampling_rate_hz = 10.0
    # 10 Hz nominal is 100 ms; stretch a few gaps by 1.5x, well inside the
    # 0.5x-2.0x plausibility band, so this is jitter rather than bad units.
    f.time = [t + (30.0 if i % 37 == 0 else 0.0) for i, t in enumerate(f.time)]

    q = _quality(f)

    assert q.flags.time_unit_mismatch is False
    assert q.subscores["continuity"].value is not None


def test_noise_burst_does_not_raise_dynamic_range():
    """Defect 2: interference must not inflate the dynamic-range score.

    Span was raw max-min, so an interference burst raised it and a recording
    that had got worse scored better.
    """
    clean = make_file()
    noisy = make_file()
    noisy.data = {
        cid: [v + (60.0 if 300 <= i < 360 else 0.0) for i, v in enumerate(vals)]
        for cid, vals in noisy.data.items()
    }

    q_clean = _quality(clean)
    q_noisy = _quality(noisy)

    assert q_noisy.subscores["dynamicRange"].value <= q_clean.subscores["dynamicRange"].value


def test_real_exposure_still_earns_dynamic_range():
    """The noise correction must not flatten genuine dynamic range to zero."""
    q = _quality(make_file())

    assert q.subscores["dynamicRange"].value > 0.0
    assert q.subscores["dynamicRange"].reason == "ok"


def test_saturation_does_not_raise_dynamic_range():
    """Defect 3b: clipping to the ADC rail must not widen the measured span.

    A sample pinned at full scale is not a measurement of the chemistry, so
    counting it as part of the channel's span gave a saturated recording a
    *higher* dynamic range than the same channel read below the rail.
    """
    clean = make_file(adc_max=4095.0)
    clipped = make_file(adc_max=4095.0)
    clipped.data = {
        cid: [
            4095.0 if 300 <= i < 360 else v
            for i, v in enumerate(vals)
        ]
        for cid, vals in clipped.data.items()
    }

    q_clean = _quality(clean)
    q_clipped = _quality(clipped)

    assert (
        q_clipped.subscores["dynamicRange"].value
        <= q_clean.subscores["dynamicRange"].value
    )
    assert q_clipped.subscores["saturationFree"].value < q_clean.subscores["saturationFree"].value
    assert q_clipped.total < q_clean.total


def test_unclipped_exposure_still_earns_dynamic_range():
    """The rail exclusion must not flatten genuine dynamic range to zero."""
    q = _quality(make_file())

    assert q.subscores["dynamicRange"].value > 0.0
    assert q.subscores["dynamicRange"].reason == "ok"


def test_dead_sensor_lowers_total():
    """Defect 4: losing a channel must not improve the score.

    Dead channels are excluded from the live-channel means, so before the fix
    a dead element raised the total by lifting the mean over the survivors.
    """
    live = _quality(make_file())
    dead = _quality(make_file(dead_channels=[CHANNELS[0]]))

    assert dead.flags.dead_sensors == [CHANNELS[0]]
    assert dead.total < live.total
    assert any("dead channel" in n for n in dead.notes)


def test_dead_sensor_penalty_scales_with_channel_count():
    """Two dead channels must cost more than one."""
    one = _quality(make_file(dead_channels=[CHANNELS[0]]))
    two = _quality(make_file(dead_channels=[CHANNELS[0], CHANNELS[1]]))

    assert two.total < one.total


def test_no_dead_sensors_adds_no_penalty_note():
    q = _quality(make_file())

    assert q.flags.dead_sensors == []
    assert not any("dead channel" in n for n in q.notes)
