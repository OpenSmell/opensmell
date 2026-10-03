# Quality-weight calibration: status, method, and what data would close it

**Status: open. The weights in `opensmell.mox.quality.WEIGHTS` are provisional
and are not calibrated against data.**

This document exists so the gap is visible to anyone reading the code, so nobody
cites the weight vector as an optimum, and so the community knows what data would
turn it into a measured result.

## Why the weights are provisional

`WEIGHTS` encodes a judgement about which failure modes matter most for metal-oxide
recordings. It is not the output of a fitting procedure. Three separate attempts to
calibrate it all failed, and the failures are themselves informative.

### Attempt 1 — real SmellNet-Base sessions (120 files)

`tools/derive_quality_weights.py`

Split-half reliability times discriminative range, measured on real recordings.
Result: unusable. Four of seven subscores were never computable and three were
constant across all 300 files.

| subscore | split-half | range | measurable |
|---|---|---|---|
| continuity | n/a | 0.000 | 1.00 |
| dynamicRange | +0.853 | 0.027 | 1.00 |
| saturationFree | +0.773 | 0.000 | 1.00 |
| baselineStability | n/a | 0.000 | 1.00 |
| signalStrength | n/a | 0.000 | 0.00 |
| recoveryCompleteness | n/a | 0.000 | 0.00 |
| durationAdequacy | n/a | 0.000 | 1.00 |

SmellNet-Base is 300 clean, uniform, perfectly regular 1 Hz ten-minute recordings
with no baseline phase and no recovery phase. That is precisely the corpus in which
a quality score has nothing to detect. Only `dynamicRange` and `saturationFree`
were measurable, and neither varied.

### Attempt 2 — defects injected into raw SmellNet

The score barely moved. This was not a scoring bug; it exposed a **design
property**: the scorer is metadata-dependent.

- `baselineStability` returns `0.0 / "no_baseline"` and never inspects the data
  unless the manifest declares a baseline source.
- Upper-rail saturation is only detectable when `adcMax` is declared.
- Dropped rows do not register as gaps unless the time vector is genuinely
  non-uniform.

This is defensible — the manifest is the data contract, and a scorer that guesses
at missing provenance is worse than one that refuses. But it means a validation
corpus must vary the *declared metadata* along with the samples.

### Attempt 3 — protocol-shaped synthetic recordings with known ground truth

`tools/stress_quality_corpus.py`, 40 recordings x 7 injected-defect conditions.

Synthetic baseline, first-order exposure and first-order recovery, six channels
with independent time constants. Each recording scored clean and with one injected
defect. Result: all seven subscores became computable, but **split-half
reliability was unmeasurable or negative for five of seven**, because clean
recordings score nearly identically:

| subscore | sensitivity | specificity | reliability |
|---|---|---|---|
| continuity | 9.5 | 0.000 | n/a |
| dynamicRange | 79.4 | 0.125 | **-0.283** |
| saturationFree | 0.5 | 0.000 | n/a |
| baselineStability | 76.6 | 0.146 | +0.980 |
| signalStrength | 147.5 | 0.000 | n/a |
| recoveryCompleteness | 46.5 | 0.002 | **-0.130** |
| durationAdequacy | 90.8 | 0.000 | n/a |

Split-half reliability is undefined when a subscore is constant across recordings.
Any corpus that made the weights estimable here would have collapsed them onto a
single subscore, which would be an artifact of the corpus rather than a property of
the instrument.

## Five defects

Four of the five were logic bugs and have since been fixed. They were independent of
the weight question, so they did not need the corpus. The fifth is a design question
that genuinely does need one.

Three of the four fixed defects share one shape: a defect in the recording *raised*
the score. Noise, saturation, and a failed element each improved a quality score for
data that had got worse. The time-column defect was the opposite failure, and is
arguably the more instructive of the two kinds: it did not reward bad data, it made
a clean recording indistinguishable from a broken one. Both kinds break the same
property, so both matter. A quality score is not evidence of quality until every one
of its inputs has been shown to be monotone in the right direction *and* to
distinguish the failure modes it claims to detect.

### 1. The time column is assumed to be milliseconds — FIXED

`nominal` is derived as `1000.0 / sampling_rate_hz`, then compared against raw
differences of `file.time`. If a caller supplies seconds, continuity collapses to
`0.0` with reason `irregular_gaps` and no hint that the units are wrong. A clean
recording scores zero for a reason that looks like real packet loss.

*Fix applied:* the observed median gap is now compared against the expected period
for the declared rate, and continuity is withheld with reason `time_unit_mismatch`
and a new `flags.time_unit_mismatch` when the ratio falls outside 0.5x to 2.0x. A
seconds-valued column now yields `None` and an explicit note instead of a
plausible-looking 0. Real jitter inside the band is still scored normally.
Regression tests: `test_time_column_in_seconds_is_flagged_not_scored_as_packet_loss`,
`test_millisecond_time_column_is_not_flagged`,
`test_gap_variability_within_tolerance_is_not_a_unit_mismatch`.

### 2. `dynamicRange` is noise-rewarding — FIXED

The subscore is `100 * mean(span / adcMax)`. Span is inflated by noise, so an
interference burst *raises* the score (measured: `+9.2`). Split-half reliability is
`-0.283`, meaning it does not report a stable property of the recording at all.

*Fix applied:* span is now a robust 5th-95th percentile range minus three times the
channel's noise floor, where the noise floor is estimated from the median absolute
successive difference rather than the overall standard deviation. The overall std
includes the exposure itself, so using it would subtract the signal being measured
and drive every clean recording to zero; the successive-difference estimate largely
ignores the slow chemical response. An interference burst now moves the score by
`+0.0` instead of `+9.2`. The reason string for a low span is now
`channel_span_below_10_percent_of_adc_range_after_noise_correction`.
Regression tests: `test_noise_burst_does_not_raise_dynamic_range`,
`test_real_exposure_still_earns_dynamic_range`.

Note that this corrects the *direction* of the defect but does not make the
subscore well-conditioned: its split-half reliability is still negative (-0.49 on
the stress corpus). See defect 3.

### 3. Subscores are not on a common scale — OPEN, needs the corpus

`continuity` and `durationAdequacy` are per-recording and move by 80+ points when a
recording is bad. `saturationFree` is a per-channel *mean*, so saturating one of
six channels fully moves it by at most 16.7, and clipping 5% of one channel by
0.8. Summing subscores under a shared weight vector is unprincipled while they sit
on different scales. This is the deepest of the four, because it undermines the
weighting scheme itself rather than any single factor.

*Why this one is still open:* the fix is to rescale each subscore so that a defect
of known severity produces a known score drop. Choosing those reference severities
requires knowing which defects matter and how much, which is a judgement about the
instrument that only real data can inform. Until then, compare subscores within a
recording rather than across devices; the README says so.

This is now the single blocking issue for calibrating the weights.

### 4. Dead sensors raise `dynamicRange` — FIXED

Dead channels are excluded from the live-channel mean, so removing a failed element
*raises* the score. Detection lives only in `flags.dead_sensors` and never reaches
the total.

*Fix applied:* each dead channel now costs `DEAD_SENSOR_PENALTY` (12.5) off the
total, capped at 100, with a note naming the channels. Two dead channels cost more
than one. On the parity fixture the total moves from 53 (both live) to 42 (one dead)
to 30 (two dead). Regression tests:
`test_dead_sensor_lowers_total`, `test_dead_sensor_penalty_scales_with_channel_count`,
`test_no_dead_sensors_adds_no_penalty_note`.

### 5. Saturation rewards itself — FIXED

Found while probing defect 2 rather than by the weight study, and it is the mirror
image of defect 2. A sample pinned at the converter rail carries no amplitude
information — the true peak is unknown and lies somewhere above full scale — but the
scorer treated it as a real peak. Driving one channel into saturation therefore
*widened* the measured span and *raised* the SNR score:

| condition | `dynamicRange` | `signalStrength` | `saturationFree` | total |
|---|---|---|---|---|
| clean (in range) | 100.0 | 26.9 | 100.0 | 65 |
| top 40 samples pinned to 4095 | 100.0 | 39.3 | 93.3 | 67 |

A saturated recording outscored a clean one. Two subscores were affected, not one:
`dynamicRange` counted rail samples inside the percentile span, and the peak term of
`signalStrength` counted them as signal amplitude.

*Fix applied:* both now measure over unclipped samples only. `saturationFree` is
unchanged and still reports the clipping, so nothing is lost by excluding those
samples elsewhere — the information is simply not usable as amplitude. The totals now
move 65 → 61 for the same pair, and `signalStrength` falls 26.9 → 10.2 rather than
rising. Regression tests: `test_saturation_does_not_raise_dynamic_range`,
`test_unclipped_exposure_still_earns_dynamic_range`.

This fix requires a manifest that declares `adcMax`. Without it the scorer has no rail
to identify, so `flags.used_default_adc_max` is set and clipping stays invisible. That
is the metadata dependence described above, working as designed — but it means a
contributor who omits `adcMax` silently forfeits this check.

## What data would close this

Calibration needs all three of the following. Any one missing is enough to make the
weights unfittable.

1. **Between-recording variance.** Real recordings spanning a genuine range of
   noise levels, baseline quality and SNR. Roughly 50 to 100 sessions is enough to
   estimate rank correlations, provided the subscores actually vary. The June 13
   2025 rig recordings qualify; the August recordings do not, because they carry no
   timestamps and their durations are unusable.
2. **Known ground-truth defects.** A documented subset where the defect is known in
   advance, so sensitivity can be measured rather than assumed. `tools/stress_quality_corpus.py`
   generates a starting point, but synthetic defects validate the scorer against
   the scorer's own model of the signal, not against the signal.
3. **At least three distinct defect types per subscore.** One defect cannot separate
   sensitivity from coincidence. The synthetic corpus provides seven.

### Call for contributions

Genuinely useful contributions, roughly in order of value:

- **Labeled defective recordings.** Any MOX recording with a *known* defect: a
  disconnected element, a saturated channel, a documented clock problem, an
  aborted acquisition. Metadata about the defect is worth more than the samples.
- **Real cross-device repetitions.** The same physical exposure recorded on two or
  more devices, ideally with a declared baseline and recovery phase. This is the
  corpus that does not currently exist anywhere, and it is what would make the
  subscores comparable across hardware.
- **Protocol-shaped real recordings.** Real sensors following a documented protocol
  with a declared pre-exposure baseline and a post-exposure purge. These make G and
  R computable, which real unstructured recordings cannot.
- **Rust/Python parity fixtures.** The Rust implementation in `opensmell-rs` should
  produce byte-identical reports for the same inputs. No parity harness exists yet.

### Contributing

`tools/derive_quality_weights.py` and `tools/stress_quality_corpus.py` are the
harness. Both take `--root`, so pointing them at a new corpus is the whole workflow:

```bash
python -m tools.derive_quality_weights --root /path/to/corpus --limit 150
python -m tools.stress_quality_corpus --n 40
```

A useful contribution reports a **negative** result too. "This corpus also cannot
identify the weights, and here is which subscore was constant" is a real
contribution, because it narrows where the calibration is possible.

## What not to do

- Do not cite the current weights as an optimum, in the paper or in the README.
- Do not tune the weights against one corpus and call the result calibrated. A
  weight vector fitted to SmellNet would place essentially all mass on
  `dynamicRange`, because that is the only subscore that varies there.
- Do not remove the `None` distinction for unmeasurable subscores to make a total
  appear. An honest partial score beats a complete wrong one.

## Related

- `opensmell/mox/quality.py` — implementation, with the provisional-weight warning
  in the module docstring.
- `tools/stress_quality_corpus.py` — protocol-shaped synthetic corpus and the
  sensitivity / specificity / reliability measurement.
- `tools/derive_quality_weights.py` — split-half reliability times discriminative
  range on real recordings.