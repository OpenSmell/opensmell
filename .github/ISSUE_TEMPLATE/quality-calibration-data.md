name: Quality calibration data
about: Contribute recordings or measurements that would calibrate the MOX quality scorer
title: "[quality-calibration] "
labels: quality-calibration
assignees: ''

---

## Why this matters

The seven quality factors in `opensmell.mox.quality` currently use **provisional
weights**. They encode a judgement about which failure modes matter most, not a
measured optimum. We have not been able to calibrate them, because no corpus we have
access to exercises the failure modes the scorer is supposed to detect.

This is documented in [`docs/quality-weight-calibration.md`](../../docs/quality-weight-calibration.md),
including the three calibration attempts that failed and the defect analysis, including
the one design question that is still open.

We are not asking anyone to fit the weights for us. We are asking for **data**.

## What would help

In rough order of value:

### 1. Labeled defective recordings — good first issue

Any metal-oxide recording where you **know** what was wrong. The metadata about the
defect is worth more than the samples.

- a disconnected or dead sensing element
- a channel driven onto its ADC rail (saturated)
- a documented clock or dropped-packet problem
- an acquisition that stopped early
- a recording with no usable baseline

If you have such a recording, please say which defect it contains and how you know.
`adcMax`, the nominal sampling rate, and the time-column unit are the three details
that matter most.

### 2. Real cross-device repetitions

The same physical exposure recorded on **two or more devices**, ideally with a
declared pre-exposure baseline and a post-exposure purge.

This corpus does not currently exist anywhere we can find, and it is what would make
the subscores comparable across hardware. SmellNet and the UCI datasets are both
single-device, and neither has usable baseline phases.

### 3. Protocol-shaped real recordings

Real sensors following a documented protocol: declared baseline, then exposure, then
purge. These make the signal-strength and recovery-completeness factors computable,
which unstructured recordings cannot do at all.

### 4. Rust / Python parity fixtures

`opensmell-rs` reimplements this scorer. We have no harness asserting the two produce
identical reports. A set of inputs plus expected outputs from both implementations
would be a genuine contribution.

## Self-contained good first issues

Four defects are fixed and have regression tests. All four shared one shape: a defect
in the recording *raised* the quality score — noise, saturation, a dead channel, and a
mislabelled time column each improved the result for data that had got worse. What
remains is mostly data work, plus one design question:

- [ ] **Add a full Rust/Python parity harness for `compute_quality`.** There are
  now numeric parity spot-checks in `opensmell-rs` covering the four fixes, but
  no harness that sweeps a corpus and compares both implementations field by
  field. This is the most valuable remaining engineering task and needs no
  hardware.
- [ ] **Port the stress-corpus harness to Rust** so both implementations can be
  scored over the same generated corpus in one run.
- [ ] **Propose reference severities for the scale fix (defect 3).** Dynamic range
  and continuity move by 80+ points for a bad recording while saturation-free, a
  per-channel mean, moves by at most 16.7 for a fully dead channel. Rescaling them
  onto one scale needs a judgement about which defects matter and how much.
  Propose reference severities with reasoning; discussion is more useful than code
  here.
- [ ] **Document `time_unit_mismatch` in the format spec.** The new flag and
  withheld-continuity behaviour should be reflected in `OSMELL_FORMAT_SPEC.md`.

## Reporting a negative result

Useful, and genuinely welcome. "I tried calibrating against dataset X and it also
cannot identify the weights, because subscore Y was constant" is a real contribution.
It narrows where calibration is possible. You do not need a successful fit to help.

## What not to do

- Do not tune the weights against one corpus and report them as calibrated. Fitting
  to SmellNet would place nearly all mass on `dynamicRange`, because it is the only
  factor that varies there.
- Do not file an issue asking what the "correct" weights are. We do not know yet, and
  that is the point of this document.

## Checklist

- [ ] I have read [`docs/quality-weight-calibration.md`](../../docs/quality-weight-calibration.md)
- [ ] My contribution is data, a negative result, or one of the listed defects
- [ ] If recordings: I have stated `adcMax`, the nominal rate, and the time-column unit