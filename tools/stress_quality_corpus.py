"""Derive quality weights from a stress corpus with known ground truth.

Why this exists
---------------
Two earlier attempts to calibrate ``opensmell.mox.quality.WEIGHTS`` by
measurement both failed, and the failures were informative.

Real SmellNet-Base sessions could not calibrate anything: four of the seven
subscores were never computable (``signalStrength`` and
``recoveryCompleteness`` on every file; ``baselineStability`` always reported
``no_baseline``) and three more were constant across all 300 files
(``continuity``, ``saturationFree``, ``durationAdequacy``). SmellNet-Base is
300 clean, uniform, perfectly regular 1 Hz ten-minute recordings with no
baseline or recovery phase -- exactly the corpus in which a quality score has
nothing to detect.

The second attempt injected defects into raw SmellNet and the score barely
reacted. That exposed a real design property rather than a scoring bug: **the
scorer is metadata-dependent.** It declines to evaluate ``baselineStability``
at all when the manifest declares no baseline, and it can only detect
saturation against a correctly declared ``adc_max``. It also trusts the time
axis it is given, so dropped rows do not register as gaps unless the time
vector is genuinely non-uniform. That is defensible -- the manifest is the
data contract -- but it means a stress corpus must vary the *declared metadata*
along with the samples.

Method
------
Synthesise protocol-shaped recordings whose defects are known exactly: a
baseline phase, a first-order exposure, a first-order recovery, six channels
with independent time constants. Score each recording clean and with one
injected defect, then measure three properties per subscore.

sensitivity
    Total fall in score across all injected defects. A subscore that does not
    react to any defect carries no information and must not carry weight.
specificity
    Between-recording spread on clean recordings, rescaled by the theoretical
    maximum. A subscore constant on good data cannot discriminate.
reliability
    Split-half Spearman correlation on clean recordings: does it report a
    property of the recording rather than of which half was shown?

Proposed weight is proportional to ``sensitivity * specificity * reliability``,
normalised to sum to one.

Usage::

    python -m tools.stress_quality_corpus [--n 40] [--seed 7]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opensmell.mox.quality import WEIGHTS, compute_quality_mox  # noqa: E402
from opensmell.types import (  # noqa: E402
    BaselineDescriptor,
    ChannelDescriptor,
    OsmellFile,
    OsmellManifest,
    SensorDescriptor,
    SessionDescriptor,
)

SUBSCORES = (
    "continuity",
    "dynamicRange",
    "saturationFree",
    "baselineStability",
    "signalStrength",
    "recoveryCompleteness",
    "durationAdequacy",
)

ADC_MAX = 4095
N_CHANNELS = 6
BASELINE_S = 30.0
EXPOSURE_S = 45.0
RECOVERY_S = 120.0


# ------------------------------------------------------------------ synthesis

def synth_recording(rng, sr=1.0):
    """Protocol-shaped recording: baseline, first-order rise, first-order decay.

    Each channel gets its own amplitude and time constant so the recording has
    realistic inter-channel structure rather than six copies of one signal.
    """
    n_base = int(round(BASELINE_S * sr))
    n_exp = int(round(EXPOSURE_S * sr))
    n_rec = int(round(RECOVERY_S * sr))

    base_level = rng.uniform(120.0, 400.0, N_CHANNELS)
    amp = rng.uniform(60.0, 600.0, N_CHANNELS)
    tau_a = rng.uniform(4.0, 12.0, N_CHANNELS)
    tau_d = rng.uniform(15.0, 45.0, N_CHANNELS)
    noise = rng.uniform(0.5, 4.0)

    # Time is carried in milliseconds, matching the `timestamp_ms` contract
    # that quality.py assumes for the time column.
    ms = 1000.0 / sr
    t_base = np.arange(n_base) * ms
    t_exp = np.arange(n_exp) * ms + BASELINE_S * 1000.0
    t_rec = np.arange(n_rec) * ms + (BASELINE_S + EXPOSURE_S) * 1000.0

    cols, times = [], []
    for c in range(N_CHANNELS):
        b = np.full(n_base, base_level[c])
        e = base_level[c] + amp[c] * (1.0 - np.exp(-t_exp / tau_a[c]))
        d = base_level[c] + amp[c] * np.exp(-(t_rec - 0.0) / tau_d[c])
        cols.append(np.concatenate([b, e, d]))
    signal = np.column_stack(cols)
    signal = signal + rng.normal(0.0, noise, signal.shape)

    times = np.concatenate([t_base, t_exp, t_rec])
    return times, np.clip(signal, 1.0, ADC_MAX - 1.0)


# ----------------------------------------------------------------- injections

def inj_none(t, x, rng):
    return t, x


def inj_gaps(t, x, rng):
    """Drop spans of samples but keep the original timestamps."""
    keep = np.ones(len(t), dtype=bool)
    for _ in range(max(1, len(t) // 10)):
        s = int(rng.integers(0, max(1, len(t) - 20)))
        keep[s : s + int(rng.integers(5, 20))] = False
    return t[keep], x[keep]


def inj_dead_channel(t, x, rng):
    c = int(rng.integers(0, x.shape[1]))
    x = x.copy()
    x[:, c] = float(np.median(x[:, c]))
    return t, x


def inj_saturation(t, x, rng):
    """Drive the top of one channel onto the declared full-scale rail."""
    c = int(rng.integers(0, x.shape[1]))
    x = x.copy()
    col = x[:, c]
    hi = float(np.percentile(col, 97.0))
    x[col > hi, c] = ADC_MAX
    return t, x


def inj_short(t, x, rng):
    k = max(12, int(len(t) * 0.03))
    return t[:k], x[:k]


def inj_baseline_contamination(t, x, rng):
    """Overwrite the declared baseline window with exposure-level values."""
    x = x.copy()
    period = float(t[1] - t[0]) if len(t) > 1 else 1000.0
    k = max(4, min(x.shape[0] - 1, int(round(BASELINE_S * 1000.0 / period))))
    c = int(rng.integers(0, x.shape[1]))
    # A genuine step change in the baseline window, not a level that matches
    # the clean baseline, otherwise there is nothing for the scorer to detect.
    x[:k, c] = x[k : 2 * k, c].mean() + 0.25 * ADC_MAX
    return t, x


def inj_noise_burst(t, x, rng):
    x = x.copy()
    s = int(len(t) * 0.45)
    w = max(5, len(t) // 25)
    burst = rng.normal(0.0, 40.0, (w, x.shape[1]))
    x[s : s + w] += burst
    return t, x


INJECTIONS = {
    "none": inj_none,
    "gaps": inj_gaps,
    "dead_channel": inj_dead_channel,
    "saturation": inj_saturation,
    "short": inj_short,
    "baseline_contamination": inj_baseline_contamination,
    "noise_burst": inj_noise_burst,
}

EXPECTED = {
    "gaps": {"continuity"},
    "dead_channel": {"dynamicRange", "saturationFree"},
    "saturation": {"saturationFree"},
    "short": {"durationAdequacy", "baselineStability"},
    "baseline_contamination": {"baselineStability"},
    "noise_burst": {"signalStrength", "baselineStability"},
}


# -------------------------------------------------------------------- scoring

def score(t, x, sr, baseline_source="pre"):
    n, m = x.shape
    channels = [ChannelDescriptor(id=f"ch{i}", unit="raw") for i in range(m)]
    baseline = (
        BaselineDescriptor(source=baseline_source, r0_samples=int(round(BASELINE_S * sr)))
        if baseline_source != "none"
        else BaselineDescriptor(source="none")
    )
    manifest = OsmellManifest(
        osmell={"format": "osmell-stress-probe"},
        sensor=SensorDescriptor(
            sensor_type="mox",
            channels=channels,
            sampling_rate_hz=sr,
            adc_bits=12,
            adc_max=ADC_MAX,
        ),
        session=SessionDescriptor(role="exposure"),
        baseline=baseline,
        extra={},
    )
    f = OsmellFile(
        manifest=manifest,
        time=[float(v) for v in t],
        data={f"ch{i}": x[:, i].tolist() for i in range(m)},
        events=None,
    )
    rep = compute_quality_mox(
        file=f, sample_count=n, guess_sampling_rate_hz=sr, unsorted=False, non_finite=0
    )
    return {k: rep.subscores[k].value for k in SUBSCORES}


def spearman(a, b):
    from scipy import stats

    if len(a) < 3:
        return float("nan")
    ra, rb = stats.rankdata(a), stats.rankdata(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--sr", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    scores = {d: {k: [] for k in SUBSCORES} for d in INJECTIONS}
    pairs = {k: [] for k in SUBSCORES}

    for i in range(args.n):
        rng = np.random.default_rng(args.seed * 100003 + i)
        t, x = synth_recording(rng, args.sr)

        for name, fn in INJECTIONS.items():
            try:
                ti, xi = fn(t.copy(), x.copy(), rng)
                out = score(ti, xi, args.sr)
            except Exception:
                continue
            for k in SUBSCORES:
                if out[k] is not None:
                    scores[name][k].append(out[k])

        half = len(t) // 2
        try:
            a = score(t[:half], x[:half], args.sr)
            b = score(t[half:], x[half:], args.sr)
            for k in SUBSCORES:
                if a[k] is not None and b[k] is not None:
                    pairs[k].append((a[k], b[k]))
        except Exception:
            pass

    print(f"Stress corpus: {args.n} protocol-shaped recordings x {len(INJECTIONS)}"
          f" conditions at {args.sr} Hz, seed {args.seed}\n")
    print(f"{'subscore':<22}{'sensitivity':>12}{'specificity':>13}{'reliability':>13}")
    print("-" * 60)

    weights = {}
    diag = {}
    for k in SUBSCORES:
        cv = np.array(scores["none"][k], dtype=float)
        clean_mean = float(cv.mean()) if len(cv) else 0.0
        iqr = float(np.subtract(*np.percentile(cv, [75, 25]))) if len(cv) >= 2 else 0.0
        spec = min(iqr / 100.0, 1.0)
        rho = spearman([p[0] for p in pairs[k]], [p[1] for p in pairs[k]])
        rel = 0.0 if not np.isfinite(rho) else max(rho, 0.0)
        sens = 0.0
        per_defect = {}
        for name in INJECTIONS:
            if name == "none":
                continue
            dv = np.array(scores[name][k], dtype=float)
            if len(dv) < 3:
                continue
            drop = max(0.0, clean_mean - float(dv.mean()))
            per_defect[name] = drop
            sens += drop
        diag[k] = per_defect
        weights[k] = sens * spec * rel
        rt = "n/a" if not np.isfinite(rho) else f"{rho:+.3f}"
        print(f"{k:<22}{sens:>12.1f}{spec:>13.3f}{rt:>13}")

    tot = sum(weights.values())
    print()
    if tot <= 0:
        print("No subscore carries positive measured weight; inspect injections.")
        return 2

    print("Proposed weights (sum to 1.00):")
    print(f"{'subscore':<22}{'current':>9}{'derived':>10}{'delta':>9}")
    print("-" * 50)
    for k in SUBSCORES:
        d = weights[k] / tot
        print(f"{k:<22}{WEIGHTS.get(k, 0.0):>9.2f}{d:>10.3f}{d - WEIGHTS.get(k, 0.0):>+9.3f}")

    print("\nDetection coverage (subscore must react to its own defect)")
    for defect, subs in EXPECTED.items():
        hit = [k for k in subs if diag[k].get(defect, 0.0) > 1.0]
        miss = [k for k in subs if diag[k].get(defect, 0.0) <= 1.0]
        flag = "" if not miss else f"   MISSED: {', '.join(miss)}"
        print(f"  {defect:<24}{len(hit)}/{len(subs)}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())