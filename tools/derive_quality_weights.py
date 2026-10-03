"""Derive the MOX quality-score weights from measured data instead of assertion.

The weights in ``opensmell.mox.quality.WEIGHTS`` were chosen by hand. This
script replaces that judgement with a measurement, using two properties a
quality instrument must actually have.

Split-half reliability
    A recording is scored on its first half and its second half. The Spearman
    correlation between the two halves' subscores measures whether a subscore
    reports something about the *recording* rather than about *which half* was
    shown. A subscore that cannot reproduce itself under a split is measuring
    noise and must not carry much weight.

Discriminative range
    A subscore that returns the same value for every recording carries no
    information, however reproducible it is. Between-session spread is
    measured as the interquartile range across recordings, rescaled to
    [0, 1] by the theoretical maximum of each subscore.

Measurability
    The share of recordings on which a subscore could be computed at all. A
    subscore that is ``None`` for most files cannot discriminate between them.

The proposed weight is proportional to ``reliability * range * measurability``,
normalised to sum to one. Scores are not weighted by how important the
underlying phenomenon sounds; they are weighted by how reliably and
informatively the subscore measures it.

Usage::

    python -m tools.derive_quality_weights [--limit N] [--sr 1.0]

SmellNet-Base sessions are used because they are the largest corpus of real
MOX recordings available with a known 1 Hz cadence.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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

# Theoretical maxima, needed to put the different subscores on one scale so
# that a spread in one cannot be compared directly against a spread in another.
MAXIMA = {
    "continuity": 100.0,
    "dynamicRange": 100.0,
    "saturationFree": 100.0,
    "baselineStability": 100.0,
    "signalStrength": 100.0,
    "recoveryCompleteness": 100.0,
    "durationAdequacy": 100.0,
}

DEFAULT_SNELLNET = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--DeweiFeng--smell-net/snapshots"
)


def find_sessions(root: str) -> list[str]:
    pattern = os.path.join(root, "*", "base_data", "*", "*", "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        files = sorted(glob.glob(os.path.join(root, "*", "*", "*", "*.csv")))
    return files


def load_half(path: str, half: str, sr: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    if half == "first":
        df = df.iloc[: len(df) // 2]
    elif half == "second":
        df = df.iloc[len(df) // 2 :]
    return df


def score_frame(df: pd.DataFrame, sr: float) -> dict[str, float | None]:
    channels = [ChannelDescriptor(id=str(c), unit="raw") for c in df.columns]
    n = len(df)
    manifest = OsmellManifest(
        osmell={"format": "osmell-quality-probe"},
        sensor=SensorDescriptor(
            sensor_type="mox",
            channels=channels,
            sampling_rate_hz=sr,
            adc_bits=12,
            adc_max=None,
        ),
        session=SessionDescriptor(role="single"),
        baseline=BaselineDescriptor(source="none"),
        extra={},
    )
    # The scorer contract is milliseconds (`timestamp_ms`). Emitting seconds
    # here would trip the new unit-mismatch guard and report continuity as
    # withheld rather than as the intended score.
    osmell = OsmellFile(
        manifest=manifest,
        time=[1000.0 * i / sr for i in range(n)],
        data={str(c): df[c].astype(float).tolist() for c in df.columns},
        events=None,
    )
    report = compute_quality_mox(
        file=osmell,
        sample_count=n,
        guess_sampling_rate_hz=sr,
        unsorted=False,
        non_finite=0,
    )
    return {k: report.subscores[k].value for k in SUBSCORES}


def spearman(a: list[float], b: list[float]) -> float:
    from scipy import stats

    if len(a) < 3:
        return float("nan")
    ra = stats.rankdata(a)
    rb = stats.rankdata(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_SNELLNET)
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--sr", type=float, default=1.0)
    args = ap.parse_args()

    files = find_sessions(args.root)
    if not files:
        print(f"No sessions found under {args.root}", file=sys.stderr)
        return 1
    files = files[: args.limit]
    print(f"Scoring {len(files)} SmellNet-Base sessions at {args.sr} Hz\n")

    first: dict[str, list[float]] = {k: [] for k in SUBSCORES}
    second: dict[str, list[float]] = {k: [] for k in SUBSCORES}
    computable = {k: 0 for k in SUBSCORES}
    pairs = {k: 0 for k in SUBSCORES}

    for path in files:
        try:
            sf = score_frame(load_half(path, "first", args.sr), args.sr)
            ss = score_frame(load_half(path, "second", args.sr), args.sr)
        except Exception:
            continue
        for k in SUBSCORES:
            if sf[k] is not None and ss[k] is not None:
                pairs[k] += 1
                computable[k] += 1
                first[k].append(sf[k])
                second[k].append(ss[k])
            elif sf[k] is not None or ss[k] is not None:
                computable[k] += 1

    total = len(files)
    rows = []
    for k in SUBSCORES:
        rho = spearman(first[k], second[k])
        allv = np.array(first[k] + second[k], dtype=float)
        if len(allv) >= 2:
            iqr = float(np.subtract(*np.percentile(allv, [75, 25])))
        else:
            iqr = 0.0
        rng = min(iqr / MAXIMA[k], 1.0) if MAXIMA[k] else 0.0
        meas = pairs[k] / total if total else 0.0
        rows.append((k, rho, rng, meas, len(first[k])))

    print(f"{'subscore':<22}{'current':>9}{'split-half':>12}{'range':>9}{'measurable':>12}")
    print("-" * 64)
    proposed = {}
    for k, rho, rng, meas, n in rows:
        r = 0.0 if not np.isfinite(rho) else max(rho, 0.0)
        proposed[k] = r * rng * meas
        rtxt = "n/a" if not np.isfinite(rho) else f"{rho:+.3f}"
        print(f"{k:<22}{WEIGHTS.get(k, 0.0):>9.2f}{rtxt:>12}{rng:>9.3f}{meas:>12.2f}")

    s = sum(proposed.values())
    print()
    if s <= 0:
        print("All candidates scored zero; no data-driven weights can be derived.")
        print("This happens when every subscore is constant across sessions, which")
        print("means the corpus does not exercise the failure modes the score")
        print("detects. A corpus with real baseline/recovery phases is required.")
        return 2

    print("Proposed weights (sum to 1.00):")
    print(f"{'subscore':<22}{'current':>9}{'derived':>10}{'delta':>9}")
    print("-" * 50)
    for k in SUBSCORES:
        d = proposed[k] / s
        print(f"{k:<22}{WEIGHTS.get(k, 0.0):>9.2f}{d:>10.3f}{d - WEIGHTS.get(k, 0.0):>+9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())