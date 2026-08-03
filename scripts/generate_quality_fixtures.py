"""Generate the shared quality-scoring fixture matrix.

Single source of truth for cross-language consistency: the SAME `.osmell`
bundles (base64-embedded) and expected 7-factor quality reports are written to
both the Python test suite (`opensmell/tests/fixtures/quality_cases.json`) and
the TypeScript test suite (`osmograph-web/lib/osmell/__tests__/fixtures/`).

Rules for contributors: do NOT hand-edit the output JSON. Change the cases
below (or the quality implementation) and re-run this script, then re-run BOTH
test suites. The golden values encode OSMELL_FORMAT_SPEC.md §7.

Run:  python scripts/generate_quality_fixtures.py
"""

import base64
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from opensmell import build_osmell, compute_quality  # noqa: E402
from opensmell.types import (  # noqa: E402
    BaselineDescriptor,
    ChannelDescriptor,
    OsmellFile,
    OsmellManifest,
    SensorDescriptor,
    SessionDescriptor,
)

CHANNELS = ["VOC", "Alcohol", "LPG", "CO", "NO2", "C2H5OH"]

# Relative amplitudes live at scale ~10^2 of adcMax=4095.
AMP = 600.0


def _envelope(i: int, n: int, recover: bool = True) -> float:
    clean = 0.15 * n
    rise_end = 0.45 * n
    plateau_end = 0.55 * n
    decay_end = 0.90 * n
    if i < clean:
        return 0.0
    if i < rise_end:
        return (i - clean) / (rise_end - clean)
    if i < plateau_end:
        return 1.0
    if not recover:
        return 1.0
    if i < decay_end:
        return 1.0 - (i - plateau_end) / (decay_end - plateau_end)
    return 0.0


def _build_case(
    name,
    n=600,
    role="exposure",
    baseline_source="auto",
    amp=AMP,
    recover=True,
    adc_max=None,
    sampling_rate_hz=None,
    irregular=False,
    dead_channels=None,
    channels=None,
):
    channels = channels or CHANNELS
    dead = dead_channels or []
    time = []
    for i in range(n):
        base_ts = i * 100
        time.append(base_ts + (100 if irregular and i % 8 == 0 else 0))
    data = {}
    for idx, c in enumerate(channels):
        base = 100.0 + idx * 30
        if c in dead:
            data[c] = [100.0] * n
        else:
            data[c] = [base + amp * _envelope(i, n, recover) for i in range(n)]
    manifest = OsmellManifest(
        osmell={"formatVersion": "1.0.0"},
        sensor=SensorDescriptor(
            sensor_type="mox",
            channels=[ChannelDescriptor(id=c, unit="adc") for c in channels],
            time_column="timestamp_ms",
            sampling_rate_hz=sampling_rate_hz,
            adc_max=adc_max,
        ),
        session=SessionDescriptor(
            role=role,
            label=name.replace("_", "-"),
            group_id="fixture",
            recorded_at="2026-01-15T10:00:00",
        ),
        baseline=BaselineDescriptor(source=baseline_source, r0_samples=15),
    )
    return OsmellFile(manifest=manifest, time=time, data=data)


def _make_cases():
    cases = [
        _build_case("ideal_exposure_auto", sampling_rate_hz=10.0, adc_max=4095),
        _build_case("ideal_exposure_auto_no_adcmax", sampling_rate_hz=10.0),
        _build_case("ideal_exposure_auto_no_rate", adc_max=4095),
        _build_case("ideal_exposure_auto_undeclared"),
        _build_case("no_baseline", baseline_source="none", sampling_rate_hz=10.0, adc_max=4095),
        _build_case("single_role", role="single", sampling_rate_hz=10.0, adc_max=4095),
        _build_case("short_exposure", n=150, sampling_rate_hz=10.0, adc_max=4095),
        _build_case("clipped_signal", amp=4100.0, sampling_rate_hz=10.0, adc_max=4095),
        _build_case("dead_channel", dead_channels=["NO2"], sampling_rate_hz=10.0, adc_max=4095),
        _build_case("no_recovery", recover=False, sampling_rate_hz=10.0, adc_max=4095),
        _build_case("low_span", amp=40.0, sampling_rate_hz=10.0, adc_max=4095),
        _build_case("irregular_gaps", irregular=True, sampling_rate_hz=10.0, adc_max=4095),
    ]
    return cases


def _expected(file):
    q = compute_quality(
        file,
        sample_count=len(file.time),
        guess_sampling_rate_hz=10.0,
    )
    subs = {k: (round(v.value, 6) if v.value is not None else None) for k, v in q.subscores.items()}
    return {
        "subscores": subs,
        "total": q.total,
        "badge": q.badge,
        "deadSensors": list(q.flags.dead_sensors),
        "usedMedianSamplingRate": q.flags.used_median_sampling_rate,
        "usedDefaultAdcMax": q.flags.used_default_adc_max,
    }


def main():
    from datetime import datetime, timezone

    document = {
        "spec": "OSMELL_FORMAT_SPEC.md §7",
        "generator": "opensmell/scripts/generate_quality_fixtures.py",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "tolerance": 0.01,
        "cases": [
            {
                "name": case.manifest.session.label.replace("-", "_"),
                "osmellB64": base64.b64encode(build_osmell(case)).decode("ascii"),
                "expected": _expected(case),
            }
            for case in _make_cases()
        ],
    }

    python_path = os.path.join(ROOT, "tests", "fixtures", "quality_cases.json")
    web_path = os.path.join(
        os.path.dirname(ROOT),
        "osmograph-web",
        "lib",
        "osmell",
        "__tests__",
        "fixtures",
        "quality_cases.json",
    )

    for path in (python_path, web_path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(document, fh, indent=2)
            fh.write("\n")
        print(f"wrote {path}")

    for c in document["cases"]:
        print(f"  {c['name']:32s} total={c['expected']['total']:>3} {c['expected']['badge']:9s}")


if __name__ == "__main__":
    main()
