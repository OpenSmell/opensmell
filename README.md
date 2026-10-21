# OpenSmell SDK

One Python SDK for digital olfaction: a portable recording container (`.osmell`), ingest,
quality scoring, an auditable MOX feature framework, reference-point calibration, a
hardware-sufficiency gate, and a thermodynamic feasibility check ("will my e-nose actually
smell this?").

Everything is reachable from the package root after `import opensmell` — there is no
separate "legacy" API to learn.

```bash
pip install opensmell
```

## The core idea: a recording is `.osmell`

A smell recording is a self-describing ZIP: `manifest.json` + `data.csv` (+ optional
`events.json`). Raw values and the baseline are preserved so any client can pick its own
normalization.

```python
import opensmell

# Load a recording
file = opensmell.parse_osmell_file("cinnamon.osmell")
csv_text = opensmell.csv_from_file(file)        # back to CSV if you need it
opensmell.write_osmell(file, "renamed.osmell")  # format version 1.0.0
```

## Ingest raw CSVs

Ingestion never raises on structural weirdness — every interpretation decision is surfaced
as a warning and errors are captured on the result.

```python
# One file
s = opensmell.ingest_file("cinnamon.csv", substance="cinnamon", role="exposure")
# s.ok, s.file (OsmellFile), s.report (QualityReport), s.warnings

# A folder of recordings, grouped by subfolder = substance
col = opensmell.ingest_folder("recordings/")
for session in col.iter_sessions():
    print(session.substance, session.ok, session.report.badge)
```

## Feature extraction

The MOX feature framework is defined **sensor-count-agnostically**. For any channel count
`c` the vector has `28·c + c(c−1)/2 + 4` features — 28 per channel, one selectivity ratio
per channel pair, and 4 global metrics. At the canonical six-channel rig that is 187.

Channel-count agnostic is not the same as modality agnostic. The implemented feature and
quality backends are MOX/chemiresistive. Use `inspect_source()` to see what the loader
detected before choosing a path:

```python
import opensmell

profile = opensmell.inspect_source("mox_6ch.csv")
profile.sensor_type          # "mox" for MQ/SGP/TGS/BME model headers, else "unknown"
profile.feature_family       # "mox", "mox-compatible", or "unsupported"
profile.models               # recognized channel -> canonical model, e.g. {"MQ135": "MQ-135"}
profile.warnings             # interpretation decisions, including inferred sensor type
```

Unknown numeric arrays are treated as `"mox-compatible"` and can use the signal-level
framework. Declared MIRIS or electrochemical recordings are reported as `"unsupported"`;
`extract_features()` and `process()` raise `UnsupportedSensorTypeError` instead of silently
applying MOX math. Their backends are still explicit stubs.

```python
# Canonical 6-channel rig -> 187 features
features, names = opensmell.extract_features("cinnamon.csv")
# features: (N_windows, 187);  names: list of 187 names
avg = features.mean(axis=0)

# Any CSV channel names work: S1..S6 are detected automatically
features, names = opensmell.extract_features("mox_6ch.csv")

# Select or rename channels explicitly when the header is ambiguous
features, names = opensmell.extract_features(
    "mox_6ch.csv",
    channels={"S1": "NO2", "S2": "C2H5OH", "S3": "VOC",
              "S4": "CO", "S5": "Alcohol", "S6": "LPG"},
)

# The same call accepts the ingest result or a parsed .osmell file
session = opensmell.ingest_file("mox_6ch.csv")
features, names = opensmell.extract_features(session)
features, names = opensmell.extract_features("recording.osmell")

# No hardwired six: names follow the same formula for any channel count
from opensmell.mox import features as _f
len(_f.feature_names(n_channels=3))   # 91
len(_f.feature_names(n_channels=4))   # 122
```

Every feature has a name, a category, a transfer class, and a failure mode — the taxonomy
is mirrored 1:1 with the web stack and the Rust SDK (kept equal by tests).

## Classify your own data

```python
import opensmell, numpy as np

# Build a labelled feature matrix
X, y = [], []
for substance in ["cinnamon", "garlic", "coffee"]:
    feats, _ = opensmell.extract_features(f"{substance}.csv")
    X.append(feats.mean(axis=0))
    y.append(substance)

model = opensmell.train(np.array(X), y)
result = opensmell.predict("unknown.csv", model)
print(f"Predicted: {result.substance} (confidence: {result.confidence:.2f})")
```

`process(filepath, model=None)` does feature extraction and (optionally) prediction in one
call and returns a `SmellResult`. Without a model it never fabricates a substance or
confidence.

## Quality scoring

Seven weighted factors (baseline stability, signal strength, continuity, recovery,
dynamic range, saturation-free, duration) produce a score and a badge —
Excellent / Good / Fair / Poor / Unknown — with every assumption flagged (used default ADC,
used median sampling rate, no baseline, non-finite samples, dead sensors, time-unit
mismatch).

```python
q = opensmell.compute_quality(file, sample_count=len(file.time),
                              guess_sampling_rate_hz=10.0)
print(q.badge, q.total)
```

**The factor weights are provisional and are not calibrated against data.** They
encode a judgement about which failure modes matter, not a measured optimum, and
should not be cited as one. Three calibration attempts have failed for reasons that
are documented, along with a defect analysis of the scorer itself and a description
of the corpus that would be needed to close the remaining one.

That analysis is in [`docs/quality-weight-calibration.md`](docs/quality-weight-calibration.md).
The harness is in `tools/derive_quality_weights.py` and `tools/stress_quality_corpus.py`.
Contributions of labeled defective recordings and real cross-device repetitions are
the most useful thing the community can supply here; see the call for contributions
in that document.

Four defects found during the calibration study have been fixed. Three shared one
shape — a defect in the recording *raised* the quality score. The fourth was the
opposite failure: it did not reward bad data, it erased the evidence that the data
was good.

- a seconds-valued time column is flagged instead of being scored as packet loss;
- `dynamicRange` no longer rewards interference;
- dead channels lower the total instead of raising it;
- saturation no longer rewards itself. A sample pinned at the converter rail carries no
  amplitude information, but it was being counted as both span and peak, so a saturated
  channel scored a wider dynamic range and a stronger signal than the same channel read
  in range. `saturationFree` still reports the clipping.

The first, `dynamicRange`, dead channels, and saturation each let worse data score
better. The time-column defect was quieter and just as damaging: a time column
labelled as seconds while holding milliseconds produced a continuity score of 0,
identical to the score for a recording that had genuinely lost its packets. A
caller reading 0 could not tell a mislabelled column from a broken one, so the
defect destroyed the subscore's ability to say anything at all. Continuity is now
withheld with reason `time_unit_mismatch` rather than silently reporting 0.

A quality score that improves for worse data, or that reports the same number for
two opposite faults, is not measuring quality. That is the reason to distrust a
score not yet shown monotone and non-degenerate in every input. Regression tests
cover each defect in both this implementation and `opensmell-rs`.

Note that the saturation fix needs a manifest declaring `adcMax`; without one there is
no rail to identify and the check stays silent.

Known limitation that remains: the seven subscores are not on a common scale.
Continuity and duration are per-recording and move by 80+ points when a recording is
bad, while saturation-free is a per-channel mean, so one saturated channel out of six
moves it by at most 16.7. Compare subscores within a recording rather than across
devices until this is fixed.

## Reference-point calibration

The SDK fits the MOX power law `R/R0 = a·C^b`, offers a datasheet quick path and a measured
precise path, and falsifies the fit by leave-one-concentration-out cross-validation.

```python
import opensmell

print(opensmell.list_sensors())                 # valid model ids
print(opensmell.sensor_gases("MQ-135"))         # valid gases: CO, Alcohol, ...

# Sensor ids are case/punctuation-insensitive; use a gas the model tabulates.
quick = opensmell.calibrate_quick("MQ-135", "Alcohol", reference_ppm=100)

rr, c = [1.5, 2.1, 3.0], [10, 50, 100]
precise = opensmell.calibrate_precise("MQ-135", "CO", rr, c)
print(precise["calibration"])          # {"a": ..., "b": ...}
print(precise["loocv"]["mean_abs_pct_error"])  # honest falsification
```

A calibration is a power-law point estimate, not an absolute truth: verified cross-device
affine calibration degrades (47% → 33%), so the SDK never presents calibrated ppm as a
physical absolute. Unknown sensor or gas names raise an error that lists the valid
options (and offers close matches), so a typo does not turn into a silent default.

## Hardware-sufficiency gate

A model trained on N channels must not silently run on fewer. The gate checks the rig's
effective dimensionality against the model's requirement and warns rather than padding a
dead channel with a mean.

```python
import opensmell

opensmell.check_rig_sufficiency(n_channels=4, model)  # warns if insufficient
opensmell.implied_channels(187)   # 6 — inverts 28c + c(c−1)/2 + 4
```

## "Will my e-nose actually smell it?" — the feasibility chain

A thermodynamic estimate (not a measurement) answering whether a substance is even a
feasible target for a MOX array, graded green / yellow / red.

```python
from opensmell import smellability

verdict = smellability.resolve_and_run("ethanol", "chemical")
print(verdict.verdict, verdict.confidence, verdict.signal_strength)

# Or estimate a brand-new molecule from its SMILES, fully offline
chem = smellability.chemical_from_smiles("C1=CC=CC(=C1)C=O", name="benzaldehyde")
```

The chain computes identity → volatility → signal → reactivity, with exposure/dilution
guidance and a cross-check against how many substances your sensor count can distinguish.
It is honest about its limits: a feasibility estimate is not a calibrated concentration, a
guarantee of mixture decomposition, or a promise across unseen devices.

## Feature taxonomy

| Group | Per-channel features | Count/channel |
|-------|----------------------|---------------|
| Device-agnostic | relative_amplitude, direction, rise_time, decay_time, auc, endpoint_delta | 6 |
| Absolute | raw_resistance, baseline_resistance, voltage, calibrated_concentration | 4 |
| Temporal | hf_transient, oscillation_freq, oscillation_amp, response_latency | 4 |
| Health | drift_rate, sensitivity_decay, noise_floor, hysteresis | 4 |
| Hardware | circuit_response, thermal_profile, adc_noise | 3 |
| Advanced | saturation_index + 6 decay terms (tau1–3, a1–3) | 7 |

That is 28 per channel, plus `C(c,2)` selectivity ratios and 4 global metrics — `187` at the
canonical six channels (`91` at three, `406` at twelve). Rs/R₀ normalization cancels Vcc and
RL in the ratio; it does not cancel the sensor constants (a, b), so cross-device transfer
requires per-rig reference-point calibration.

## Configuration presets

Twelve use cases ship as YAML in [`presets/`](presets/): the channel list, the feature
groups to compute, the baseline protocol, the anomaly thresholds and — where a monitor
paged on a schedule — a false-alarm budget. Each file states why its sensors and its
cadence were chosen, and marks every figure the author could not stand behind in
`provisional:`.

```python
import opensmell

opensmell.list_presets()             # 12 ids
preset = opensmell.load_preset("gas_leak_fire_smoke")

preset.sampling_rate_hz              # 10.0  — mandatory, never defaulted
preset.sensors                       # [mq2, mq4, mq5, mq7, mq8, mq9] with roles
preset.r0_samples                    # baseline length at this cadence: 18000
preset.advisories()                  # non-fatal observations, e.g. cadence_below_1hz
preset.to_sensor_descriptor()        # the array as a .osmell SensorDescriptor
```

```console
$ opensmell-presets list
$ opensmell-presets show food_freshness
$ opensmell-presets validate --strict      # advisories become failures
$ opensmell-presets field gas_leak_fire_smoke sensors[3].model
```

The loader is strict on purpose: a missing or non-positive cadence, an unknown feature
group, a target gas the constants table does not hold for that model, a declared Rs/R₀
that disagrees with `sensors.json`, a threshold tuned without a stated basis, or a
`provisional:` entry that resolves to nothing all raise with the file and field named.
Advisories are the opposite case — a real observation that is not a reason to refuse,
such as a per-minute cadence whose 120 s resolution floor makes the kinetic feature
groups meaningless. Presets are searched in `$OPENSMELL_PRESETS_DIR`, the source tree,
the package, then `<prefix>/share/opensmell/presets`.

## MCP server (for AI agents)

`opensmell-mcp` exposes a read-only slice of the SDK over the
[Model Context Protocol](https://modelcontextprotocol.io) so an agent working in a
terminal can inspect presets and recordings without re-deriving the format. It is
strictly local: no network, no telemetry, no writes, no hardware. Six tools are
exposed — `list_presets`, `get_preset`, `validate_recording`, `analyze_recording`,
`compute_health_features`, `capabilities` — and failures come back as actionable
strings, never tracebacks. The four-way anomaly verdict
(Normal / GradualDrift / Poisoning / Dead) lives in the Rust SDK and is deliberately
*not* reimplemented here; `compute_health_features` returns the health features it
consumes and says so.

```bash
pip install "opensmell[mcp]"
```

Register it with Claude Code:

```console
$ claude mcp add opensmell -- opensmell-mcp
```

Or drop this into any MCP client's config:

```json
{
  "mcpServers": {
    "opensmell": {
      "command": "opensmell-mcp",
      "args": []
    }
  }
}
```

Ask an agent `capabilities()` first when unsure what is in scope.

## Recording convenience functions

The thin recording short-hands wrap the same extractor as the `.osmell` path (they are not
a separate implementation), for quick one-liners. Each accepts a CSV/.osmell path, an
`OsmellFile`, or the `IngestedSession` returned by `ingest_file`:

- `inspect_source(path, channels=None)` → `SensorProfile` (family, models, feature path, warnings)
- `load_recording(path, channels=None)` → Rs/R₀-normalized array
- `extract_features(path, channels=None)` → `(N_windows, 28c+…)` array and names (MOX/chemiresistive only)
- `process(path, model=None, channels=None)` → `SmellResult` (extract, optionally predict)
- `train(X, y)` → StandardScaler + RandomForest pipeline (attaches the dimensional floor)
- `predict(path, model, channels=None)` → `process` with a model
- `feature_names(n_channels=None)` → ordered names for any channel count

## Data model types

`OsmellFile`, `OsmellManifest`, `SensorDescriptor`, `ChannelDescriptor`,
`CalibrationDescriptor`, `SessionDescriptor`, `SessionEvent`, `ParsedSample`,
`ChannelStats`, `QualityReport` — serializing to camelCase JSON via `to_dict()` /
`from_dict()`.

## Dependencies

- Python 3.10+
- numpy, pandas, scikit-learn, scipy
- optional: `mcp` — the `opensmell-mcp` server (`pip install "opensmell[mcp]"`)

## Related

- [opensmell-rs](https://github.com/opensmell/opensmell-rs) — mirror Rust SDK; same taxonomy
  kept equal by tests (`framework_feature_len(c)`)
- [interoperability](https://github.com/opensmell/interoperability) — cross-device bounds and
  calibration experiments
- [Chemoprint](https://github.com/opensmell/chemoprint) — the molecule-half representation
  from SMILES

Browse the full reference at [opensmell.xyz/docs/python](https://opensmell.xyz/docs/python).