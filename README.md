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

```python
import opensmell

# Canonical 6-channel rig -> 187 features
features, names = opensmell.extract_features("cinnamon.csv")
# features: (N_windows, 187);  names: list of 187 names
avg = features.mean(axis=0)

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
used median sampling rate, no baseline, non-finite samples, dead sensors).

```python
q = opensmell.compute_quality(file, sample_count=len(file.time),
                              guess_sampling_rate_hz=10.0)
print(q.badge, q.total)
```

## Reference-point calibration

The SDK fits the MOX power law `R/R0 = a·C^b`, offers a datasheet quick path and a measured
precise path, and falsifies the fit by leave-one-concentration-out cross-validation.

```python
import opensmell

quick = opensmell.calibrate_quick("MQ135", "c2h5oh", reference_ppm=100)

rr, c = [1.5, 2.1, 3.0], [10, 50, 100]
precise = opensmell.calibrate_precise("MQ135", "co", rr, c)
print(precise["calibration"])          # {"a": ..., "b": ...}
print(precise["loocv"]["mean_abs_pct_error"])  # honest falsification
```

A calibration is a power-law point estimate, not an absolute truth: verified cross-device
affine calibration degrades (47% → 33%), so the SDK never presents calibrated ppm as a
physical absolute.

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

## CSV convenience functions

The thin CSV short-hands wrap the same extractor as the `.osmell` path (they are not a
separate implementation), for quick one-liners:

- `load_recording(path)` → Rs/R₀-normalized array
- `extract_features(path)` → `(N_windows, 28c+…)` array and names
- `process(path, model=None)` → `SmellResult` (extract, optionally predict)
- `train(X, y)` → StandardScaler + RandomForest pipeline (attaches the dimensional floor)
- `predict(path, model)` → `process` with a model
- `feature_names(n_channels=None)` → ordered names for any channel count

## Data model types

`OsmellFile`, `OsmellManifest`, `SensorDescriptor`, `ChannelDescriptor`,
`CalibrationDescriptor`, `SessionDescriptor`, `SessionEvent`, `ParsedSample`,
`ChannelStats`, `QualityReport` — serializing to camelCase JSON via `to_dict()` /
`from_dict()`.

## Dependencies

- Python 3.10+
- numpy, pandas, scikit-learn, scipy

## Related

- [opensmell-rs](https://github.com/opensmell/opensmell-rs) — mirror Rust SDK; same taxonomy
  kept equal by tests (`framework_feature_len(c)`)
- [interoperability](https://github.com/opensmell/interoperability) — cross-device bounds and
  calibration experiments
- [Chemoprint](https://github.com/opensmell/chemoprint) — the molecule-half representation
  from SMILES

Browse the full reference at [opensmell.org/docs/python](https://opensmell.org/docs/python).