# OpenSmell — Master Reference

> **Single source of truth** for the OpenSmell project: the Python SDK, the `.osmell`
> exchange format, the 187-feature MOX framework, sensor theory, the Osmograph Web
> platform, and the project audit.
>
> This document consolidates the following sources (originals folded in on
> 2026-08-03):
> - `opensmell/README.md` (SDK quick-start; 145-feature count is stale — see §2.6)
> - `opensmell/spec/ALGORITHMS.md` (multi-exponential decay model)
> - `docs/FEATURES.md` (feature reference)
> - `opensmell/docs/SENSOR_THEORY.md` (sensor theory)
> - `osmograph-web/DOCUMENTATION.md` (web platform reference)
> - `osmograph-web/OSMELL_FORMAT_SPEC.md` (`.osmell` v1.1.0 spec)
> - `AUDIT_REPORT.md` (2026-06-19) and `AUDIT_FINAL.md` (2026-06-16)

---

## Table of Contents

1. [Overview](#1-overview)
2. [The OpenSmell Python SDK](#2-the-opensmell-python-sdk)
3. [The `.osmell` Format Specification (v1.1.0)](#3-the-osmell-format-specification-v110)
4. [Feature Reference: the 187-Dimension Framework](#4-feature-reference-the-187-dimension-framework)
5. [Multi-Exponential Decay Model](#5-multi-exponential-decay-model)
6. [Sensor Theory](#6-sensor-theory)
7. [Osmograph Web Platform](#7-osmograph-web-platform)
8. [Project Audit](#8-project-audit)
9. [Changelog & Reproducibility](#9-changelog--reproducibility)

---

## 1. Overview

OpenSmell is an open stack for electronic-nose (e-nose) data: build your own
device, record odor samples, extract device-agnostic features, and train
classifiers to recognise substances. The stack covers:

- **Hardware** — `electronic-nose/`: BOM, ESP32 firmware, wiring, build guide.
- **SDK** — `opensmell/`: pip-installable Python package; feature extraction,
  classifier training, prediction, `.osmell` read/write.
- **Desktop GUI** — `Osmograph/`: PyQt5 recorder, firmware flashing, classifier
  training, real-time prediction.
- **Web platform** — `osmograph-web/`: library/import/compare/train/smellability.
- **Data** — `SmellNet/`: 50 substances, 300+ recordings, 6 MOX sensors
  (HuggingFace `DeweiFeng/smell-net`, arXiv 2506.00239).
- **Research** — `research/`, `encoder/`, `Chemoprint/`, `interoperability/`:
  calibration experiments, chemoprint descriptor, paradigm proofs, session
  invariance, adversarial validation.

The core SDK feature framework extracts **187 features per recording** from MOX
(MOS) sensor time-series, normalized by `Rs/R0` so circuit-level constants
(`Vcc`, `RL`) cancel mathematically. This document is the authoritative
reference for all of it.

---

## 2. The OpenSmell Python SDK

Extract device-agnostic framework features from electronic-nose recordings.

### 2.1 Quick start

```bash
pip install .
```

```python
import opensmell

# From CSV file — columns: NO2, C2H5OH, VOC, CO, Alcohol, LPG
features, names = opensmell.extract_features("cinnamon.csv")
# features: (N_windows, 187) numpy array
# names: list of 187 feature name strings

# Average features across all windows
avg_features = features.mean(axis=0)
```

### 2.2 Training a custom classifier

```python
import opensmell
import numpy as np

# Build a feature matrix from multiple recordings
X, y = [], []
for substance in ["cinnamon", "garlic", "coffee"]:
    feats, _ = opensmell.extract_features(f"{substance}.csv")
    X.append(feats.mean(axis=0))
    y.append(substance)

X = np.array(X)

# Train
model = opensmell.train(X, y)

# Predict a new recording
result = opensmell.predict("unknown.csv", model)
print(f"Predicted: {result.substance} (confidence: {result.confidence:.2f})")
```

### 2.3 Package layout (v3)

The real v3 layout is a sensor-agnostic core with sensor-specific subpackages.
Top-level modules define the interfaces; `opensmell.mox` implements them for
metal-oxide (SnO₂) sensors; `miris`/`electrochemical` are explicit
`NotImplementedError` stubs pinning the framework contract for future families.

**Top-level `opensmell/` (8 modules):**

| Module | Responsibility |
|--------|----------------|
| `__init__.py` | Authoritative public API surface: v3 sensor-agnostic functions + types, plus the preserved legacy v2 API (see §2.6/§2.7). |
| `features.py` | Sensor-agnostic entry point; re-exports the MOX framework functions from `.mox.features` so pre-v3 callers keep working; adds `run_processor` dispatch. |
| `io.py` | `.osmell` bundle read/write via stdlib `zipfile`; mirrors `osmograph-web/lib/osmell/io.ts` 1:1. |
| `csv.py` | `parse_csv` + `guess_sensor_type`; mirrors `osmograph-web/lib/osmell/csv.ts` 1:1. |
| `types.py` | Sensor-agnostic data model (`OsmellFile`, `OsmellManifest`, descriptors, `QualityReport`) + shared quality/session constants; mirrors `osmograph-web/lib/osmell/types.ts` 1:1. |
| `normalize.py` | Statistical helpers `median` / `mean` / `std` (population, ddof=0); mirrors `osmograph-web/lib/osmell/normalize.ts`. |
| `result.py` | `SmellResult` dataclass + `chemoprint` property. |
| `quality.py` | Top-level `compute_quality` dispatch to the sensor-specific scorer. |

**`opensmell/mox/` (5 modules):**

| Module | Responsibility |
|--------|----------------|
| `__init__.py` | Imports `features`, `normalize`, `quality`, `smellability`. |
| `features.py` | MOX framework features: per-channel dimension functions (`compute_channel_*`), `extract_all_framework_features`, `feature_names`, `compute_multi_exp_decay`, `process_mox`. |
| `normalize.py` | MOX R0 normalization: `r0_from_samples`, `baseline_for_channel`, `channel_stats`, `normalized_series` (web parity). |
| `preprocessing.py` | Legacy CSV path: `load_csv`, `rs_r0_normalize`, `segment`, `expand_channels`; constants `WINDOW_SIZE = 100`, `WINDOW_STRIDE = 10`, `SENSOR_NAMES`. |
| `quality.py` | Spec-compliant 7-factor MOX scorer `compute_quality_mox` (§3.10). |

**`opensmell/mox/smellability/` (13 modules):** the MOX thermodynamic
feasibility chain — a 1:1 Python port of `osmograph-web/lib/smellability/`:

| Module | Responsibility |
|--------|----------------|
| `__init__.py` | Re-export surface matching the web `index.ts`. |
| `chain.py` | 4-step chain (identity → volatility → headspace → MOX redox), guidance, verdict aggregation. |
| `types.py` | Dataclasses (`Chemical`, `Composite`, `FeasibilityVerdict`, …) with camelCase JSON round-trips. |
| `constants.py` | Band tables, `MAX_SUBSTANCES`, `CLASS_TERMS`, ambient constants. |
| `compounds.py` | 48 curated compounds + `COMPOUND_BY_ID` + `REFERENCE_COMPOUND` (ethanol). |
| `composites.py` | 26 composite (mixture) profiles + `COMPOSITE_BY_ID`. |
| `transport.py` | Antoine / Clausius-Clapeyron vapor pressure, Fuller diffusion, flux. |
| `ontology.py` | `PERCEPTS` (14) + `MOX_BOUNDARIES` (4 capabilities / 5 limitations). |
| `groups.py` | Functional-group inference from SMILES (incl. Kekulé→aromatic normalization). |
| `enrichment.py` | Live PubChem lookup + boiling-point extraction + JSON cache. |
| `provisional.py` | Provisional chemicals built from PubChem enrichment. |
| `search.py` | Substance search + exact resolution (web parity scoring). |
| `user_dictionary.py` | User dictionary persisted as JSON on disk. |

**Stub packages (framework contract only):**

| Module | Responsibility |
|--------|----------------|
| `miris/__init__.py` | Raises `NotImplementedError` for `normalize` / `features` / `quality`. |
| `electrochemical/__init__.py` | Raises `NotImplementedError` for `normalize` / `features` / `quality`. |

### 2.4 Sensor-agnostic dispatch (`run_processor`)

`opensmell.features.run_processor(file)` dispatches feature extraction on the
sensor family declared in `file.manifest.sensor.sensor_type` (web
`runProcessor` parity):

| `sensor_type` | Result |
|---------------|--------|
| `mox` | `process_mox(file)` — per-channel kinetic features + normalized series |
| `miris`, `electrochemical` | `{"sensor_type": ..., "normalized": file.data}` (reserved) |
| anything else | `{"sensor_type": "other"}` |

`opensmell.quality.compute_quality` dispatches quality scoring the same way
(`opensmell.mox.quality` → `compute_quality_mox`); `miris` / `electrochemical` /
unknown sensor types raise `NotImplementedError` rather than silently scoring
with MOX math.

### 2.5 `.osmell` I/O (`opensmell.io`)

- `parse_osmell(data: bytes) -> OsmellFile` — opens the ZIP with stdlib
  `zipfile` (no archive dependency); requires `manifest.json` + `data.csv`
  (raises `ValueError` otherwise), rejects empty `data.csv`, cross-validates
  channel IDs in **both** directions (`data.csv` column not declared in the
  manifest, and manifest channel missing from `data.csv`), and reads
  `events.json` when present.
- `parse_osmell_file(path)`, `build_osmell(file) -> bytes` (DEFLATE),
  `write_osmell(file, path)`, `csv_from_file(file)`, `default_file_name(file)`
  (`<label>_<role>_<date>.osmell`), and the backwards-compat alias
  `load_osmell = parse_osmell_file`.
- MIME type constant: `OSMELL_MIME_TYPE = "application/vnd.opensmell.osmell"`.

### 2.6 Public API (v3 sensor-agnostic)

| Function / name | Description |
|-----------------|-------------|
| `parse_csv(text, ...)` | Parse CSV → `CsvParseResult` (header, time column, samples, channel ids, guessed sampling rate, non-finite / unsorted flags) |
| `guess_sensor_type(...)` | Guess the sensor family from CSV columns |
| `parse_osmell(data: bytes)` | Read an `.osmell` bundle from bytes → `OsmellFile` |
| `parse_osmell_file(path)` / `load_osmell` | Read an `.osmell` bundle from disk |
| `build_osmell(file) -> bytes` | Serialize a bundle (ZIP, DEFLATE) |
| `write_osmell(file, path)` | Write a bundle to disk |
| `csv_from_file(file)` | Regenerate the `data.csv` text from an `OsmellFile` |
| `default_file_name(file)` | `<label>_<role>_<date>.osmell` |
| `process_mox(file)` | Per-channel MOX kinetic features + normalized series (web `processMox` parity) |
| `run_processor(file)` | Dispatch feature extraction by `sensor_type` (see §2.4) |
| `compute_quality(file, sample_count, guess_sampling_rate_hz, unsorted=False, non_finite=0)` | 7-factor quality report (MOX) |
| Types | `OsmellManifest`, `OsmellFile`, `SensorDescriptor`, `SessionDescriptor`, `ChannelDescriptor`, `SessionEvent`, `ParsedSample`, `ChannelStats`, `QualityReport` |
| `OSMELL_FORMAT_VERSION` | `"1.0.0"` (the SDK's container version; the spec itself is v1.1.0, §3) |

Shared constants in `opensmell.types` that pin the scoring contract:
`DEFAULT_ADC_MAX = 4095`, `DEFAULT_R0_SAMPLES = 15`, `DEAD_CV_THRESHOLD = 0.001`,
`NOISE_CV_LIMIT = 0.05`, `SNR_TARGET = 10`, `FULL_SCORE_DURATION_S = 60`,
`MIN_SPAN_FRACTION = 0.1`, `GAP_TOLERANCE = 0.1`.

### 2.7 Legacy v2 surface (preserved)

| Function | Description |
|----------|-------------|
| `extract_features(filepath)` | Extract `(N_windows × 187)` features from a CSV |
| `feature_names()` | The 187 MOX feature names in extraction order |
| `process(filepath, model=None)` | Single-call: extract features, optionally predict |
| `train(X, y, n_estimators=200)` | Train a sklearn Pipeline (StandardScaler + RandomForest, balanced, `random_state=42`) |
| `predict(filepath, model)` | Shorthand for `process` with a model |
| `load_recording(filepath)` | Load and `Rs/R0`-normalize a CSV |

### 2.8 Output

`SmellResult` fields:

| Field | Type | Description |
|-------|------|-------------|
| `substance` | str or None | Predicted label (only from `predict`) |
| `confidence` | float or None | Prediction probability |
| `warning` | str or None | `"Low confidence"` (< 0.5) / `"Moderate confidence"` (< 0.7) |
| `features` | np.ndarray | 187-dim average framework features |
| `feature_names` | list | 187 feature name strings |
| `n_windows` | int | Number of sliding windows processed |
| `chemoprint` | np.ndarray (property) | `features[:min(29, len)]` zero-padded to 29 dims; `zeros(29)` when empty |

### 2.9 Dependencies

- Python 3.10+ (project targets 3.12)
- numpy, pandas, scikit-learn, scipy
- For the `.osmell` IO layer: no extra dependency (stdlib `zipfile`)

### 2.10 Note on feature count

Legacy READMEs and the root `README.md` state **145 features**. The current
framework extracts **187**: the 145 legacy dimensions plus 7 per-channel
*advanced* features (saturation index + six multi-exponential decay constants
`tau1/tau2/tau3`, `a1/a2/a3`) added in milestone M3 (see §5 and §9).

### 2.11 Related

- `interoperability/` — canonical experiments validating the framework.
- `Chemoprint/` — 29-dim physicochemical descriptor from SMILES.

---

## 3. The `.osmell` Format Specification (v1.1.0)

**Version:** 1.1.0 · **Status:** Draft for review · **Authors:** OpenSmell project
**Format extension:** `.osmell` · **MIME type:** `application/vnd.opensmell.osmell`
(alias `application/x-osmell`)

### 3.1 Overview

The OSMELL format ("Open Smell Exchange format") is a **sensor-agnostic
container** for e-nose recordings. A single `.osmell` file holds:

- the **raw time-series data** from the sensor array,
- a **machine-readable manifest** describing device, channels, sampling rate,
  and session,
- an explicit record of the **baseline / target protocol** that makes MOX
  measurements comparable across devices, days, and operators,
- an optional **data-quality report** computed from the recording itself.

The format is deliberately boring: a ZIP archive containing one CSV and one
JSON file. Everything inside is plain text, inspectable with any unzipper.

#### Design goals

1. **Sensor-agnostic.** The container stores data + metadata; sensor-specific
   math lives in pipelines keyed by `sensorType` (MOX today; MIRIS spectral
   arrays or electrochemical cells tomorrow).
2. **Explicit sessions.** Baseline and exposure are recorded and labeled
   *separately and explicitly*. No software ever guesses which part of a stream
   was "the baseline".
3. **Session-invariant.** With a recorded baseline `R0`, MOX responses normalize
   to `(R − R0)/R0`, making recordings comparable across hardware and conditions.
4. **Rigorously documented.** Every header, unit, and scoring formula is defined
   with no hidden magic numbers.
5. **Easy.** One ZIP write to create; one ZIP read + one CSV parse to read.

#### Format at a glance

```
coffee_2026-08-01.osmell
├── manifest.json          (required) — format version, device, channels, session
├── data.csv               (required) — time-series samples
├── baseline.csv           (optional) — explicit baseline recording
├── events.json            (optional) — labeled intervals
└── quality.json           (optional) — computed quality report
```

Members are stored uncompressed or deflate-compressed. Filenames are
case-sensitive and exactly as listed.

### 3.2 Naming and registration

- Extension `.osmell` (lowercase). A `.osmell` file is a valid ZIP archive
  (`PK\x03\x04` magic); readers MUST verify the magic before parsing.
- MIME `application/vnd.opensmell.osmell`.
- `.osm` is registered to OpenStreetMap, `.osx` to Apple, `.mox` is claimed by
  unrelated vendors — the ecosystem uses a name it owns. v1.0.0 succeeds the
  informal CSV convention `timestamp_ms, VOC, Alcohol, LPG, CO, NO2, C2H5OH`.

### 3.3 Manifest (`manifest.json`)

A single UTF-8 JSON object, no trailing comments. Unknown fields MUST be
preserved by re-exporters.

```json
{
  "osmell": {
    "formatVersion": "1.1.0",
    "specUrl": "https://github.com/opensmell/osmograph-web/blob/main/OSMELL_FORMAT_SPEC.md"
  },
  "sensor": {
    "sensorType": "mox",
    "device": {
      "model": "Osmograph v1",
      "serial": "Osmograph-A1B2C3",
      "firmware": "0.4.0"
    },
    "channels": [
      { "id": "VOC",    "unit": "adc", "target": "VOC"   },
      { "id": "Alcohol","unit": "adc", "target": "Ethanol"},
      { "id": "LPG",    "unit": "adc", "target": "LPG"    },
      { "id": "CO",     "unit": "adc", "target": "CO"    },
      { "id": "NO2",    "unit": "adc", "target": "NO2"   },
      { "id": "C2H5OH", "unit": "adc", "target": "Ethanol"}
    ],
    "samplingRateHz": 10.0,
    "adcBits": 12,
    "adcMax": 4095,
    "timeColumn": "timestamp_ms"
  },
  "session": {
    "role": "exposure",
    "label": "coffee",
    "groupId": "c8f3a9c1-7b2e-4f5d-9a6b-1e2f3a4b5c6d",
    "recordedAt": "2026-08-01T04:00:00.000Z",
    "durationMs": 60000,
    "notes": ""
  },
  "baseline": {
    "source": "auto",
    "file": "baseline.csv",
    "r0Samples": 15
  },
  "software": {
    "recorder": "Osmograph 0.4.0",
    "importer": "Osmograph Web 0.1.0"
  }
}
```

#### `osmell` object (required)

| Field | Type | Required | Meaning |
|---|---|---|---|
| `formatVersion` | string | yes | This spec version, e.g. `"1.1.0"` |
| `specUrl` | string | no | Canonical URL of the spec |

#### `sensor` object (required)

| Field | Type | Required | Meaning |
|---|---|---|---|
| `sensorType` | string | yes | `mox`, `miris`, `electrochemical`, `other`, `unknown` |
| `device.model` | string | no | Human-readable device model |
| `device.serial` | string | no | Device serial, e.g. `Osmograph-XXXXXX` |
| `device.firmware` | string | no | Firmware version |
| `channels` | array | yes | Ordered channel descriptors (see §3.4) |
| `samplingRateHz` | number | no | Nominal sampling rate in Hz |
| `adcBits` | integer | no | ADC resolution in bits |
| `adcMax` | number | no | Full-scale ADC value (12-bit → `4095`) |
| `timeColumn` | string | yes | `timestamp_ms` or `elapsed_ms`; MUST be the first CSV column |

`(adcBits, adcMax)` is informational; quality scoring uses `adcMax` as the
clipping bound, falling back to a default of `4095` (noted in the report).

#### `session` object (required)

| Field | Type | Required | Meaning |
|---|---|---|---|
| `role` | string | yes | `baseline`, `exposure`, or `single` |
| `label` | string | no | Free-text sample label (e.g. `"coffee"`) |
| `groupId` | string | no | UUIDv4 linking baseline + exposures |
| `recordedAt` | string | no | ISO 8601 UTC timestamp |
| `durationMs` | number | no | Duration in ms |
| `notes` | string | no | Free text |

#### `baseline` object (optional)

| Field | Type | Required | Meaning |
|---|---|---|---|
| `source` | string | yes | `explicit`, `auto`, or `none` — MUST be set |
| `file` | string | no | Member filename; required when `source == "explicit"` |
| `r0Samples` | integer | no | Leading samples for auto-R0 (default `15`) |

#### `software` object (optional)

| Field | Type | Meaning |
|---|---|---|
| `recorder` | string | Software/hardware that produced the recording |
| `importer` | string | Software that created this `.osmell` from raw data |

### 3.4 Data (`data.csv`)

- UTF-8, no BOM, header first, column names unique and matching
  `sensor.channels[].id` plus exactly one time column.
- Time column first, integer milliseconds: `timestamp_ms` (wall-clock, UTC,
  preferred) or `elapsed_ms` (since recording start). Rows sorted ascending;
  readers SHOULD reject or re-order (with warning) unsorted files.
- Values MUST be finite IEEE-754 numbers; `NaN`/`Inf`/`-Inf` are invalid and
  count against the quality score.
- Missing data is represented by **omitted rows** (gaps in the time column),
  never sentinels.

#### Channel descriptors

| Field | Type | Required | Meaning |
|---|---|---|---|
| `id` | string | yes | Column name in `data.csv`; MUST match header exactly |
| `unit` | string | yes | Physical unit; SHOULD be `adc`, `volt`, `ohm`, `ppm`, `norm` |
| `target` | string | no | Target analyte or role |

`mox` pipelines operate on `adc`/`ohm`; normalize *after* reading, never by
mutating the stored file.

### 3.5 Baseline file (`baseline.csv`, optional)

A separate pre-exposure (clean-air) recording with the **same** header as
`data.csv`. Present when `session.role == "baseline"` or when
`baseline.source == "explicit"`.

### 3.6 Events (`events.json`, optional)

Array of labeled intervals delimiting phases within one continuous recording:

```json
[
  {
    "label": "exposure",
    "startMs": 5000,
    "endMs": 35000,
    "note": "lid opened"
  }
]
```

| Field | Type | Required | Meaning |
|---|---|---|---|
| `label` | string | yes | Event/phase label |
| `startMs` | number | yes | Start offset in ms from first sample |
| `endMs` | number | no | End offset; open-ended if absent |
| `note` | string | no | Free text |

Events annotate phases *inside a single file*; the recommended workflow still
records baseline and exposure as **separate files**.

### 3.7 Sensor-agnostic routing

`manifest.sensor.sensorType` selects the processing pipeline; the container is
identical for all types.

| sensorType | Pipeline intent |
|---|---|
| `mox` | MOX resistance kinetics: R0 normalization, rise/decay, AUC, saturation index |
| `miris` | Spectral absorption arrays (reserved) |
| `electrochemical` | Amperometric/potentiometric cells (reserved) |
| `other`/`unknown` | Generic pass-through: no normalization, basic quality only |

Readers MUST NOT infer `sensorType` from channel names. Missing → treat as
`unknown` and surface a warning.

### 3.8 Session protocol

MOX sensors are temperature- and humidity-sensitive; two recordings of the same
odor on different days can look very different in raw counts yet identical after
baseline normalization.

**Recommended workflow (separate files):**
1. Record a **baseline** (clean air / empty chamber) immediately before the
   experiment; mark `role: "baseline"`.
2. Record one **exposure** per sample; `role: "exposure"`, `label` = sample name.
3. All recordings in one session share the same `groupId`.

**Import Wizard accommodation (single files):** a continuous baseline-then-
exposure file is split at a **user-confirmed midpoint boundary** into two
`.osmell` files with a shared `groupId`. The boundary is always
human-confirmed; software never auto-detects phases.

**Auto-R0:** when no baseline file exists, `R0` = median of the first
`r0Samples` samples (default 15, ≈1.5 s at 10 Hz). Valid only when the leading
window is known clean-air. Files using auto-R0 MUST declare
`baseline.source: "auto"` — an explicit declaration of **weaker evidence**:
consumers MUST NOT treat it as equivalent to an explicit baseline for
cross-session comparison, and the quality report MUST flag it. Writers MUST use
`"explicit"` when a baseline is linked and `"none"` when no R0 exists; never
leave `source` unset.

### 3.9 MOX normalization and features

**Definitions** — for channel values `x[0..N-1]` and baseline `R0`:

```
R0   = median of the baseline recording (per channel), or
       median of x[0..r0Samples-1] when baseline.source == "auto"

normalized[i] = (x[i] − R0) / R0
```

`normalized[i]` is unitless and device-agnostic — the relative change in signal
against its reference. `(R − R0)/R0` is the canonical MOX response measure and
is insensitive to multiplicative gain drift and to the specific baseline
resistance of the sensor element.

**Dead-sensor guard:**

```
cv = std(x) / R0   (R0 > 0)
dead  ⟺  cv < 0.001
```

Dead channels MUST be excluded from derived features and flagged in the quality
report. This mirrors the reference SDK guard.

**Feature framework (informative):** 187 features per recording, including
relative amplitude `max|(x − R0)/R0|` and response direction, rise/decay time,
AUC, saturation index, and multi-exponential recovery fits. The `.osmell` file
stores *raw data*, never pre-computed features, so features can always be
recomputed. Pre-computed values may be cached in `quality.json`/`features.json`
and MUST be labeled as derived.

### 3.10 Data-quality scoring

The quality report (`quality.json`) is a **0–100 score composed of seven
sub-scores**, each defined by an explicit, hard-to-vary formula computed from
the recording itself — no external truth.

| Sub-score | Weight | Formula |
|---|---|---|
| Continuity `C` | 0.15 | `regular(k) ⟺ |g[k] − T| ≤ 0.10·T` (T = 1000/samplingRateHz ms); `C = 100 · regular/(N − 1)`; median-gap T when rate undeclared; single sample scores 100 |
| Dynamic range `D` | 0.10 | `span_k = (max(x) − min(x))/adcMax`; `D_k = 100·clamp(span_k·10, 0, 1)`; mean over channels (dead excluded) |
| Saturation-free `S` | 0.10 | `clipped_k = count(x ≥ adcMax OR x ≤ 0)`; `S_k = 100·(1 − clipped_k/N)`; mean over channels |
| Baseline stability `B` | 0.20 | `cv_window = std(window)/R0`; `B = 100·clamp(1 − cv_window/0.05, 0, 1)`; no R0 → 0 (`no_baseline`); **auto-R0 capped at 50** (reason `auto_r0`) |
| Signal strength / SNR `G` | 0.20 | `peak_k = max|(x − R0)/R0|`; `noise_k = std(window)/R0`; `SNR = peak/max(noise, 1e-6)`; `G_k = 100·clamp(SNR/10, 0, 1)`; `G = max over channels`; null unless exposure + R0 |
| Recovery completeness `R` | 0.15 | `final_win_k = median of last 15 normalized`; `recovered = 1 − clamp(|final_win|/max(peak, 1e-6), 0, 1)`; `R_k = 100·recovered`; mean; null unless exposure + R0 |
| Duration adequacy `T` | 0.10 | `t_s = (N − 1)/samplingRateHz`; `T = 100·clamp(t_s/60, 0, 1)`; median gap when rate undeclared |

**Total and badge:**

```
weights = { C: 0.15, D: 0.10, S: 0.10, B: 0.20, G: 0.20, R: 0.15, T: 0.10 }
sum_w   = Σ w_i over non-null sub-scores
score   = round( Σ w_i · sub_i / sum_w )
```

| Score range | Badge | Guidance |
|---|---|---|
| 90–100 | Excellent | Ready for analysis and publication |
| 75–89 | Good | Ready; review flagged notes first |
| 50–74 | Fair | Usable with caveats listed |
| 0–49 | Poor | Investigate hardware/protocol before use |

`G` and `R` are null for non-exposure roles.

**Report shape:**

```json
{
  "format": "opensmell-quality",
  "version": "1",
  "computedAt": "2026-08-01T04:05:00.000Z",
  "total": 87,
  "badge": "Good",
  "subscores": {
    "continuity": 92, "dynamicRange": 81, "saturationFree": 100,
    "baselineStability": 88, "signalStrength": null, "recoveryCompleteness": 85,
    "durationAdequacy": 74
  },
  "flags": {
    "deadSensors": ["NO2"],
    "unsortedRows": false,
    "nonFiniteSamples": 0,
    "usedDefaultAdcMax": false,
    "usedMedianSamplingRate": false
  },
  "reasons": {
    "baselineStability": "r0_window_cv_too_high",
    "signalStrength": "no_exposure_signal"
  },
  "notes": []
}
```

Every sub-score MUST carry the reason it was reduced below 100.

### 3.11 Validation rules

A conforming **reader** MUST: verify the ZIP magic; require `manifest.json` +
`data.csv`; parse the manifest as UTF-8 JSON (reject on failure, no partial
data); require `osmell.formatVersion` + `sensor.sensorType` (warn on
`unknown`); verify the CSV header matches `sensor.channels[].id` plus exactly
one time column named by `timeColumn`; parse all values as finite numbers;
sort-check the time column.

A conforming **writer** MUST: use exact member casing; write `data.csv` with
`\n` row endings (`\r\n` accepted on read); emit `session.role`, `session.label`
when supplied, and a shared `groupId` across a split session; never write
non-finite values; preserve unknown members and manifest fields on re-export.

### 3.12 Extensibility and versioning

- **Backward compatibility:** readers MAY read any `formatVersion` with the same
  major version (`1.*`); unknown fields/members are ignored but preserved.
- **Forward compatibility:** writers MUST preserve unknown fields/members.
- **Major versions** (e.g. `2.0.0`) may change required fields or scoring
  formulas; they are opted into explicitly by the reader.

### 3.13 Reference implementations

| Implementation | Repository | Status |
|---|---|---|
| Osmograph desktop (Python) | `github.com/opensmell/osmograph` | Current recorder; emits legacy CSV, upgraded via Import Wizard |
| OpenSmell SDK (Python) | `opensmell/opensmell` | Feature framework; reads/writes `.osmell` via `opensmell.io` |
| Osmograph Web (TypeScript) | `osmograph-web` | Reference parser, scorer, writer |

### 3.14 Appendix A — Legacy CSV upgrade path

Legacy CSVs use `timestamp_ms, VOC, Alcohol, LPG, CO, NO2, C2H5OH` with 12-bit
ADC values. The Import Wizard converts them by: reading the header into
`sensor.channels` (`unit:"adc"`, `adcBits:12`, `adcMax:4095`,
`timeColumn:"timestamp_ms"`); guessing `samplingRateHz` from the median gap;
asking for `role`/`label`/baseline; midpoint-splitting single files at a
user-confirmed boundary.

### 3.15 Appendix B — Minimal example

```
$ unzip -l coffee.osmell
  manifest.json
  data.csv

$ cat data.csv
timestamp_ms,VOC,Alcohol,LPG,CO,NO2,C2H5OH
1750000000000,2048,2010,1998,2005,2022,2033
1750000000100,2049,2011,1998,2006,2021,2033
...
```

With `manifest.json` declaring `role:"baseline"`, `samplingRateHz:10`, channels
matching the header, and a `groupId` shared with the exposure files.

---

## 4. Feature Reference: the 187-Dimension Framework

> Complete technical documentation of every feature extracted from MOX
> time-series data, built on the OpenSmell Interoperability Framework — a
> device-agnostic approach to digital olfaction.

### 4.1 Architecture overview

Features are extracted along five dimensions, each answering a distinct
question:

| Dimension | Question It Answers | Use Case |
|---|---|---|
| Device-Agnostic | "What is the substance, independent of the device?" | Cross-device substance identification |
| Absolute | "What is the actual concentration?" | Quantitative sensing, regulatory compliance |
| Temporal | "How fast did the substance appear/disappear?" | Safety systems, gas leak detection |
| Health | "Is the sensor working properly?" | Predictive maintenance, data quality |
| Hardware | "What device measured this?" | Device fingerprinting, circuit diagnostics |

Plus: **advanced kinetic features** (multi-exponential decay + saturation
index), **cross-channel selectivity ratios** (chemical fingerprint between
sensor pairs), and **global metrics** (aggregate statistics across the array).

**Normalization pipeline:**

```
Raw voltage → Rs/R0 normalization → Segmentation → Feature extraction → Fingerprint vector
```

The `Rs/R0` normalization mathematically cancels all circuit-level constants
(`Vcc`, `RL`); see §4.6 for the proof.

**Recording protocol** (required for reliable temporal features):

```
Phase 1: Baseline (30-60s) → Phase 2: Exposure (30-60s) → Phase 3: Recovery (60-180s)
    Clean air          Substance introduced         Clean air
```

### 4.2 The five dimensions (per-channel features)

`N_CHANNELS = 6`. Names follow `ch{N}_<dim>_<feature>`. All ranges/units below
describe typical MOX behavior; exact computation lives in
`opensmell/opensmell/mox/features.py`.

#### Dimension 1: Device-Agnostic (6 per channel)

Normalization: `Rs/R0`, `R0` = median of the first 15 samples. `-1.0` marks an
unavailable value (too-short series, dead channel).

| Feature | Name | Range | Unit | Meaning |
|---|---|---|---|---|
| `relative_amplitude` | `ch{N}_da_relative_amplitude` | 0.0–∞ (typ. 0.01–5.0) | dimensionless | Peak normalized resistance change; primary quantitative signal; ∝ concentration via power law `Rs/R0 = a·C^b` |
| `direction` | `ch{N}_da_direction` | −1, 0, +1 | categorical | +1 reducing (resistance drops — most VOCs/alcohols/H₂), −1 oxidizing (resistance rises — O₃, NO₂, Cl₂), 0 no response |
| `rise_time` | `ch{N}_da_rise_time` | −1.0 or 0–60+ s | seconds | 10%→90% of peak during exposure; adsorption kinetics; small volatile molecules fast (<3 s), bulky molecules slow (>10 s) |
| `decay_time` | `ch{N}_da_decay_time` | −1.0 or 0–180+ s | seconds | 90%→10% of peak during recovery; desorption/binding strength; weakly bound fast (<5 s), strongly bound slow (>30 s) |
| `auc` | `ch{N}_da_auc` | 0.0–∞ | dimensionless | Area under `|R(t)−R₀|/R₀`; total chemical dose (concentration × time) |
| `endpoint_delta` | `ch{N}_da_endpoint_delta` | −∞ to ∞ (typ. −1..1) | dimensionless | Final reading vs baseline; near 0 = complete recovery; persistent nonzero = contamination/drift |

#### Dimension 2: Absolute (4 per channel)

No normalization — raw or calibrated values.

| Feature | Name | Range | Unit | Meaning |
|---|---|---|---|---|
| `raw_resistance` | `ch{N}_abs_raw_resistance` | 0–10⁶+ | Ω | Surface resistance at end of window; the fundamental observable |
| `baseline_resistance` | `ch{N}_abs_baseline_resistance` | 100–10⁶+ | Ω | Clean-air resistance R₀; drifting R₀ ⇒ aging/environment change |
| `voltage` | `ch{N}_abs_voltage` | 0–V_cc | V | Circuit-level ADC reading; `V(t) = V_cc·RL/(Rs(t)+RL)` |
| `calibrated_concentration` | `ch{N}_abs_calibrated_concentration` | 0–10000+ | ppm | Derived from `C = (Rs/(a·R₀))^(1/b)` with datasheet constants a/b (empirical default `a=1, b=−0.5`). An **estimate**, not certified measurement |

Typical baselines (MQ at 25 °C, 50% RH): MQ-135 100k–1M Ω (NH₃/NOx/VOC), MQ-3
200k–2M Ω (alcohol), MQ-7 10k–100k Ω (CO), MQ-2 5k–50k Ω (LPG/propane/H₂).

#### Dimension 3: Temporal (4 per channel)

| Feature | Name | Range | Unit | Meaning |
|---|---|---|---|---|
| `hf_transient` | `ch{N}_temp_hf_transient` | 0.0–∞ | mean\|dR/dt\|/R₀ | Mean absolute rate of change; fast chemical events, leaks, noise |
| `oscillation_freq` | `ch{N}_temp_oscillation_freq` | 0–sr/2 | Hz | Dominant frequency via periodogram of detrended signal (0 = stable; 0.1–0.5 Hz = environmental cycles; >1 Hz = mains/PWM noise) |
| `oscillation_amp` | `ch{N}_temp_oscillation_amp` | 0.0–∞ | dimensionless | √peak PSD; paired with freq to identify noise sources |
| `response_latency` | `ch{N}_temp_response_latency` | −1.0 or 0–60 s | seconds | Time until signal exceeds 3σ of baseline noise; first detectable exposure moment |

#### Dimension 4: Health (4 per channel)

| Feature | Name | Range | Unit | Meaning |
|---|---|---|---|---|
| `drift_rate` | `ch{N}_health_drift_rate` | −∞ to ∞ | fractional/recording | `(mean(series[-10:]) − R₀)/R₀`; yellow >0.10, red >0.20 |
| `sensitivity_decay` | `ch{N}_health_sensitivity_decay` | 0.0–∞ | fractional | Long-term degradation of response magnitude (defaults to 0 within a single recording) |
| `noise_floor` | `ch{N}_health_noise_floor` | 0.0–∞ | σ(R)/R₀ | `std(R[0:r0_samples])/R₀`; <0.01 excellent, 0.01–0.03 normal, >0.10 very noisy |
| `hysteresis` | `ch{N}_health_hysteresis` | 0.0–1.0 | dimensionless | `\|∫ads − ∫des\|/max(∫ads, ∫des)`; >0.5 indicates surface poisoning |

#### Dimension 5: Hardware (3 per channel)

| Feature | Name | Range | Unit | Meaning |
|---|---|---|---|---|
| `circuit_response` | `ch{N}_hw_circuit_response` | 0–4095 (ADC) | raw ADC/volt | Mean signal level; the device fingerprint (Vcc/RL/sensor baseline) |
| `thermal_profile` | `ch{N}_hw_thermal_profile` | 0.0–∞ | std of signal | Std dev; captures heater cycling / temperature behavior |
| `adc_noise` | `ch{N}_hw_adc_noise` | 0.0–∞ | std of residuals | HF noise after removing a 5-sample moving average; ESP32 ADC ≈2–5 counts, ADS1115 ≈0.5–1, Pi ≈3–7 |

### 4.3 Advanced features (7 per channel)

#### Multi-exponential decay constants

`ch{N}_decay_tau1/2/3`, `ch{N}_decay_a1/2/3` — range −1.0 (fit failed) or
0.0–60+ s (tau) / amplitude (a). The recovery phase is fit to a sum of
exponentials (see §5 for the model, defaults, and reproducibility). Typical
time constants: τ₁ fast 0.5–5 s (weakly adsorbed surface species), τ₂ medium
5–15 s (moderately bound species in pores), τ₃ slow 15–60+ s (strongly bound or
reacted species). The ratio τ₂/τ₁ indicates surface heterogeneity. A large,
growing τ₃ across recordings is a precursor to sensor poisoning.

#### Saturation index

`ch{N}_advanced_saturation_index` — range 0.0 (no response) to 1.0 (fully
saturated), dimensionless. Langmuir-like approximation:

```
saturation = R_obs / (R_obs + k·noise_floor)     k ≈ 10
```

Regimes: 0.0–0.3 linear (quantitative), 0.3–0.7 nonlinear (response
compresses), 0.7–1.0 saturated. For quantitative measurements keep saturation
< 0.5; above 0.8, dilute or use a less sensitive sensor.

### 4.4 Cross-channel features — selectivity ratios

`sel_ratio_ch{i}_ch{j}` for all sensor pairs (i < j):

```
selectivity_ratio(i,j) = amplitude_i / amplitude_j
```

Range 0.0–∞, dimensionless. This is the **most discriminative feature group**:
the framework ablation study shows removing selectivity ratios causes the
largest drop in classification accuracy (4.9 percentage points). Ratios are
approximately **concentration-invariant** in the linear regime, which is how
substances are identified independent of concentration.

**Total count:** `N(N−1)/2` (15 for 6 sensors, 3 for 3 sensors).

Example ethanol ratios (MQ-3 / MQ-135 = 2.5, MQ-7 / MQ-135 = 0.3,
MQ-3 / MQ-7 = 8.3).

### 4.5 Global metrics

| Feature | Meaning |
|---|---|
| `global_max_delta_ratio` | Max relative_amplitude across active channels (strongest sensor) |
| `global_mean_delta_ratio` | Mean relative_amplitude across active channels (overall intensity) |
| `global_n_active_channels` | Channels with detectable response (>2× noise floor); 1–2 simple headspace, 5–6 complex mixture |
| `global_total_auc` | Sum of AUC across active channels (total chemical exposure) |

### 4.6 The Rs/R0 normalization (mathematical proof)

Starting from the voltage divider `V(t) = V_cc · R_L / (R_s(t) + R_L)`:

```
R_s(t) = R_L · (V_cc − V(t)) / V(t)
R₀     = R_L · (V_cc − V₀) / V₀

R_s(t) / R₀ = [R_L·(V_cc − V(t))/V(t)] / [R_L·(V_cc − V₀)/V₀]
             = [(V_cc − V(t))/V(t)] / [(V_cc − V₀)/V₀]
```

**Result:** both `R_L` and `V_cc` cancel completely. Two devices with different
Vcc (3.3 V vs 5 V) and RL (10 kΩ vs 47 kΩ) but the same sensor model produce
the same `Rs/R₀` curve for the same gas.

**What this does NOT do:** it does not cancel the sensor-specific sensitivity
constants `a` and `b` in `Rs/R₀ = a·C^b` — which vary across models, across
units of the same model (20–30% manufacturing tolerance), and with temperature.
**Zero-shot cross-device transfer is therefore mathematically impossible
without calibration.** A **two-point calibration** determines (a, b) per
channel:

1. Measure response at known `C₁` → `(Rs₁/R₀)`
2. Measure response at known `C₂` → `(Rs₂/R₀)`
3. `b = log(Rs₁/Rs₂) / log(C₁/C₂)`
4. `a = (Rs₁/R₀) / C₁^b`

After calibration, device-agnostic features become fully transferable.

### 4.7 Feature counts by sensor count

| Feature group | Per channel | N=3 | N=6 | N=12 |
|---|---|---|---|---|
| Device-Agnostic | 6 | 18 | 36 | 72 |
| Absolute | 4 | 12 | 24 | 48 |
| Temporal | 4 | 12 | 24 | 48 |
| Health | 4 | 12 | 24 | 48 |
| Hardware | 3 | 9 | 18 | 36 |
| Advanced (saturation + decay) | 7 | 21 | 42 | 84 |
| Selectivity ratios | N(N−1)/2 | 3 | 15 | 66 |
| Global | — | 4 | 4 | 4 |
| **Total** | | **91** | **187** | **406** |

**Total with 6 sensors: 187 features.**

### 4.8 Interpretation guide

A fingerprint is a normalized feature vector, min-max scaled to 0.0–1.0 across
your dataset. The **radar chart** displays 8 key device-agnostic features:
relative amplitude, rise time, decay time, AUC, endpoint delta, saturation
index, selectivity ratio, and number of active channels.

**Substance identification logic:**

```
Unknown fingerprint
       ↓
Cosine similarity to library
       ↓
Best match > 85% → "Identified as [substance]"
Best match 60-85% → "Likely [substance] (verify)"
Best match < 60% → "Unknown — nearest: [substance]"
```

**When features are unreliable:**

| Condition | Affected features | Mitigation |
|---|---|---|
| No recovery phase | decay_time, tau's, hysteresis | Skip temporal features |
| Saturated sensor | relative_amplitude, calibrated_concentration | Check saturation_index first |
| Sensor warming up | all features | Health dimension readiness check |
| Extreme humidity | all features | Humidity compensation |
| Very short recording (<15 s) | rise_time, decay_time | Use amplitude + AUC only |
| Concentration unknown | selectivity_ratios degrade | Ratios are concentration-invariant — still valid |

### 4.9 Use-case recipes

- **Food spoilage:** target ethylene/ammonia/sulfides; key features relative
  amplitude (ammonia), selectivity ratios, drift_rate (rising baseline);
  pattern = increasing AUC, MQ-135 dominant, endpoint_delta accumulation.
- **Gas leak:** methane/propane/CO; key features hf_transient, response_latency,
  oscillation_freq; pattern = transient spike, short latency, MQ-2/MQ-7
  dominant.
- **Breath analysis (research):** acetone/ethanol/ammonia; selectivity ratios,
  multi-exp decay, saturation_index; MQ-3 dominant for ethanol.
- **Environmental monitoring:** NO₂/O₃/VOCs; direction (oxidizing vs reducing),
  noise_floor, drift_rate; NO₂ gives negative direction.

---

## 5. Multi-Exponential Decay Model

Source: `opensmell/spec/ALGORITHMS.md`; implementation
`opensmell/opensmell/mox/features.py::compute_multi_exp_decay`.

### 5.1 Model forms

```
Single: y(t) = a1·exp(-t/tau1) + c
Double: y(t) = a1·exp(-t/tau1) + a2·exp(-t/tau2) + c
Triple: y(t) = a1·exp(-t/tau1) + a2·exp(-t/tau2) + a3·exp(-t/tau3) + c
```

The default model is **bi-exponential** (`compute_multi_exp_decay(..., n_components=2)`):
a fast component (tau1 ≈ 1–3 s) for weakly adsorbed surface species and a slow
component (tau2 ≈ 10–30 s) for strongly bound species. The ratio tau2/tau1 is a
measure of surface heterogeneity.

The **tri-exponential** model is over-parameterized for MOX recovery data: the
three components are not uniquely identifiable, so the fit can alternate
between near-equal-cost local minima. It is therefore **opt-in only** via
`n_components=3` (and `len(recovery) >= 30`) and is documented as advanced and
possibly nondeterministic.

### 5.2 Fit mechanics

- `compute_multi_exp_decay(series, peak_idx=None, sr=10, n_components=2)`.
- Fits are attempted single-exp, then bi-exp, then (opt-in) tri-exp; each more
  complex model is only attempted when enough data exists. The final result
  holds the best-of-chain params and a `cost` key (residual sum of squares of
  the final fit).
- Returns `-1.0` for all tau/amplitude values (and `cost`) when fitting fails
  (short series, constant signal, or optimizer non-convergence).
- Feature names: `ch{N}_decay_tau1/2/3`, `ch{N}_decay_a1/2/3` (tau3/a3 are
  `-1.0` for the default bi-exponential fit).

### 5.3 Reproducibility note

The decay fit uses scipy's MINPACK optimizer (`scipy.optimize.curve_fit`). On
data where several parameter sets produce essentially the same residual cost,
MINPACK may converge to different local minima from one run to the next.

- **Decay tau values may vary slightly across runs** (measured up to ~6%
  relative difference), and occasional tau/amplitude values may flip to the
  `-1.0` "fit failed" sentinel when the optimizer reports non-convergence.
- **Fit quality (residual cost) remains consistent across runs** — the
  differing solutions are equal-cost minima, not worse fits.

This is a known limitation of the optimizer on equal-cost solutions, not a flaw
in the model. All returned solutions are equally valid fits to the recovery
curve.

**Guarantees enforced by `tests/test_legacy_api.py::test_determinism`:**

1. Every non-decay feature is byte-identical between runs.
2. Decay tau values agree within 10% relative difference whenever both runs
   produced a valid fit.
3. At most 20 features may differ at all (observed range 0–20 across 91
   pairwise runs; decay amplitudes may flip between equal-cost solutions).

For publication-grade work, verify fits manually or fix the optimizer seed.

---

## 6. Sensor Theory

Source: `opensmell/docs/SENSOR_THEORY.md`.

### 6.1 How electronic noses work

An e-nose is an array of cross-sensitive gas sensors; each responds to multiple
volatile compounds with a different sensitivity profile. The combined response
is a **pattern** (a vector in N-dimensional space) matched against known
substances.

**Cross-sensitivity** — a MOX sensor (e.g. MQ-135) responds to NH₃, NOₓ,
alcohol, benzene, smoke, and CO₂ simultaneously, each with different
sensitivity. Not a bug; the mechanism. The overlap creates a unique
"fingerprint" across the array.

**The inverted curse of dimensionality** — more sensors = more dimensions =
more separation, but real gas sensors are correlated (temperature and humidity
affect all MOX sensors similarly), so **effective dimensionality** is always
less than the raw sensor count.

**Johnson-Lindenstrauss lemma** — any set of *n* points in high-dimensional
space can be embedded into ~O(log n) dimensions preserving pairwise distances
approximately. To reliably distinguish *n* substances you need at least
`8 ln(n)/e²` dimensions for distortion e: ~12 dimensions for 50 substances at
e = 0.5, ~34 at e = 0.3. Correlated responses push this higher in practice.

### 6.2 Sensor count vs discriminative power

| Sensors | Effective dims | Max substances | Worth building? | Typical use |
|---|---|---|---|---|
| 1 | ~0.5–1 | 1 | Detector, not nose | Leak alarm |
| 2 | ~1–1.5 | 2–4 | Marginal | Binary QC |
| 3 | ~1.5–2 | 3–7 | Proof-of-concept | Beverage ID |
| 4 | ~2–3 | 8–12 | Conditional | Broad categories |
| 5 | ~3–4 | 12–20 | Yes | Most applications |
| 6 | ~4–5 | 20–40 | Yes | Research-grade |
| 8+ | ~5–6 | 30–60 | Diminishing returns | Specialized |

A single sensor is a **gas detector**, not a nose (threshold "is there
something?" only). Two same-family MOX sensors correlate strongly (effective
dims ≈1). Three give real 2D separation after normalization. Four is the entry
point for practical substance identification. Six is a standard research
configuration. The 6th sensor typically adds <10% separation over the 5th.

### 6.3 Sensor selection matters more than count

| Array type | Raw count | Effective dims | ~Distinguishable substances |
|---|---|---|---|
| All MQ-series MOX | 6 | 3–4 | 12–20 |
| MQ + electrochemical + PID | 4 | 3–4 | 12–20 |
| All identical MOX | 6 | 1 | 1 |
| Micro-hotplate array (different T) | 4 | 3–5 | 8–30 |

### 6.4 Johnson-Lindenstrauss guideline

```
d_min > 8 ln(n) / e²
```

| n (substances) | e = 0.5 | e = 0.3 | e = 0.1 |
|---|---|---|---|
| 5 | 4 | 10 | 92 |
| 10 | 5 | 13 | 115 |
| 20 | 6 | 15 | 133 |
| 50 | 7 | 17 | 157 |

In practice, correlations double or triple these numbers.

### 6.5 Failure modes common to all sensor counts

1. **Humidity dominance** — water vapor affects all MOX sensors; differential
   measurement or per-recording z-score normalization essential.
2. **Temperature drift** — MOX sensitivity shifts 2–5%/°C; onboard logging +
   compensation required.
3. **Sensor poisoning** — H₂S, siloxanes, halogens permanently degrade MOX;
   lifespan 1–3 years clean air, weeks in harsh environments.
4. **Baseline drift** — periodic auto-zero against reference clean air mandatory.
5. **Mixture ambiguity** — a 50/50 blend can look like a pure third substance; a
   fundamental limit of e-nose technology.
6. **Batch variation** — ±20% sensitivity between same-part-number sensors;
   every unit needs individual calibration.

### 6.6 References

- Johnson & Lindenstrauss (1984), *Contemporary Mathematics* 26, 189–206.
- Gardner & Bartlett (1994), *Sens. Actuators B* 18(1-3), 210–211.
- Wilson & Baietto (2009), *Sensors* 9(7), 5099–5148.
- Marco & Gutierrez-Galvez (2012), *Sens. Actuators B* 166-167, 217–226.

---

## 7. Osmograph Web Platform

Source: `osmograph-web/DOCUMENTATION.md`. Repo `osmograph-web/`, app served at
`mox.opensmell.xyz`, stack Next.js 16 (App Router, Turbopack), TypeScript,
Tailwind v4, shadcn/ui, jszip, recharts, vitest.

### 7.1 The product at a glance

One shell (`components/suite/suite-shell.tsx`) holds five views:

| View | Nav id | What it does |
|---|---|---|
| Library | `library` | All imported sessions grouped by experiment (`groupId`); per-session quality badge, detail drawer, `.osmell` export |
| Import | `import` | Drop `.csv`/`.txt`/`.osmell` (many at once); auto channel detection, baseline normalization, quality scoring |
| Compare | `compare` | Overlay sessions normalized to their own R0 on a shared relative-time axis |
| Train | `train` | Readiness-gated classifier trainer (per-class minimums); pipeline ships later |
| Smellability | `smellability` | Physics/chemistry verdict: detectability, strength, speed, and library confusability |

State lives in `SessionProvider` (`session-context.tsx`); nothing persisted
except localStorage keys (§7.9). No accounts, no backend.

### 7.2 Normalization math (`lib/osmell/normalize.ts`)

```
R0 = median(baseline channel)                              // baseline.source == "explicit"
   = median(x[0..r0Samples-1])                             // baseline.source == "auto"  (default 15)
normalized[i] = (x[i] − R0) / R0
cv = std(x) / R0                                           // R0 > 0
dead  ⟺  cv < 0.001                                        // DEAD_CV_THRESHOLD
```

- R0 fallback when median ≤ 0: mean of positive values, else 1.
- `normalizedSeries` returns NaN series if R0 not finite/≤ 0.

### 7.3 Data-quality scoring: spec vs shipped (unreconciled discrepancy)

The shipped implementation (`lib/osmell/quality.ts`, `session-cards.tsx`)
scores **six** sub-scores — Recovery completeness is **not implemented** — with
weights `{ continuity: 0.20, dynamicRange: 0.15, saturationFree: 0.15,
baselineStability: 0.20, signalStrength: 0.20, durationAdequacy: 0.10 }`.

Differences vs spec §7 (§3.10 above):
- No `recoveryCompleteness`; its 0.15 weight redistributed to continuity
  (+0.05) and dynamicRange/saturationFree (+0.05 each).
- Baseline stability is **not capped at 50** for `auto` sources (spec says it
  must be); `flags.noBaseline` only drives a note and a 0 score when `none`.
- Duration uses `sampleCount / samplingRateHz` (not `(N−1)/rate`).
- Continuity returns 50 (`irregular_gaps` + `no_sampling_rate_declared`) when no
  rate is declared instead of using the median gap.
- Signal strength: `noise = max(base.cv, 1e-6)`; `G = max over channels`.
- Badge adds `"Unknown"` when total is null (non-exposure role).

The Library `QualityCard` renders the six implemented sub-scores with the
implemented weights.

### 7.4 MOX feature extraction (`lib/osmell/processors.ts`)

`processMox(file)` per channel (dead channels excluded): `relativeAmplitude` =
max|normalized|; `direction` = ±1 (sign of peak); `riseTimeMs` = 10%→90% span
crossing; `auc` = trapezoidal integral; `decayTimeMs` **reserved (always null)**.
`runProcessor` routes by sensorType; `guessSensorType` maps a header to `mox`
when ≥2 of `[VOC, Alcohol, LPG, CO, NO2, C2H5OH]` appear. The web ships a small
subset for display; the full 187-feature framework lives in the Python SDK.

CSV parser (`csv.ts`): quoted-field aware, trims rows, skips `#` comments,
requires `timestamp_ms`/`elapsed_ms`, counts non-finite values, detects +
re-sorts out-of-order rows, guesses sampling rate from median gap.

IO (`io.ts`): `parseOsmell` (jszip, validates header↔channels, reads
`events.json`), `buildOsmell` (manifest + csv + optional events, DEFLATE),
`defaultFileName` → `label_role_date.osmell`.

### 7.5 Smellability engine

> Will this substance produce a detectable response on a MOX array, and how
> strong/fast? Answered with an explicit physics/chemistry chain that never
> fabricates missing data.

It grades **physical feasibility** (volatility × redox against an array capacity
bound), not a calibrated measurement. Everything is built as explicit
`ChainStep` objects so UI, docs, and math describe the same object
(`chain.ts:138 runConstituentChain`).

**Physical constants** (`transport.ts`, `constants.ts`): `R = 8.314` J/(mol·K),
`N_A = 6.022e23`, `P_ATM = 101325` Pa, 1 mmHg = 133.322 Pa, ambient 25 °C /
298.15 K, `MOX_FLOOR_PPM = 1`, reference = ethanol, `DEFAULT_SENSOR_COUNT = 6`,
`DEFAULT_DISTANCE_M = 0.1`.

**Step 1 — Identity & properties.** MW, boiling point, vapor pressure @25 °C,
functional groups, redox activity, CAS, SMILES, odor descriptor, source refs
(`compounds.ts`, 46 curated compounds).

**Step 2 — Volatility.** `effectiveVaporPressure` precedence: (1) curated
`vaporPressure25`; (2) Antoine `log10(P/mmHg) = A − B/(T/°C + C)`, `P_Pa =
10^(A−B/(T+C)) × 133.322`; (3) gas below ambient → `pa = 101325`; (4)
Clausius–Clapeyron from boiling point + Trouton
`P(T) = P_ATM·exp[−(ΔH_vap/R)(1/T − 1/T_boil)]`, `ΔH_vap = 88·T_boil` J/mol;
(5) nothing known → `pa = 0`, source `unknown`.

Volatility bands (Pa @25 °C): very high ≥10⁴, high 10³–10⁴, moderate 10²–10³,
low 1–10², negligible <1.

**Step 3 — Headspace concentration.** `C_headspace = P_vap/P_atm`, in ppm
`(P_vap/101325)×10⁶`, compared against `MOX_FLOOR_PPM = 1`. Grades: ≥1000 ppm
strong, 100–1000 moderate, 10–100 weak, 1–10 marginal, <1 none. The verdict is
driven by **absolute** headspace ppm (ethanol is so volatile that 5% of its
headspace is still thousands of ppm).

Diffusion/incident flux (informational): Fuller–Schettler–Giddings
`D = 0.00143·T^1.75/(P_atm·(V_air^(1/3)+V_i^(1/3))²)·sqrt(1/M_air+1/M)` in
cm²/s, `V_i = 1.1·M`.

**Step 4 — MOX reactivity.** MOX detects reducing gases reacting with O⁻/O²⁻
surface oxygen at ~300–400 °C: alcohols, aldehydes, ketones, esters, alkanes,
alkenes, terpenes, aromatics, thiols, sulfides, amines, H₂, CO, combustibles.
Hard stops (non-redox): N₂, O₂ (not a reducing analyte), CO₂, noble gases.
Boundary: water modulates baseline (humidity) but is not a reducing VOC.

**Step 5 — Array capacity & cross-sensitivity (contextual).** Capacity bound
(`MAX_SUBSTANCES`, from canonical Table 2):

| Sensors | Distinguishable substances |
|---|---|
| 3 | 6 |
| 4 | 12 (interpolated) |
| 5 | 20 (interpolated) |
| 6 | 40 |
| 12 | 200 |
| 24 | 10,000 |

`buildCrossCheck` also scans library labels for name/synonym overlap and flags
"possible overlap" substances.

**Verdict semantics:** `verdict` = green (detectable) / yellow (partially) /
red (not); worst step wins. `signalStrength` = strong/moderate/weak/none.
`responseSpeed`: gas or P_vap ≥ 1000 Pa → fast, ≥ 100 → medium, ≥ 1 → slow,
else unknown. `confidence`: any `unknown` source → low, any `estimated` →
medium, all `measured` → high. Guidance prescribes baseline → exposure →
recovery, tuned by expected signal (short 10–30 s exposures for strong/fast,
maximized headspace + 60–120 s windows for weak, ≈1:10 dilution for
strong/fast).

**Composite (mixture) verdicts** (`composites.ts`, 27 seeded composites): run
the chain per constituent, normalize weight fractions, `redWeight > 0.5` → red,
`nonGreenWeight > 0.4` → yellow; signal/speed from the **dominant** constituent;
confidence low if any constituent unknown. Worked examples:
- **Banana** — isoamyl acetate 0.50 @700 Pa (≈6,900 ppm, strong) + isoamyl
  butyrate 0.15 (≈590 ppm) + butyl acetate 0.10 (≈13,100 ppm) + isoamyl
  isovalerate 0.10 (≈390 ppm) + hexanal 0.05 (≈13,100 ppm) + (E)-2-hexenal 0.05
  (≈5,900 ppm) + 1-hexanol 0.05 (≈1,300 ppm) → **green, strong, fast**.
- **Cinnamon** — cinnamaldehyde 0.65 @1.3 Pa (≈13 ppm, weak) + eugenol 0.20
  @2.7 Pa (≈27 ppm, weak) + linalool 0.05 (≈260 ppm) + limonene 0.05 (≈2,000
  ppm) → **yellow, weak, slow**.

**Human vs MOX asymmetry:** human thresholds are ppb-level; the MOX floor is
~1 ppm. A spice you smell clearly can still be `yellow` on the array. The
verdict grades the *instrument*, not the smell.

**Class verdicts** for the 14 `CLASS_TERMS` (alcohol, aldehyde, ketone, ester,
carboxylic acid, alkane, alkene, terpene, thiol, sulfide, amine, phenol,
aromatic, ether): always `yellow`, `low` confidence, `moderate`/`medium`, with
a note to resolve to a specific compound.

**Percept mapping** (`ontology.ts`): 14 percepts map chemistry to what the array
"reads as"; low-volatility percepts `{spicy-balsamic, smoky-phenolic}`. Nine MOX
boundaries: can (rough family, size ordering, volatility, redox); cannot (exact
structure/isomers, absolute ppm, non-redox gases, trace <1 ppm, mixture
decomposition).

### 7.6 Live resolution (PubChem) & provisional chemistry

- `lookupPubChem(query)` → PUG property endpoint; ~1.5 s cold; cached in
  localStorage `osmell-pubchem-cache` (cap 200, LRU); throttled ≥300 ms.
  Returns Kekulé SMILES (uppercase C with alternating `=`), unlike the curated
  lowercase aromatic `c1ccccc1`.
- `lookupPubChemBoilingPoint(query)` → resolves CID then walks the pug_view
  section tree; 8 s abort; returns first plausible °C in (−200, 600). Regex
  handles `"281.6±35.0 °C"` → 281.6. Verified: ethanol 78.2 °C, vanillin
  285 °C.
- `buildProvisionalChemical(enriched, bp)`: MW + BP from PubChem → `measured`;
  VP back-computed via Clausius–Clapeyron + Trouton → `estimated`; groups
  inferred from SMILES; `redoxActive` for reducing gases, `nonRedox` for
  inerts. UI shows amber "Provisional" banner; estimated properties are always
  surfaced honestly.
- Local tier-2 dictionary (`user-dictionary.ts`): localStorage
  `osmell-user-dictionary`, cap 200, dedupe by id, marked `my dictionary ·
  estimated` (−5 search penalty). Contribution requests queue in
  `osmell-contributions`.

### 7.7 Functional-group inference from SMILES (`groups.ts`)

A deliberately conservative structural heuristic (not a full SMILES parser).
`scanSmiles` handles bonds, branches, ring closures, bracket atoms; recognizes
both lowercase aromatic and Kekulé rings; Kekulé ring detection (6-ring all C
with 3 non-adjacent doubles; 5-ring 4 C + 1 hetero with 2 separated doubles);
**phenol vs methoxy** decided by connectivity (Ar–O–H = phenol, Ar–O–CH₃ =
ether), never by text pattern. Inferred groups: aromatic, sulfur, amine,
carboxylic acid, ester, ketone, diketone, aldehyde, phenol, alcohol, ether,
thiol, thioether, alkene, furan, alkane. Terpene is deliberately NOT inferable
from SMILES (keyword/odor matching only). 24 test fixtures prove group
inference.

### 7.8 Search (`lib/smellability/search.ts`)

Normalized query; field scoring: exact 100, starts-with (len ≥2) 75, contains
(len ≥2) 55, all tokens 65; CAS exact 95; SMILES exact 90; dictionary entries
−5. Threshold ≥ 40, top-8. Sources: 46 compounds, user dictionary, 27
composites, 14 classes. At zero candidates the UI offers "Not in the dictionary
— resolve live via PubChem".

### 7.9 Client-side storage keys

| Key | Purpose | Cap |
|---|---|---|
| `osmell-bench` | Pinned Smellability verdicts | 12, LRU |
| `osmell-user-dictionary` | Saved provisional chemicals | 200 |
| `osmell-pubchem-cache` | Enriched PubChem lookups | 200 |
| `osmell-contributions` | Curation request queue | unbounded |
| `opensmell-theme` | Theme (`dark` default) | — |

Sessions are in-memory only; refreshing clears the library.

### 7.10 Import, Compare, Train

- **Import:** `.csv`/`.txt`/`.osmell`, many at once, drag-and-drop. Sequential
  atomic per-file loop; results list with per-file quality badge and inline
  errors. Loose CSVs build a manifest (`formatVersion "1.0.0"`, `sensorType
  "mox"`, `unit:"adc"`, `adcBits 12`, `adcMax 4095`, rate from median gap,
  `role "exposure"`, label from filename, `baseline.source "auto"`,
  `r0Samples 15`). `.osmell` files parsed as-is and rescored.
- **Compare:** plots `(R − R0)/R0` per channel on shared relative-time axis
  (`t = i/10` s); R0 provenance noted; selection via Library checkboxes.
- **Train:** per-class gate — `MIN_PER_CLASS = 5`, `MIN_CLASSES = 2`,
  `ready ⟺ ≥2 distinct labels AND every label has ≥5 labeled exposures`.
  Rationale: with ~7 features/channel the vector is high-dimensional while a
  MOX response is dominated by few variance sources; <5 sessions per label
  would fit drift, not chemistry. The actual training pipeline (187-feature
  extraction, split by label, fit, cross-validated accuracy) is declared
  "coming in the next slice".

### 7.11 The honesty rules (evidence-driven)

Limits come from **measured lab failures** (`docs/smellability/calibration-lessons.md`):

1. **Affine calibration failed** on real cross-device data (47% → 33%); the
   engine never claims calibrated ppm; headspace ppm is a thermodynamic
   estimate.
2. **Pure-anchor calibration can't cover odorant space** — six pure compounds
   cover ~0.1% of 4,565 odorants (convex-hull analysis); ship a contribution
   loop, not a "calibrate to these bottles" flow.
3. **Session invariance comes from learning** — universal encoder 1D-CNN on
   held-out sessions: **81.78% accuracy / 80.33% macro-F1** vs pre-registered
   >70% (random baseline 2%). Latent space is session-invariant for *trained*
   substances; NOT zero-shot generalizable.
4. **Effective dimensionality ≪ sensor count** — two same-family MOX ≈1 dim,
   three ≈1.5–2, four ≈2–3; humidity is common-mode across SnO₂.
5. **Drift / batch ±20% / humidity set the capture rules** — guidance always
   prescribes clean-air baseline → exposure → recovery.
6. **Normalization finding** — z-scores beat Rs/R₀ for encoder input; paradigm
   features outperform statistical features cross-device; `.osmell` must
   preserve raw + baseline structure so any client picks its normalization.

**Is / is-not table:**

| The verdict **is** | The verdict **is not** |
|---|---|
| Physical feasibility (volatility × redox) | A calibrated concentration |
| A capacity grade within your labeled library | A guarantee of mixture decomposition |
| Honest uncertainty (low/medium when estimated) | A promise across unseen devices/sessions |
| Actionable capture guidance | A replacement for the recorder protocol |

### 7.12 Test map (`lib/smellability/__tests__/` — 8 files, 90 assertions)

Run with `npx vitest run` from `osmograph-web/`.

| File | Proves |
|---|---|
| `chain.test.ts` | Verdicts for curated substances (ethanol green/strong/fast; cinnamaldehyde yellow/weak/slow; N₂ red non-redox) |
| `transport.test.ts` | Antoine vs Clausius–Clapeyron consistency, diffusion/flux math |
| `ontology.test.ts` | Percept mapping, low-volatility handling, boundary relevance |
| `groups.test.ts` | 24-fixture functional-group inference incl. Kekulé vs aromatic, phenol-vs-methoxy |
| `enrichment.test.ts` | Boiling-point parsing incl. `±` uncertainty, pug_view section-tree extraction |
| `provisional.test.ts` | Estimated-flag honesty, inorganic non-redox vs reducing-gas redox, VP-from-BP |
| `user-dictionary.test.ts` | localStorage store: save/dedupe/remove/map/200-cap |
| `live-resolution.test.ts` | End-to-end PubChem → provisional chemical → verdict → save (mocked) |

Quality gates: `npm run build` passes; `npx tsc --noEmit` clean; `npm run lint`
at documented baseline (5 errors, 3 warnings — pre-existing: shadcn
set-state-in-effect ×3, `use-toast` actionTypes, `resolveAndRun` impure render,
`groups.ts` unused `d`).

### 7.13 Known gaps / next slices (honest)

- Quality scoring implementation diverges from spec §7 (6 vs 7 sub-scores;
  auto-R0 cap missing; median-gap continuity; duration uses N not N−1).
- Train pipeline not implemented (gate ships; 187-feature extraction + fit +
  cross-validated accuracy deferred).
- `decayTimeMs` always null; multi-exponential recovery fits not shipped.
- Composite profiles are representative subsets (coffee 200+ VOCs → 5), not
  full inventories.
- Verdicts are physical feasibility, never calibrated measurements.
- GitHub remote moved to `https://github.com/OpenSmell/osmograph-web.git`
  (capital O); `origin` still points at the lowercase URL.

---

## 8. Project Audit

Consolidated from `AUDIT_REPORT.md` (2026-06-19, automated, all 13 GitHub repos
+ local workspace) and `AUDIT_FINAL.md` (2026-06-16, all 19 local directories).

### 8.1 Repository inventory

**13 GitHub repos** under `github.com/opensmell`: Osmograph (desktop GUI, active),
electronic-nose (hardware, active), encoder (1D-CNN VAE, active), smell-pipeline
(chemoprint data pipeline, active), opensmell (SDK, active), Chemoprint
(29-dim descriptor, active), chemoprint-optimization (active),
affine-calibration-failed (negative result), session-invariance (81.78% proof),
Chemoprint-Apps, data-commons, Smellability (archived, ODT prediction),
OpenSmell-web (incomplete TypeScript frontend).

**Local-only:** OpenSmell-Legacy (GNN era), glyphchem, odor_prediction_models,
scent_search, SmellNet (fork), publishable.

### 8.2 Model inventory (highlights)

- **Encoder** (`opensmell/opensmell/weights/`): 1D-CNN VAE, 3 conv layers
  (6→64→128→256), latent 256, 255,872 params; losses MAE (1.0) + chemoprint MSE
  (0.5) + SupCon contrastive (0.3) + KL (0→0.1 over 50 epochs); AdamW lr 5e-4.
  - `encoder_v1.pth`: global z-score, mean pooling, **mean R²=0.882, median
    0.946** (documented).
  - `encoder_attn.pth`: attention pooling, mean R²=0.876.
  - `encoder_v2.pth` (Rs/R0, attention): **preferred by `load_auto()` but
    UNDOCUMENTED** — no training report.
  - `encoder_v3_best.pth` (per-rec z-score, attention): **UNDOCUMENTED**.
- **Session invariance:** MLP (12-dim) 74.84% / 72.28% F1; 1D-CNN (100×6)
  **81.78% / 80.33% F1**.
- **SmellNet:** ~85 `.pth` (MLP/CNN/LSTM/Transformer/Fusion/Translation/
  Contrastive/Text) + ~72 sweep checkpoints — largely undocumented.
- **Osmograph classifiers:** 13 sklearn `.pkl` (RF/LR, multi-class/binary).
- **Leffingwell predictor:** 77 odor categories from 29-dim chemoprint; test F1
  macro 0.301, micro 0.404.

### 8.3 Experiment results

| Experiment | Result | Status |
|---|---|---|
| Session invariance (1D-CNN) | 81.78% acc / 80.33% F1 (>70% prereg) | ✅ confirmed |
| Encoder v1 chemoprint R² | 0.882 mean / 0.946 median | ✅ confirmed |
| Encoder attention R² | 0.876 | ✅ confirmed |
| Leave-substance-out CV | **R² = −14.62** (fold 4: −55.71) — no novel-substance generalization | ✅ negative result |
| Paradigm features | 57–66% 50-class; cross-device 57% garlic/ginger | partial |
| Paradigm→taxonomy | intra 0.9612, inter 1.0110, ratio 0.9507 | weak but real |
| UCI drift stability | intra 0.066, inter 0.243, ratio 0.272 (36 months) | ✅ confirmed |
| Upward compatibility | best 29.1% (chance 25%) — NOT demonstrated | ✅ negative |
| ECI upward compatibility | 23.4% — does not help | ✅ negative |
| Adapter (MSE) | 0.95 cosine on held-out lemon | ✅ confirmed (simulation) |
| Adapter (cosine loss) | 0.81 — FAIL | ✅ negative |
| Parameterised adapter | 0.879 on held-out config D | ✅ confirmed |
| Conv1 fine-tune | garlic–ginger cos 0.409 | partial |
| Chemoprint validation | R²=0.982 (UCI), R²=0.877 (ODT) | ✅ confirmed |
| Convex hull | 6 reference compounds cover ~0.1% of 4,565 odorants | ✅ confirmed |
| Affine calibration | 47% → 33% (worse) | ✅ negative |
| Domain-adversarial encoder | domain classifier 100% — FAIL | ✅ negative |
| Full validation stress | channel-death −0.046 PASS; kinetic mismatch <0.001 PASS; 25% data ablation FAIL | mixed |

### 8.4 Datasets

- **SmellNet:** HF `DeweiFeng/smell-net`, arXiv 2506.00239; 50 food substances,
  6 MOX (NO2, C2H5OH, VOC, CO, Alcohol, LPG) + 6 environmental sensors; 250
  train + 50 test files, ~180,718 timesteps.
- **FooDB chemoprints:** 44 substances × 29 dims (`smell-pipeline/data/`).
- **UCI Gas Drift:** 6 gases, 16 MOX sensors, 10 batches over 36 months,
  ~13,910 samples, 128-dim precomputed features.
- **UCI Twin:** URL 404 — never downloaded (ideal cross-device validation
  unavailable).
- **Osmo taxonomy:** 50 SmellNet substances → 8 grand families (Woody 19,
  Fruity 10, Green 8, Herbal 7, Citrus 2, Soulful 2, Mineral 1, Floral 1).
- **User device:** garlic (3 sessions), ginger (2), cinnamon, banana, onion,
  room air, mosquito coil; 3-column raw ADC.
- **GC-MS (SmellNet):** elemental composition for 50 substances.

### 8.5 Paradigm extractor consistency (INCONSISTENT)

Three implementations disagree: Osmograph uses R0 = mean of first **3** samples
and a fixed 30-dim vector (5 features × 6 channels: delta_ratio, direction,
mean_slope, auc, endpoint_delta); the research extractor uses R0 = mean of
first **5** samples, adds rise_time + cross-channel selectivity ratios + global
features (variable length ~50–60 dims), dead threshold `std/R0 < 0.001`; the
**opensmell SDK implements NO paradigm features** (its legacy path used z-score
normalization). The current SDK's `process_mox` uses the web-parity R0
normalization (`baseline_for_channel` / `channel_stats`).

### 8.6 Quantified false/misleading claims

| # | Claim | Problem | Severity |
|---|---|---|---|
| 1 | "Four paradigms exhaust the set of transformations" | Unproven assertion | MEDIUM |
| 2 | "Near-perfect novel substance identification (cos 0.998)" | 1 of 5 pairings only | HIGH |
| 3 | "Commercial array can be calibrated to output chemoprint" | Validated on precomputed UCI features, not raw array data | HIGH |
| 4 | "Device-agnostic by construction" | Cross-device accuracy 57% for 2 substances | MEDIUM |
| 5 | "No calibration from the end user" | True only for 44 seen substances on the training device | LOW |
| 6 | "81.78% session invariance" | Holds out LAST 2 sessions — possibly order-confounded | LOW |
| 7 | "44 substances" vs SmellNet's 50 | 6 missing (not in FooDB) | LOW |
| 8 | "R²=0.892" | Actual 0.8817 | LOW |
| 9 | "Adapter: Universal Approximation Theorem guarantees" | Assumes continuous mapping; "sufficient data" undefined | MEDIUM |
| 10 | `research/full-validation/` | Referenced but does not exist | LOW |

### 8.7 Publishable results (ranked)

1. Session invariance 1D-CNN 81.78% / 80.33% — **ready**.
2. Chemoprint computational validation R²=0.982 (UCI) / 0.877 (ODT) — **ready**.
3. Leave-substance-out failure R²=−14.62 — **ready** (important negative).
4. UCI long-term stability intra/inter 0.272 over 36 months — **ready**.
5. Paradigm feature theory (physical grounding) — **ready** (theoretical).
6. Paradigm→taxonomy correlation 0.95 — **conditional**.
7. Cross-device paradigm transfer 57% — **not yet** (too few substances).
8. Smellability ODT prediction R²=0.575 — **conditional** (mediocre).

**Deal-breakers for a top venue:** device invariance never tested on real paired
hardware; cross-device validation only 2–4 substances; no held-out substance
generalization; honest framing = "session-invariance for 44 food substances on a
single device, with a theoretical framework for future cross-device work". Do
**not** lead with "interoperability" / "device-agnostic" / "universal standard".

### 8.8 Critical bugs & missing items

- `research/full-validation/` referenced but missing (MEDIUM).
- `encoder/tests/` empty (MEDIUM).
- `expand_channels()`: channel 5 always gets LPG training mean — 3-sensor
  devices lose information on that channel (MEDIUM, design limitation).
- `convex_hull.py` output not persisted — 0.1% claim not independently
  reproducible (LOW).
- `encoder_v2.pth` preferred by `load_auto()` but undocumented (MEDIUM).
- Taxonomy label ambiguity (almond → Cherry/Almondy vs commonly Woody/Nutty)
  (LOW).

### 8.9 Paper structure options (condensed)

- **A: Full system paper** — "OpenSmell: An Open Standard and SDK for Electronic
  Nose Interoperability" (interop problem → 4-paradigm theory → extraction →
  chemoprint → session-invariance → SmellNet dataset → taxonomy correlation →
  limitations → conclusion).
- **B: Paradigm feature paper** — "Paradigm Features: Device-Agnostic
  Representations for Metal-Oxide Gas Sensors" (narrower; experiments on
  invariance, cross-device, drift stability, perceptual correlation).
- **C: Negative results paper** — "Why Electronic Noses Don't Generalise"
  (session-level works R²=0.882; substance-level fails −14.62; recommend
  multi-device data + domain-adversarial training).

**Critical path to paper readiness (P0):** resolve paradigm extractor
inconsistency (canonical R0 method + feature set); implement paradigm features
in the SDK; document encoder_v2/v3; real paired hardware adapter test.

---

## 9. Changelog & Reproducibility

### 9.1 Milestone M3 (recent work)

Landed in commit `0876f2b` on `opensmell` master (18 files, +4501):

- **Bi-exponential default** in `compute_multi_exp_decay` (previously
  single-exp-first). The decay model is now: single-exp fallback attempted
  first, bi-exp default, tri-exp opt-in via `n_components=3` (and ≥30 recovery
  points). Result dict gained a `cost` key (residual sum of squares).
- **13 smellability modules** added to the SDK; `tests/test_smellability.py`
  added; `mox/__init__.py` exports smellability.
- **`spec/ALGORITHMS.md`** created (now folded into §5 of this document).
- **`test_determinism` rewritten** in `tests/test_legacy_api.py` to account for
  the MINPACK equal-cost limitation. Cell labels use `sorted(feature_names)`
  (matching `_feature_vector`'s sorted-key order; note `feature_names()`
  returns extraction order, so pre-existing name/value index mismatch from the
  first element exists and is documented in the test comment). Assertions:
  1. non-decay features `np.allclose`-identical between runs;
  2. differing-feature count ≤ 20;
  3. both-valid decay taus within 10% rtol.

**Empirical profile** (from the M3 investigation): pairwise runs show 0–20
differing cells (mean ≈5), always `*_decay_*` features; apparent `ch1_da_*`,
`ch2_da_*`, `ch5_hw_*` diffs were an artifact of sorting vs extraction-order
labeling, not real differences. Fit cost is stable (0.0 relative difference
across 286 fits).

### 9.2 Gate status (last verified)

- opensmell pytest: 113/113 pass.
- osmograph-web vitest: 102/102 pass; `tsc --noEmit`: 0 errors; eslint: 5
  pre-existing errors + 3 warnings (unrelated to M3).

### 9.3 Reproducibility guarantees (recap)

- Non-decay features are deterministic (byte-identical across runs).
- Decay fits may land on equal-cost MINPACK minima → taus can vary up to ~6%
  relative between runs; amplitudes may flip; residual `cost` is stable; rare
  `-1.0` "fit failed" sentinels on non-convergence.
- For publication-grade work: verify fits manually or fix the optimizer seed.

---

*OpenSmell — master reference consolidated 2026-08-03. Supersedes the folded-in
originals. Issues and pull requests welcome at `github.com/OpenSmell`.*
