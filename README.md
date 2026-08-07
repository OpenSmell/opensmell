# OpenSmell SDK

Extract the OpenSmell framework features from electronic nose recordings.

```python
import opensmell

# Extract 145 framework features from a CSV
features, names = opensmell.extract_features("recording.csv")

# Train a classifier on your own data
model = opensmell.train(features_matrix, labels)

# Predict a new recording
result = opensmell.predict("recording.csv", model)
print(result.substance, result.confidence)
```

## Quick start

```bash
pip install .
```

```python
import opensmell

# From CSV file — columns: NO2, C2H5OH, VOC, CO, Alcohol, LPG
features, names = opensmell.extract_features("cinnamon.csv")
# features: (N_windows, 145) numpy array
# names: list of 145 feature name strings

# Average features across all windows
avg_features = features.mean(axis=0)
```

### Training a custom classifier

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

## How it works

The SDK applies Rs/R₀ normalization (baseline correction) to raw sensor readings, then extracts 145 features across 5 taxonomy categories:

| Category | Features per channel | Purpose |
|----------|---------------------|---------|
| Device-Agnostic | 6 | Circuit-effect normalization (ΔR/R₀, rise time, decay time, AUC) |
| Absolute | 4 | Quantitative sensing (raw resistance, voltage) |
| Temporal | 4 | Fast dynamics (oscillation frequency, transients) |
| Health | 4 | Predictive maintenance (drift rate, noise floor) |
| Hardware | 3 | Device characterization (ADC noise, thermal profile) |

Plus 15 cross-channel selectivity ratios and 4 global features. Total: 145 dimensions.

This feature set is mathematically proven to cancel both Vcc and RL variations. It does not cancel sensor constants (a, b): zero-shot cross-device transfer is falsified in the interoperability experiments, and quantification requires per-rig reference-point calibration.

## Output

| Field | Type | Description |
|-------|------|-------------|
| `substance` | str or None | Predicted label (only from `predict`) |
| `confidence` | float or None | Prediction probability |
| `features` | np.ndarray | 145-dim average framework features |
| `feature_names` | list | 145 feature name strings |
| `n_windows` | int | Number of sliding windows processed |

## API

| Function | Description |
|----------|-------------|
| `extract_features(filepath)` | Extract (N_windows × 145) features from CSV |
| `process(filepath, model=None)` | Single-call: extract features, optionally predict |
| `train(X, y)` | Train a sklearn Pipeline (StandardScaler + RandomForest) |
| `predict(filepath, model)` | Shorthand for process with a model |
| `load_recording(filepath)` | Load and Rs/R₀-normalize a CSV |

## Dependencies

- Python 3.10+
- numpy, pandas, scikit-learn, scipy

## Related

- [interoperability](https://github.com/opensmell/interoperability) — Canonical experiments, cross-device bounds, and calibration
- [Chemoprint](https://github.com/opensmell/chemoprint) — 29-dim physicochemical descriptor from SMILES
