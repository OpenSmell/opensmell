# OpenSmell SDK

A session‑invariant encoder for 6‑sensor MOX electronic noses. One function call turns raw sensor data into a latent vector and chemical fingerprint.

```python
import opensmell

result = opensmell.process("my_recording.csv")
print(result.substance)    # "cinnamon"
print(result.confidence)   # 0.99
print(result.chemoprint)   # 29-dim array
```

## Status

**v1.0 — What it does:** Maps 6-sensor e-nose data (SmellNet sensor configuration) to a 256-dim latent space and a 29-dim chemoprint. Trained on 44 food substances (of 50 in SmellNet; 6 lacking FooDB data). Session-invariant (same food, different days → same output). Proven R² = 0.882 on held-out sessions of known substances.

**What it does NOT do:**
- **Novel substances.** The encoder cannot generalise to foods it hasn't seen. Leave-substance-out validation gives negative R². It identifies substances it trained on, not unknown ones.
- **Cross-device.** All data from one sensor board. Device-invariance requires multi-device data (not yet collected).
- **Environmental or industrial samples.** Trained on 44 foods only.
- **Functional group granularity.** The 29 chemoprint dimensions are limited to basic structural properties. Rare functional groups (present in <20% of foods) have poor reconstruction (dim 16 R² = 0.43).

**Independent reproduction:** Clone the encoder repo. Run `src/train_encoder.py`. Our reported R² = 0.882 should reproduce within ±0.05. If not, file an issue.

## Quick start

```bash
pip install .
```

```python
import opensmell

# From CSV file — 100+ rows with columns NO2, C2H5OH, VOC, CO, Alcohol, LPG
result = opensmell.process("cinnamon_6.csv")

# From live sensor array (100 time steps × 6 sensors)
import numpy as np
live_data = np.random.randn(100, 6)  # replace with real readings
result = opensmell.process_array(live_data)

# Check confidence — warning if < 0.7 threshold
if result.warning:
    print(result.warning)
    opensmell.export_for_contribution("my_recording.csv", result, output_dir="./contrib/")
```

## Output

| Field | Type | Description |
|-------|------|-------------|
| `substance` | str | Predicted substance name (nearest prototype) |
| `confidence` | float | Cosine similarity to nearest prototype (0–1) |
| `warning` | str or None | Warning if confidence < 0.7 (OOD detection) |
| `should_contribute` | bool | True if confidence < 0.7 |
| `chemoprint` | np.ndarray | 29-dim physicochemical descriptor |
| `latent` | np.ndarray | 256-dim latent vector |
| `contribution_url` | str | URL for data contribution |

## How confidence works

The SDK stores 44 prototype latent vectors (mean latent per training substance). At inference, it computes cosine similarity between the new latent and all prototypes. The highest similarity is the predicted substance; that similarity is the confidence.

Known in-distribution substances score > 0.99. Extreme OOD signals score ~0.69. The 0.7 threshold is conservative — tune it per application.

## Related repos

| Repo | Role |
|------|------|
| [encoder](https://github.com/opensmell/encoder) | Training code, model checkpoints, limitations |
| [chemoprint](https://github.com/opensmell/chemoprint) | Ground-truth 29-dim physicochemical descriptor |
| [data-commons](https://github.com/opensmell/data-commons) | Standard format for contributed e-nose datasets |
| [session-invariance](https://github.com/opensmell/session-invariance) | Proof that latent spaces can be session-invariant |
