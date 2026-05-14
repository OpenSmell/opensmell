# OpenSmell SDK

A universal translator for electronic noses. One function call turns raw sensor data into a chemical fingerprint.

```python
import opensmell

result = opensmell.process("my_recording.csv")
print(result.substance)    # "cinnamon"
print(result.confidence)   # 0.94
print(result.chemoprint)   # 29-dim array
```

## Status

**v1.0** — Encoder proven with mean chemoprint R² = 0.892 on held-out substances. Session-invariant. Device-invariance in development.

## Quick start

```bash
pip install .
```

```python
import opensmell

# From CSV file
result = opensmell.process("cinnamon_6.csv")

# From live sensor array (100 time steps x 6 sensors)
import numpy as np
live_data = np.random.randn(100, 6)  # replace with real readings
result = opensmell.process_array(live_data)

# Check confidence
if result.warning:
    print(result.warning)
    opensmell.export_for_contribution("my_recording.csv", result, output_dir="./contrib/")
```

## Output

| Field | Type | Description |
|-------|------|-------------|
| `substance` | str | Predicted substance name |
| `confidence` | float | Cosine similarity to nearest prototype (0–1) |
| `warning` | str or None | Warning if confidence < 0.7 |
| `should_contribute` | bool | True if confidence < 0.5 |
| `chemoprint` | np.ndarray | 29-dim physicochemical descriptor |
| `latent` | np.ndarray | 256-dim latent vector |
| `contribution_url` | str | URL for data contribution |

## Related repos

| Repo | Role |
|------|------|
| [universal-encoder](https://github.com/opensmell/universal-encoder) | Training code and model checkpoints |
| [chemoprint](https://github.com/opensmell/chemoprint) | Ground-truth 29-dim physicochemical descriptor |
| [data-commons](https://github.com/opensmell/data-commons) | Standard format for contributed e-nose datasets |
| [session-invariance](https://github.com/opensmell/session-invariance) | Proof of session-invariant latent spaces |
