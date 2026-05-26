# OpenSmell: A Technical Account

**Version 1.0 — Current State, Boundaries, and Next Steps**

---

This document describes the OpenSmell project: what problem it addresses, what has been built, what has been tested, what has failed, and what remains to be done.

---

Humans have externalised sight and sound into reproducible systems. The chemical sense has not been externalised—not because hardware is absent, but because electronic noses share no common language.

## 1. The Problem

An e-nose sensor outputs raw physical measurements—typically resistance values from metal-oxide semiconductor (MOX) elements, or voltage changes from a divider circuit, or analog-to-digital converter counts. These raw values vary with:

- Sensor manufacturing tolerances (two sensors of the same model differ out of the box)
- Sensor age (baseline resistance drifts over months)
- Temperature and humidity (MOX conductivity is environmentally sensitive)
- Exposure history (prior VOCs alter surface chemistry)
- Circuit design (divider resistor values, supply voltage, ADC resolution)

Consequence: the same substance measured on two devices produces different raw readings. The same substance measured on the same device on different days produces different raw readings. There is no common scale, no common format, and no common language across the industry.

This fragmentation prevents:

- Sharing smell datasets across laboratories
- Training a classifier on one device and deploying it on another
- Building an application ecosystem where any app works with any hardware
- Creating a cumulative, community-owned dataset that improves over time

### 1.1 Why Not Perceptual Labels

An earlier approach trained a graph neural network (GNN) to predict human odor descriptors—categories like "floral," "musky," and "citrus"—directly from molecular structure using the Pyrfume dataset. The model overfit and failed to generalise to novel molecules.

Human odor descriptors are subjective and inconsistently applied across raters and contexts. The same molecule can be described differently by different people, or by the same person at different times. Training a model on these labels is analogous to training a weather model on opinions about whether it "feels cold." The target is too noisy.

Perception itself is not absolute for any sense. Brightness and loudness are perceived logarithmically, not linearly; colour categories shift with language and culture. This does not prevent us from building cameras and microphones that measure physical quantities—wavelength, intensity, frequency, amplitude—and leaving perceptual interpretation to application-layer software.

The same factoring applies here. The Chemoprint encodes measurable molecular properties. Perceptual labels, taxonomies, and classifiers can be built as heads on top of the latent space. Any representation—ECFP4, MACCS, a learned embedding, a custom taxonomy—can be plugged in. The latent space is the shared foundation. What sits above it is limited only by what developers choose to build.

---

## 2. Why Simple Calibration Fails

### 2.1 Affine Calibration

The most straightforward approach: assume two devices differ by a per-sensor gain and offset. Measure a set of known substances on both devices, compute the correction factors, and apply them to all future readings.

Tested on the UCI Gas Sensor Array Drift Dataset (16 MOX sensors, 6 pure gases, 36 months of drift data). Different time batches were treated as different "devices." An affine transform was learned from 60 calibration points and applied to held-out measurements.

Classification accuracy before calibration: 47%. After calibration: 33%.

The correction made performance worse. Sensor drift is not a linear gain-and-offset process. It involves non-linear changes in sensitivity profiles, cross-sensitivity patterns, and baseline resistance that interact with environmental conditions.

*This approach is archived at `opensmell/affine-calibration-failed`.*

### 2.2 Anchor Compound Interpolation

A more sophisticated approach: choose a small set of pure compounds as anchors. Expose a new device to these anchors. Learn a smooth mapping (e.g., Gaussian Process) from the device's raw readings to the known chemical properties of the anchors. Interpolate to predict properties for any other substance.

To test whether interpolation is valid, we computed the convex hull of six candidate anchors (methanol, hexane, acetone, acetic acid, toluene, isopropanol) in 29-dimensional chemical space. We then checked how many real-world odorants lie inside that hull.

Using 4,565 odorants from the GoodScents database: **0.1% lay inside the convex hull.** For 99.9% of real smells, the model would be extrapolating. Extrapolation in a 29-dimensional space from six data points is not approximately correct—it is meaningless.

The anchor approach is mathematically insufficient. Chemical space cannot be covered by a small set of pure compounds.

---

## 3. The Chemoprint

### 3.1 Definition

The Chemoprint is a 29-dimensional vector of physicochemical properties computable from a molecule's SMILES string using RDKit, an open-source cheminformatics library.

| Indices | Category | Properties |
|---------|----------|------------|
| 0–11 | Base properties | Molecular weight, heavy atom count, rotatable bonds, ring count, aromatic ring count, fraction Csp³, LogP, topological polar surface area, H-bond donors, H-bond acceptors, net charge, heteroatom count |
| 12–14 | Topological indices | Wiener index, Zagreb index (M1), eccentricity (graph diameter) — measures of molecular branching and shape |
| 15–28 | Functional group indicators | 14 binary flags: alcohol, aldehyde, ketone, carboxylic acid, ester, ether, primary amine, secondary amine, tertiary amine, nitro, thiol, sulfide, aromatic nitrogen, halogen |

Every dimension has a name and a physical interpretation. The vector is deterministic—the same SMILES always produces the same Chemoprint.

### 3.2 Validation on Pure Compounds

Using the UCI Gas Sensor Array Drift Dataset (16 sensors, 6 pure gases, 36 months), a Random Forest regressor was trained to map raw 128-dimensional sensor readings to the 29-dimensional Chemoprint.

On a held-out test set (20% of samples): **R² (variance-weighted) = 0.982.** All 29 dimensions achieved R² > 0.97.

A commercial sensor array can be calibrated to output a fixed, interpretable chemical vector for pure compounds with near-perfect accuracy. The Chemoprint is measurable, not merely theoretical.

**Caveat.** The UCI dataset contains only six chemically distinct gases measured on a single sensor configuration over 36 months. A high R² on this dataset demonstrates measurability, not universal applicability. Generalisation to structurally diverse mixtures, different sensor types, or real-world conditions requires broader validation.

### 3.3 Benchmark Against Industry Fingerprints

| Fingerprint | Dimensions | QSAR R² |
|-------------|-----------|---------|
| ECFP4 | 1,024 | 0.427 |
| MACCS | 167 | 0.497 |
| Chemoprint | 29 | 0.403 |

The Chemoprint achieves competitive predictive power at 5–35× compression. It is slightly weaker than MACCS on raw QSAR performance, but every dimension is interpretable—you can inspect dimension 3 and identify it as the number of hydrogen bond donors. You cannot do this with MACCS bit #783.

For an open standard that must be understood, trusted, and extended by multiple parties, interpretability is a practical requirement.

### 3.4 Limitations of the Current Chemoprint

The 29 structural dimensions do not capture:

- **Chirality.** Mirror-image molecules (R- vs S-carvone) can produce different odor percepts (spearmint vs caraway). RDKit can compute a chirality index; this dimension is planned for v1.1.
- **Vibrational modes.** If the vibrational theory of olfaction is correct, infrared absorption frequencies contribute to odor quality. Computing these requires quantum chemistry (DFT calculations); planned for v2.0 when a reference dataset is available.
- **Isotope effects.** Deuterated compounds are detected as different by some organisms. Planned as a mass-shift index.
- **Conformational flexibility.** Molecules that flex into multiple shapes may activate multiple receptors. Planned as a weighted rotatable-bond index.
- **Mixture interaction terms.** Two molecules together can produce a percept that is not the sum of their individual profiles (synergy, suppression, masking). Planned for v3.0 when sufficient mixture data exists.

Each addition is backward-compatible. Software written for v1.0 ignores extra dimensions. The version number increments. Old applications continue to function.

---

## 4. The Learned Latent Space

### 4.1 Rationale

The calibration failures share a cause: they assume a mathematical form for the mapping between devices or sessions. The affine approach assumed linearity. The anchor approach assumed that a smooth interpolation from six points covers a 29-dimensional space. Both were wrong.

The alternative: learn the mapping from data. Train a neural network on raw sensor readings from multiple measurement sessions. The network's internal representation—the layer before the output—becomes a latent vector that is invariant to session-specific variation because the network must produce the same output regardless of when the measurement was taken.

### 4.2 Session-Invariance Proof

Tested on the SmellNet dataset (Dewei Feng et al., MIT Media Lab): 50 food substances, 6 MOX sensors (NO₂, C₂H₅OH, VOC, CO, Alcohol, LPG), approximately 828,000 time-series readings across 300 recordings, multiple measurement sessions per substance on different days.

A 1D convolutional neural network (1D-CNN) was trained on raw 100-time-step × 6-sensor segments. The evaluation used held-out measurement sessions—entire recording days the model had never seen. The success criterion was pre-registered at >70% accuracy (random baseline: 2%).

| Model | Test Accuracy | Macro F1 |
|-------|--------------|----------|
| MLP (12-dim mean/std features) | 74.84% | 72.28% |
| 1D-CNN (raw 100×6 time series) | 81.78% | 80.33% |

Both models exceeded the threshold. t-SNE visualizations show clear clustering by substance across different measurement days. The latent space is session-invariant for substances in the training set.

*This proof is published at `opensmell/session-invariance`. The code is self-contained and reproducible.*

---

## 5. The Universal Encoder

### 5.1 Architecture

The universal encoder is a neural network that maps raw 6-sensor time-series data (100 time steps × 6 sensors) to a 256-dimensional latent vector. It is trained with four simultaneous objectives:

1. **Reconstruction loss (MAE):** The encoder must reconstruct the original sensor signal from its latent representation. This forces the latent space to preserve all information the sensors capture, without requiring any labels.

2. **Chemoprint prediction loss:** A head network maps the latent vector to the 29-dimensional Chemoprint. The encoder is penalized for prediction error. This forces the latent space to encode molecular properties.

3. **Contrastive loss:** Recordings of the same substance from different sessions are pulled together in latent space; recordings of different substances are pushed apart. This enforces session-invariance directly.

4. **KL divergence loss:** The latent space is constrained to be smooth and continuous (VAE-style reparameterization with μ and σ). This enables interpolation between similar substances.

The encoder is a 1D-CNN with three convolutional layers (6→64→128→256), max pooling, and adaptive average pooling. Training uses AdamW with learning rate scheduling and early stopping (patience=20 epochs on validation loss).

### 5.2 Training Data

Ground-truth Chemoprints were required for the 50 SmellNet substances. SmellNet's GC-MS data files contain pre-computed feature vectors and elemental compositions, not compound names or SMILES strings. The FooDB REST API was non-functional.

Solution: The FooDB database dump (87 MB zip, containing a 3.5 GB Content.json file) was processed locally. The extraction pipeline:

1. Match 44 of 50 SmellNet substance names to FooDB food IDs via a manually curated mapping.
2. Stream Content.json from inside the zip (never written to disk).
3. Map each food to its associated volatile compounds (~7,900 unique compounds across all foods).
4. Retrieve SMILES strings from Compound.json.
5. Compute the 29-dim Chemoprint for each compound.
6. Average per food (equal weight—FooDB lacks concentration data).

Six substances were absent from FooDB (chamomile, chestnuts, peanuts, pecans, pistachios, walnuts). Forty-four had usable Chemoprints.

### 5.3 Results

The encoder was trained on 1,642 segments from 250 SmellNet training recordings, validated on 205 segments, and tested on 926 segments from held-out measurement sessions—entire recording days the model never saw.

**Held-out sessions of known substances:**
- Mean Chemoprint R² = **0.882** across all 29 dimensions
- Median R² = 0.946
- Success criterion (R² > 0.7): **met**

The encoder predicts molecular properties from raw sensor data with high fidelity when the substance was present in the training set, on measurement days it has never seen. No calibration is required. The latent space is session-invariant.

**Held-out substances (leave-11-substances-out cross-validation):**
- Mean Chemoprint R² = **-14.62**
- Success criterion (R² > 0.7): **not met**

The encoder does not generalize to substances it has never seen during training. Forty-four foods with six broad-spectrum MOX sensors is insufficient to learn the full mapping from sensor space to molecular property space. The encoder functions as a substance identification system with session invariance; it is not a universal chemistry predictor.

| Claim | Pre-registered threshold | Actual result | Status |
|-------|--------------------------|---------------|--------|
| Session-invariance (classification) | Accuracy > 70% | 81.78% | Confirmed |
| Session-invariance (chemoprint) | R² > 0.7 | 0.882 | Confirmed |
| Substance generalization | R² > 0.7 | -14.62 | Not confirmed |

### 5.4 Weak Dimensions

Chemoprint dimensions 16, 24, and 25 (functional group indicators for secondary amine, halogen, and nitro) achieved R² values of 0.44, 0.00, and 0.91 respectively—substantially lower than the other dimensions. Some binary flags appear rarely or never in the training set (dim 24 — halogen — R²=0.0). Six broad-spectrum MOX sensors lack the resolution to distinguish closely related functional groups at the concentrations present in complex food mixtures.

The encoder predicts continuous properties (molecular weight, LogP) more accurately than discrete structural features. This is consistent with the physics: MOX sensors respond continuously to VOC concentration, not selectively to specific functional groups. Higher-resolution sensors or larger training sets would be required to improve these dimensions.

---

## 6. The Two-Layer Architecture

The design separates the system into two layers with different stability guarantees.

**Layer 1: The Universal Encoder (frozen once trained)**

A function mapping raw sensor data to a 256-dimensional latent vector. Trained with the four objectives described above. Once trained, the encoder is frozen. It does not change when new applications are built or new scientific discoveries are made.

The encoder makes no theoretical commitments about olfaction. If vibrational theory is correct, the encoder preserves vibrational information because sensors respond to molecular vibrations and the MAE loss preserves whatever information is in the signal. If chirality matters, the encoder preserves chirality because chiral molecules produce different sensor responses. The encoder is a faithful but theory-agnostic compression of the sensor data.

**Layer 2: Heads (swappable, versioned)**

Heads are small models trained on the frozen latent space to predict specific outputs. Any representation can serve as a head—the Chemoprint, ECFP4, MACCS, a learned embedding, a perceptual taxonomy, a food spoilage classifier, or a breath analysis model. The latent space is the shared foundation; what sits above it is limited only by what developers choose to build.

The Chemoprint head v1.0 predicts 29 structural dimensions. Future versions add dimensions:

| Version | Dimensions | Additions |
|---------|-----------|-----------|
| v1.0 | 29 | Structural properties |
| v1.1 | 32 | Chirality index |
| v2.0 | 40 | Vibrational modes |
| v3.0 | TBD | Mixture interaction coefficients |

Applications written for v1.0 ignore extra dimensions. New applications use new heads. Applications that bypass heads entirely—using the latent space directly for classification or similarity search—are unaffected by head versioning.

The encoder version increments when retrained on significantly larger or more diverse datasets. Old encoder versions remain available for reproducibility.

---

## 7. The SDK

The OpenSmell SDK is a pip-installable Python package bundling the trained encoder, normalization statistics, and prototype latent vectors for 44 substances.

```python
pip install opensmell

import opensmell

# From a file
result = opensmell.process("recording.csv")
# result.substance     → "cinnamon"
# result.confidence    → 0.994  (cosine similarity to nearest prototype)
# result.chemoprint    → 29-dim numpy array
# result.latent        → 256-dim numpy array

# From a live sensor buffer
result = opensmell.process_array(buffer)  # shape (100, 6)

# Novel substance warning
if result.confidence < 0.7:
    print(result.warning)
    # "This substance may not be in the training set.
    #  Consider contributing your recording."
```

The SDK requires no calibration, no downloads, and no hardware-specific configuration for substances in the training set. Confidence below 0.7 triggers an automatic suggestion to contribute data to the community dataset.

The SDK is a research preview. It has not been tested outside the SmellNet training distribution.

---

## 8. The Data Commons

The OpenSmell Data Commons defines a standard format for community e-nose data contributions.

**FORMAT.md** specifies:
- CSV schema: `{substance}_{device_id}_{session_date}.csv`
- Required columns: `timestamp` (optional), `sensor_1` through `sensor_N` (any number)
- Sensor metadata (in accompanying JSON): unit (resistance_ohm, voltage, rs_r0, adc), sensor model, circuit description
- Optional: temperature_celsius, humidity_percent

An upload notebook validates submissions and pushes them to `opensmell/community` on HuggingFace. The dataset is versioned, public, and loadable via:

```python
from datasets import load_dataset
data = load_dataset("opensmell/community", split="train")
```

The pipeline is a placeholder. If you want to contribute data, join the Discord or open an issue—we’ll get it working together.

---

## 9. Repository Structure

| Repository | Contents |
|-----------|----------|
| `opensmell` | SDK. `pip install opensmell`. |
| `universal-encoder` | Training code, model weights, validation reports. |
| `session-invariance` | Classification proof on SmellNet. Reproducible. |
| `chemoprint` | 29-dim molecular fingerprint library. Pip-installable. |
| `chemoprint-apps` | 15+ tools: similarity search, odor prediction, QSAR. |
| `data-commons` | Community data pipeline and format specification. |
| `affine-calibration-failed` | Archived record: why affine calibration fails. |

All repositories are MIT-licensed. All experiments are reproducible on a standard laptop without a GPU.

---

## 10. What Remains to Be Done

**Substance generalization.** The current encoder does not generalize to substances absent from the training set. This is a dataset size limitation: 44 foods is insufficient. The Data Commons pipeline is designed to receive contributions that will increase this number. Hundreds of training substances are needed for robust generalization. The architecture supports this—only more data is required.

**Device-invariance.** All training data comes from a single sensor board (SmellNet's FS-12). Cross-device generalization has not been demonstrated. Two approaches are being explored:

- **Zero-shot (no adapter):** Feed data from a different sensor configuration
  directly through the encoder and check whether the latent space generalises.
  Preliminary simulation results (see `electronic-nose/test_zero_shot.py`)
  indicate this does **not** work — the encoder was trained exclusively on 6-
  channel SmellNet data and cannot generalise to even a simulated 4-channel
  configuration without a mapping step.

- **Per-device adapter:** A lightweight MLP trained on paired recordings from
  two devices. Simulation results (see `electronic-nose/train_adapter.py` and
  `electronic-nose/test_adapter.py`) are pending. If the adapter works, each
  new hardware configuration would require a small set of calibration
  recordings (~5 substances, ~60 seconds each).

- **Domain-adversarial encoder:** Retrain the encoder on multi-device data with
  a domain-adversarial head that forces device-invariance natively. This
  requires data from multiple hardware platforms and is the long-term solution.
  Multi-device data is the prerequisite.

**Federated learning.** A global model can be improved without centralizing raw sensor data. Each participant downloads the model, trains locally on their own recordings, and sends back only model updates (gradients). The server aggregates updates. No raw data leaves the participant's device. The federated learning server is planned but not yet implemented.

**Reference hardware.** An open-source e-nose design (ESP32 + 4–6 MOX sensors) is needed as a reference implementation and an entry point for contributors who do not own commercial hardware. Bill of materials, PCB files, and firmware are planned but not yet released.

**Independent reproduction.** All results reported here are self-reported. The code and data are public. Independent reproduction by external researchers is invited and necessary.

---

## 11. How to Contribute

**If you have an electronic nose:** Follow the Data Commons format specification. Record 20 labeled substances. Upload via the provided notebook. Your device becomes part of the training set for the next encoder version.

**If you are a researcher:** Reproduce the results. File issues if results differ. Propose alternative architectures. The adversarial process is the mechanism of improvement.

**If you are an engineer:** The multi-device autoencoder, federated learning server, and reference hardware design are open engineering problems with clear specifications.

**If you are a domain expert:** Audit the Chemoprint descriptor set. What molecular properties relevant to olfaction are missing? What volatile compounds are reliable markers for the applications you work on (food safety, clinical diagnostics, environmental monitoring)?

All repositories are at `https://github.com/opensmell`. The Discord is at `https://discord.gg/CGER3tHxbH`.

---

*This document will be updated as the project evolves. The current version reflects the state of the project as of the date of writing. All claims are bounded by the evidence described herein. Errors, omissions, and failures of reproduction should be reported as GitHub issues.*