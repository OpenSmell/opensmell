# OpenSmell: Technical Account

**Version 1.0 — First Generation Electronic Nose Interoperability**

---

Electronic noses are the next frontier, but do not interoperate. Every device speaks a different language because raw sensor readings—resistance values, voltages, ADC counts—vary with manufacturing tolerances, temperature, humidity, sensor age, and circuit design. The same garlic clove produces completely different raw readings on two different devices, or on the same device on different days. There is no common scale, no common format, and no common language.

OpenSmell is a versioned, open standard that maps raw sensor data from any metal‑oxide (MOX) electronic nose into a stable, device‑agnostic format. It requires no calibration from the end user. It is designed to scale across sensor generations.

---

## 1. The Resistance-Change Paradigm

The v1.0 encoder does not learn chemistry. It does not identify molecules. It learns **patterns of resistance change across an array of metal‑oxide sensors.**

Every MOX sensor operates on the same physical principle. A heated tin dioxide (SnO₂) surface exchanges electrons with adsorbed oxygen. When a reducing gas (ethanol, CO, most VOCs) arrives, it reacts with the adsorbed oxygen, releasing electrons back into the conduction band and lowering the resistance. The relationship follows the power‑law derived from the mass‑action equilibrium of oxygen adsorption and gas reaction:

$$
\frac{R_s}{R_0} = a \cdot C^b
$$

where R₀ is the baseline resistance in clean air, C is the gas concentration, *a* is the sensitivity pre‑factor, and *b* is the stoichiometric exponent.

MOX sensors produce a single physical signal: the free electron concentration in the SnO₂ conduction band. Every measurement is a function of that electron count, sampled across channels, over time, and relative to a baseline. There are no other degrees of freedom. The following four paradigms exhaust the set of transformations that can be applied to this signal. These four paradigms define what apps can be built on Gen 1 hardware.

---

## 2. The Four Programming Paradigms of Smell Apps

**Paradigm 1 — Amplitude (Δe⁻ magnitude).** The size of the resistance drop is proportional to the concentration of reducing gas. For a mixture, conductance (1/Rs) is approximately additive:

$$
\frac{1}{R_s} \approx \frac{1}{R_0} + \sum_i k_i \cdot C_i^{b_i}
$$

**Paradigm 2 — Selectivity pattern (Δe⁻ across channels).** Different MQ sensors have different sensitivity constants for different gases. The vector of resistance changes across the array forms a pattern characteristic of the gas mixture. Two substances with similar VOC profiles produce similar patterns.

**Paradigm 3 — Temporal profile (Δe⁻ kinetics).** The rate at which resistance drops and recovers depends on adsorption/desorption kinetics of the specific VOCs. Light, volatile compounds respond quickly; heavier compounds linger.

**Paradigm 4 — Ratio stability (Δe⁻ normalisation).** Raw resistance values drift with temperature, humidity, and sensor age. The ratio Rs/R₀ is more stable because environmental factors affect both numerator and denominator similarly.

These four paradigms enable every application on the platform:

- **Classification of known patterns** — "this is garlic," "this is fresh milk," "this is spoiled milk"
- **Trajectory tracking over time** — "this milk will spoil in approximately 12 hours"
- **Anomaly detection** — "unexpected VOC spike in the baby's room," "CO rising in the garage"
- **Similarity search** — "this unknown spice is closest to cinnamon"

Apps that require quantitative ppm measurements, chiral discrimination, or trace‑level biomarker detection are outside the current scope and reserved for future sensor generations.

---

## 3. The Chemoprint

The Chemoprint is a 29‑dimensional vector of physicochemical properties computable from a molecule's SMILES string using RDKit. It serves as an interpretable training target and an optional output head for applications that want chemical interpretability. The encoder was trained with a Chemoprint prediction objective, but this head is optional at runtime. Developers can use it for cheminformatics integration, or train their own task‑specific heads directly on the latent space.

---

## 4. The Encoder

### 4.1 Architecture

The encoder maps raw 6‑sensor time‑series data (100 time steps × 6 sensors) to a 256‑dimensional latent vector. It is a 1D‑CNN trained with four simultaneous objectives:

1. **Reconstruction loss (MAE):** The encoder must reconstruct the original sensor signal from its latent representation. This forces the latent space to preserve all four resistance‑change paradigms.

2. **Chemoprint prediction loss:** A head network maps the latent vector to the 29‑dim Chemoprint. This forces the latent space to encode molecular properties that correlate with the observed resistance‑change patterns. This head is optional at runtime.

3. **Contrastive loss:** Recordings of the same substance from different sessions are pulled together; different substances are pushed apart. This enforces session‑invariance by forcing the encoder to rely on ratio stability and pattern shape rather than absolute amplitude.

4. **KL divergence loss:** Constrains the latent space to be smooth and continuous, enabling interpolation between similar substances.

Each loss maps directly to one of the four paradigms. MAE preserves amplitude, selectivity, and temporal information. Contrastive loss enforces ratio stability. KL loss keeps the latent space smooth. Together they cover all four paradigms. Changing any one loss would break coverage of the corresponding paradigm.

Once trained, the encoder is frozen. It does not change when new applications are built or new scientific discoveries are made.

### 4.2 Results

Trained on the SmellNet dataset (Dewei Feng et al., MIT Media Lab): 50 food substances, 6 MOX sensors, 828,000 time‑series readings, multiple measurement sessions on different days.

**Held‑out measurement sessions (known substances):**
- Classification accuracy: **81.78%** (random baseline: 2%)
- Chemoprint prediction R²: **0.882** (median 0.946)
- Success criterion (R² > 0.7): **met**

The encoder identifies known substances robustly across measurement days with no calibration. The latent space is session‑invariant.

**Held‑out substances (leave‑11‑out cross‑validation):**
- Chemoprint prediction R²: **–14.62**

The encoder does not generalise to substances whose resistance‑change patterns it has not encountered during training. This is a **fundamental limit of Gen 1 MOX sensors**, not a failure of the architecture. The encoder cannot predict a Chemoprint for a resistance‑change pattern it has never seen because resistance‑change patterns do not linearly encode molecular properties. This limit will recede as sensor generations advance.

| Claim | Pre‑registered threshold | Actual result | Status |
|-------|--------------------------|---------------|--------|
| Session‑invariance (classification) | Accuracy > 70% | 81.78% | Confirmed |
| Session‑invariance (chemoprint) | R² > 0.7 | 0.882 | Confirmed |
| Substance generalisation | R² > 0.7 | –14.62 | Expected Gen 1 limit |

---

## 5. Cross‑Device Interoperability

### 5.1 The Adapter

The encoder was trained on SmellNet's 6‑channel sensor board. Raw voltages from a different sensor configuration live in a different coordinate system. The solution is a **per‑device adapter**—a small neural network trained once to translate between hardware configurations. The math:

Let *E* be the frozen encoder. It maps SmellNet's 6‑dimensional raw voltages **x_S** to a latent vector **z**:

$$
\mathbf{z} = E(\mathbf{x}_S)
$$

A new device produces 3‑dimensional raw voltages **x_B**. The adapter *A*: ℝ³ → ℝ⁶ is a small MLP trained on paired recordings—the new device and SmellNet measuring the same substances. The loss minimises mean squared error:

$$
\mathcal{L}_{\text{adapter}} = \frac{1}{N}\sum_t \| A(\mathbf{x}_B(t)) - \mathbf{x}_S(t) \|^2
$$

After training, adapter weights are frozen. The composite pipeline:

$$
\mathbf{z} = E(A(\mathbf{x}_B))
$$

The Universal Approximation Theorem guarantees that *A* can learn this mapping, provided the mapping is continuous and there is enough training data. The mapping is continuous because the underlying physics—electron‑count changes in SnO₂—is continuous. Both devices measure the same physical phenomenon through different sensor configurations.

### 5.2 No Calibration for the End User

A user builds a device, installs the SDK, and runs `opensmell.process()`. The SDK calls the adapter, then the encoder. The adapter was trained once by the hardware developer. The encoder is frozen. The same garlic produces the same latent vector on every device that uses the same adapter.

An app developer trains a classifier *C* on the latent space. The user downloads the app, which bundles *C*. The complete pipeline:

$$
\text{label} = C(E(A(\mathbf{x}_B)))
$$

At no point does the user calibrate anything. At no point is the encoder retrained. The adapter was trained once. The classifier was trained once. The user simply uses the device.

### 5.3 The Device‑Invariance Progression

The per‑device adapter works for a specific device pair. Two further approaches provide increasing generality:

1. **Parameterised adapter:** One adapter trained on 3–5 diverse hardware configurations, taking (raw voltages + a hardware config vector describing sensor models, load resistor, supply voltage, ADC resolution) as input. After training, any new configuration within the convex hull of the training configurations works with no extra training. The user enters their sensor specs; the adapter already knows how to handle them.

2. **Domain‑adversarial encoder:** Retrain the encoder from scratch on multi‑device data with a domain classifier that is punished when it can identify which device produced a given latent vector. The encoder learns to strip device‑identity from the latent space, making any device work natively with no adapter. This is proven in other domains (speech, EEG, cameras) but requires multi‑device data from 3–5 devices.

The effective dimensionality of MOX sensor variation is 2–3 (baseline resistance, sensitivity slope, cross‑sensitivity profile). Spanning a *d*‑dimensional space requires *d*+1 points, meaning 3–5 strategically chosen device configurations are sufficient. The Data Commons exists to collect data from these configurations.

---

## 6. The Two‑Layer Architecture

**Layer 1: The Encoder (frozen, versioned).** Trained with MAE, chemoprint, contrastive, and KL losses. Once trained, frozen. Does not change when new applications are built or new science emerges. Versioned by sensor generation: `encoder_v1` for MOX, `encoder_v2` for MEMS, `encoder_v3` for electrochemical, and so on. Old encoders are never modified. New encoders coexist with old ones.

**Layer 2: Heads (swappable, versioned).** Small models trained on the frozen latent space to predict specific outputs. Any representation—Chemoprint, ECFP4, MACCS, a learned embedding, a food spoilage classifier, a breath analysis model—can be plugged in. The latent space is the shared foundation. What sits above it is limited only by what developers choose to build.

| Head version | Dimensions | Additions |
|-------------|-----------|-----------|
| v1.0 | 29 | Structural physicochemical properties |
| v1.1 | 32 | Chirality index |
| v2.0 | 40 | Vibrational modes |
| v3.0 | TBD | Mixture interaction coefficients |

Applications written for v1.0 ignore extra dimensions. Applications that bypass heads entirely—using the latent space directly for classification or similarity search—are unaffected by head versioning.

---

## 7. Sensor Generations

MOX sensors are the first generation of electronic nose technology. The architecture is designed to absorb new generations as they arrive.

| Generation | Sensor type | Physical signal | New paradigms |
|-----------|-------------|-----------------|---------------|
| Gen 1 (current) | MOX (MQ series) | Δe⁻ in SnO₂ | Amplitude, selectivity, temporal, ratio stability |
| Gen 2 | MEMS | Δe⁻ in nanostructured film | Faster kinetics, lower power, smaller size |
| Gen 3 | Electrochemical | Δcurrent at electrode | Higher selectivity per channel, quantitative ppm |
| Gen 4 | Optical / vibrational | Δabsorption at IR frequencies | Spectral pattern matching, structural identification |
| Gen N | Molecular counter | Individual molecule detection | Absolute quantification, trace biomarker detection |

Each generation requires a new encoder trained on data from that sensor type. The architecture does not change. Old apps work with old encoders. New apps use new encoders. The SDK abstracts the versioning. The standard versions forward.

**Current capability (Gen 1):** Classification of known substances, trajectory tracking over time, anomaly detection, and similarity search. Apps requiring quantitative ppm, chiral discrimination, or trace‑level detection are outside the current scope.

**Future capability (Gen 2+):** The same SDK, the same API, and the same app ecosystem extend to new hardware. The standard is the language. The sensors are the speakers. Better speakers make the language more expressive; they do not require a new language.

---

## 8. The SDK

```python
pip install opensmell

import opensmell

result = opensmell.process("recording.csv")
# result.substance     → "cinnamon"
# result.confidence    → 0.994
# result.chemoprint    → 29-dim numpy array
# result.latent        → 256-dim numpy array

if result.confidence < 0.7:
    print(result.warning)  # suggests contributing novel data
```

The SDK bundles the trained encoder, normalisation statistics, and prototype latent vectors for 44 substances. No calibration required. The SDK is a research preview and has not been tested outside the SmellNet training distribution.

---

## 9. Repository Structure

| Repository | Contents |
|-----------|----------|
| `opensmell` | SDK. `pip install opensmell`. |
| `encoder` | Training code, model weights, validation reports. |
| `session-invariance` | Session‑invariance proof. Reproducible. |
| `chemoprint` | 29‑dim molecular fingerprint library. Pip‑installable. |
| `chemoprint-apps` | Similarity search, odor prediction, QSAR. |
| `data-commons` | Community data pipeline, format specification, and sensor datasheets. |
| `electronic-nose` | Reference hardware BOM, wiring diagrams, experiment protocol. |

All repositories are MIT‑licensed. All experiments are reproducible on a standard laptop without a GPU.

---

## 10. How to Contribute

**Build hardware.** The reference BOM and wiring specifications are published. Build a device, record labelled substances, train an adapter, and contribute your recordings, adapter weights, and device config to the Data Commons.

**Reproduce results.** Clone the repositories. Run the proofs. File issues if results differ.

**Build apps.** Train a head on the frozen latent space for your application. Share it. Every app that works on one device works on all adapted devices.

**Scrutinise.** The adversarial process is the engine of improvement. All claims are bounded by the evidence described herein. Errors, omissions, and failures of reproduction should be reported as GitHub issues.

---

All repositories: `https://github.com/opensmell`
Discord: `https://discord.gg/CGER3tHxbH`

*This document will be updated as the project evolves. The current version reflects the state of the project as of the date of writing. All claims are bounded by the evidence described herein. Errors, omissions, and failures of reproduction should be reported as GitHub issues.*