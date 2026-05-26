# OpenSmell — Mission

[TECHNICAL.md](https://github.com/opensmell/opensmell/blob/master/TECHNICAL.md).
[Discord Community](https://discord.gg/CGER3tHxbH)
Donate USDC on Polygon: 0x699d0178f16484509f57d4d77f310b6b617621ce


---

## 1. The Pattern of Technological Progress

Technological advancement follows a recursive pattern: knowledge creates tools, tools automate manual processes, and freed human capital solves harder problems. Major social shifts—industrialisation, the internet—are downstream effects of prior technological emancipation, not their initial cause.

The mechanism is the externalisation of human capability. We model our innate faculties into reproducible, scalable systems. Fire externalised metabolic digestion. Writing externalised memory. The camera externalised vision, mastering the physics of light. The microphone externalised hearing, mastering the physics of sound waves.

But we did more than copy our senses. We improved them. The camera let us see beyond the visible spectrum—infrared, ultraviolet, X‑rays. The microphone let us hear ultrasonic frequencies and reconstruct whispers from across a room. We did not just digitise our biology; we transcended it. Today we can see a single atom and hear the echo of the Big Bang.

Each transition converted a private, biological experience into a public, programmable utility, creating immense collective value. A critical gap remains.

## 2. The Anomaly: The Undigitised Chemical Sense

Where is the digital nose? We lack a standard file format for smell, a universal olfactory sensor, or a device capable of broadcasting a smell profile as seamlessly as a speaker emits sound.

Our primary instrument for chemical sensing remains biological: the human nose. Its output is a subjective, unshareable percept. Unlike the deterministic physics of light and sound, the mapping from a molecule's structure to its perceived odour is not defined by a simple, finite set of rules. This is not an engineering oversight. It is a profound explanatory gap in our scientific knowledge.

## 3. The Core Scientific Challenge

Digitisation requires compression. We achieved this for vision with wavelength and intensity, and for hearing with frequency, amplitude, and timbre. Olfaction presents a problem of far higher dimensionality. The human olfactory system employs roughly 400 distinct receptor types. A single odorant molecule can activate multiple receptors, and a single receptor can be activated by numerous molecules. The result is a combinatorial explosion of possible signals and a perceptual space that is high‑dimensional and poorly mapped.

But a predictive theory of olfaction is not the only path to digitisation. Just as we digitised sound without fully understanding psychoacoustics, we can digitise smell without fully mapping the perceptual space. The key is to find a stable, measurable, and device‑invariant representation—an intermediate that bridges the physical chemistry of molecules and the messy reality of sensor hardware.

## 4. The OpenSmell Thesis

The path forward cannot be through deeper proprietary silos. The solution is to treat olfaction as an open scientific and engineering problem.

OpenSmell is a public experiment in knowledge creation. Our methodology is to build and share working infrastructure, then learn from its successes and failures in the open. We have already made substantial progress, and we have also encountered clear failures. Both are published, criticisable, and informative.

### 4.1 What We Have Built

We defined an open standard called the Chemoprint: a 29‑dimensional vector of physicochemical properties computable from any molecule's structure using open‑source tools. Every property is deterministic, and every dimension has a name a chemist can read. The Chemoprint is not a black box. It is a human‑readable chemical fingerprint—and it can be measured by an electronic nose.

### 4.2 Hardware Validation: Pure Compounds

Using a public gas‑sensor dataset recorded over 36 months, we showed that a commercial sensor array can be calibrated to output the Chemoprint for pure compounds with near‑perfect accuracy. The Chemoprint is not a theoretical abstraction; it is a measurable quantity.

### 4.3 The Limits of Simple Calibration

We tested the idea of calibrating any new e‑nose using six pure reference compounds. A convex hull analysis of 4,565 common odorants revealed that the six anchors cover only 0.1% of real smells. For the other 99.9%, the model would be extrapolating wildly. The six‑anchor approach is dead.

We also tested a simpler calibration method: assume two devices differ by a per‑sensor gain and offset. On the same public dataset, classification accuracy dropped from 47% before calibration to 33% after calibration. The correction made performance worse. Long‑term sensor drift is non‑linear and cannot be corrected by a simple linear transform.

These negative results are as important as the positive ones. They tell us that calibration is a hard problem, and that simple solutions are insufficient. The field needs a new approach.

## 5. From Explanation to Infrastructure

If we cannot calibrate with a handful of anchors and cannot use a simple affine transform, what can we do?

The answer is to learn a shared representation from data contributed by many devices. Train a neural network on a large, diverse dataset of raw sensor readings from many different e‑noses, all measuring the same smells. The network learns to project any device's readings into a common latent space where the same smell produces the same representation, regardless of the device.

Once such a latent space exists, the need for per‑device calibration disappears. The more devices contribute data, the better the representation becomes. The system improves with use and with diversity of data.

To scale without centralising raw sensor data—addressing both privacy and bandwidth—we plan to use federated learning. Each participant downloads the model, trains it on their local e‑nose readings, and sends back only the model updates. The server aggregates updates and improves the global model. No raw sensor data ever leaves the participant's device.

## 6. The Imperative for Openness

Foundational infrastructure thrives through open, collaborative development. The internet, Linux, Python, Wikipedia—none were built by a single company or closed lab. They grew because anyone could contribute, criticise, and improve.

Openness is not merely beneficial for olfaction; it is a methodological necessity. Only a transparent process of public code, data, and peer critique can accelerate the iterative failures and insights required to solve a problem of this complexity. OpenSmell publishes all artefacts—successes and failures alike—as a public good.

## 7. Current Status, Roadmap, and Vision

**Completed foundation:**

- Chemoprint specification and implementation (29‑dim, open‑source).
- Hardware validation on pure compounds.
- Convex hull analysis demonstrating the failure of anchor‑based calibration.
- Affine calibration failure documented and archived.
- Session‑invariance proof: the same substance measured on different days produces the same output with no calibration (81.8% accuracy across 50 food substances).
- A pip‑installable SDK, a community data pipeline, and a reference hardware specification.

**Immediate next steps:**

- Collect multi‑device data to train a device‑invariant encoder.
- Release federated learning tools so participants can improve the global model without sharing raw data.
- Launch the public data repository on HuggingFace and invite community contributions.

**Future horizon:**

- Open‑source reference e‑nose design (ESP32 + sensors, total cost under $50).
- A smell‑app marketplace where any developer can build and deploy applications that work on any OpenSmell‑compatible device.
- Multi‑modal foundation models integrating chemical structure, sensor data, and semantic odour descriptors.
- Applications: food spoilage detection, breath‑based health screening, gas‑leak safety alarms, personalised fragrance blending.

## 8. A Call for Open Collaboration

OpenSmell is a community endeavour. We seek collaborators across disciplines: cheminformaticians, machine learning engineers, hardware designers, domain experts in olfaction and food science, and community builders. The most valuable contribution, regardless of background, is critical engagement. Scrutinise the designs, attempt to reproduce the results, and try to falsify the assumptions. File detailed issues. This adversarial process is the engine of robust discovery.

## 9. Conclusion

In 1975, a Kodak engineer built the first digital camera. It was the size of a toaster, weighed eight pounds, and recorded a 0.01‑megapixel image onto a cassette tape. His colleagues probably asked, "Why would anyone want to look at pictures on a television?"

Today, more than 80 trillion photos are taken every year. The visual world has been commoditised—not by an open ecosystem of sensors, formats, and software. The same transformation will happen to smell. It will start with clunky devices, hobbyist data, academic arguments about shape versus vibration, and failed experiments. Then it will be everywhere.

Digitising smell is more than building gadgets. It is constructing a fundamental lens on molecular reality—one that promises to transform health, ecology, agriculture, and material science. For this new sense to benefit all, its foundation must be built as a public good: open, auditable, and unowned.

The path is clear. The tools are available. The first pieces are in place. We invite you to join the effort to build this essential infrastructure together.

---

**Get involved**

- Code & data: [github.com/opensmell](https://github.com/opensmell)
- Technical account: [TECHNICAL.md](https://github.com/opensmell/opensmell/blob/master/TECHNICAL.md)
- Community: [Join Discord](https://discord.gg/CGER3tHxbH)
- Donate USDC on Polygon: 0x699d0178f16484509f57d4d77f310b6b617621ce