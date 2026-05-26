# Changelog

## 2026-05-19 — Pre-hardware audit and interoperability testing

### Added

- `electronic-nose/test_zero_shot.py`: Tests whether the encoder identifies
  substances from a simulated 4-MQ-sensor device with no adaptation.
- `electronic-nose/train_adapter.py`: Trains a lightweight MLP (4→32→6) that
  maps MQ-sensor readings to SmellNet-format readings.
- `electronic-nose/test_adapter.py`: Validates the adapter on held-out
  substances, reporting chemoprint cosine similarity to the FooDB reference.

### Changed

- `electronic-nose/visualizer.py`: Replaced fake chemoprint categories
  ("Base/Mid/Top/Texture/Signature") with the real 29 dimension names from the
  canonical chemoprint spec, grouped as base properties (0–11), topological
  indices (12–14), and functional groups (15–28).
- `electronic-nose/visualizer.py`: `--live` flag now exits with error code
  instead of printing a message and silently returning.
- `electronic-nose/visualizer.py`: Auto-find now picks a random CSV from the
  SmellNet cache instead of always returning the alphabetically-first file.
- `opensmell/opensmell/preprocessing.py`: Added optional `sensor_map` parameter
  to `load_csv()` for flexible column remapping. If provided, columns are
  renamed before validation. Existing behaviour unchanged when `sensor_map` is
  `None`.

### Known Issues (documented, not fixed)

- **Contrastive loss silent zero-gradient**
  (`universal-encoder/src/train_encoder.py:258`): When a batch contains ≤1
  sample per class, `supcon_loss` returns 0. Occurs frequently with 44 classes
  and batch_size=64. The encoder achieved R²=0.882 despite this. Future
  retraining should use gradient accumulation or a different contrastive
  formulation.
- **Training segments use 50% overlap** (`session-invariance/`): Adjacent
  training samples share half their data. Test set is unaffected because split
  is by session before segmentation. Reported accuracy of 81.78% may be
  slightly inflated by this autocorrelation.
- **Confidence threshold (0.7) is uncalibrated**: No statistical OOD detection.
  Random noise scores ~0.69 against food prototypes. The threshold was chosen
  empirically from training-set similarity distributions.
- **Training substances are the intersection of SmellNet and FooDB** (44 of
  50): 6 SmellNet substances absent from FooDB were excluded. The 44 are not a
  representative sample of chemical space.
- **Encoder is session-invariant but not device-invariant**: All training data
  from one sensor board. Cross-device generalisation requires an adapter or
  domain-adversarial retraining.
