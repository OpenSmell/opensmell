# Plan: Constant Extraction for Non-Power-Law (MEMS/VOC) Sensors

Status: **PLAN ONLY — not implemented.** This is a future work item. The
immediate goal was to record how we would *rigorously* obtain concentration
response constants for sensors that do not publish a power-law `(a,b)` pair
(SGP30/SGP40, TGS8100, BME680, and any future MOS-MEMS / electrochemical part).

## Why this is needed

The embedded `sensors.json` gives each power-law (MQ-family) sensor a converted
`rr = a·C^b` response per gas, from the MQUnifiedSensorsLib example `.ino`
tables. The MEMS family — even though some (TGS8100, BME680) output a real gas
resistance — has **no authoritative, publicly available power-law to a target
gas concentration**:

- **Sensirion SGP30 / SGP40**: processed outputs only (eCO2 ppm / TVOC ppb /
  VOC index 0–500), produced by a closed on-chip baseline algorithm. No
  resistance-to-concentration power law is published. The raw gas ADC exposed by
  some variants has no vendor-provided mapping.
- **Figaro TGS8100**: datasheet gives a *single* sensitivity ratio
  (Rs(10 ppm H2)/Rs(air) ≈ 0.6) and an iso-butane curve. One point cannot define
  a two-parameter power law.
- **Bosch BME680**: outputs MOX gas resistance (kΩ), documented as a *relative*
  VOC sensor. Bosch provides only the closed-source BSEC library for IAQ; there
  is no per-gas sensitivity table.

Fabricating `(a,b)` values for these would be scientifically wrong, so the SDK
currently marks them `power_law_calibratable = False` and refuses to produce
constants for them.

## Objective

Build a reproducible pipeline that, for a given sensor, yields:
1. A defensible concentration-response model for the gases it genuinely
   responds to (not necessarily a single power law), OR
2. A documented, explicit "no reliable public constants" verdict with the
   evidence trail, so future readers do not re-dig the same ground.

## Proposed approach

### Phase A — Literature & datasheet harvest (offline, static)
- Collect the authoritative primary docs per sensor (official datasheet,
  app notes, vendor integration guides, IEEE/peer-reviewed characterization
  papers) and record stable URLs + retrieval date in a JSON provenance file.
- Extract every hard data point the vendor actually publishes:
  - sensitivity ratio(s) at specific concentrations/temperatures/humidity
  - Rs/R0 curves from datasheet figures (via WebPlotDigitizer / plot digitizer
    on the official figure, NOT on third-party re-plots)
  - operating/calibration condition (20 °C, 65 % RH, load resistor)
- For each sensor, attempt a defensible fit (power law and, if the data
  supports it, an alternative single-transducer model) with error bounds and
  goodness-of-fit, rejecting fits that use only one point.

### Phase B — Empirical cross-check (requires hardware, later phase)
- Controlled exposure rig: known reference gases (or ethanol/iso-butane as
  transfer references) at several concentrations in a chamber.
- Fit per-rig, per-channel reference points (§4.6) and compare against the
  harvested constants to measure transfer error.
- This is the only way to validate cross-device transferability for
  relative-response sensors.

### Phase C — Schema & SDK integration (after A and B yield defensible data)
- Extend `sensors.json` with a new response-model discriminator (e.g.
  `"model": "power_law" | "iso_log" | "voc_index" | "empirical"`) so the
  calibration machinery knows *how* to invert a reading, not just which
  constants to use.
- Add provenance fields: `retrieved_on`, `figure_source`, `fit_quality`
  (r², n_points), `conditions` (temp/humidity).
- Extend `opensmell/opensmell/constants` loaders and `calibration.py` to
  consume the new model types without breaking power-law MQ channels.

## Decisions / scope
- **Do not invent constants.** Any entry without a validated source stays
  `power_law_calibratable = False`.
- **Prioritize** real device builds (osmograph-desktop, opensmell SDK prod
  paths) over this using-labour project. This doc is the placeholder so the
  knowledge and links are not lost.
- Mirrors: `sensors.json` and `data-commons/sensor_constants.json` are kept
  byte-identical; every update here must resync the mirror.

## Links gathered so far (verifiable)
- Sensirion SGP30: https://sensirion.com/products/catalog/SGP30
- Sensirion SGP40: https://www.sensirion.com/products/catalog/SGP40
- Figaro TGS8100 datasheet: https://www.figarosensor.com/product/docs/TGS8100%280914%29.pdf
- Figaro TGS8100 product page: http://www.figarosensor.com/feature/tgs8100.html
- Bosch BME680 datasheet: https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bme680-ds001.pdf
- Bosch BME680 product page: https://www.bosch-sensortec.com/en/products/environmental-sensors/gas-sensors/bme680
