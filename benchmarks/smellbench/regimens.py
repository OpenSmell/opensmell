"""Versioned regimen registry for every OpenSmell benchmark.

Every score is qualified by its regimen:

    chemistry x environment x protocol x timing x device-generation

The registry is curated from each module's own docstring (the modules remain
the source of truth; this is the machine-readable mirror).  A regimen always
accompanies a score on the report; data quality is a PASS/FLAG gate, never a
score.
"""

from .schema import REGIMEN_FIELDS, REGIMEN_VERSION

#: canonical (chemistry, environment, protocol, timing, device) field order
FIELDS = REGIMEN_FIELDS

REGISTRY = [
    {
        "key": "physics_selfcheck",
        "module": "bench_physics_selfcheck",
        "series": "Bench 0",
        "title": "physics self-check",
        "claim_class": "A",
        "claim": (
            "the physics/ causal chain returns physically sane numbers end to "
            "end, so every downstream benchmark inherits a checked foundation"),
        "regimen": {
            "chemistry": [
                "langmuir adsorption (theta(C=K) == 0.5)",
                "O2- / O- / O2^2- theoretical exponents",
                "masking_metric oracle verdicts: ideal / masking / synergistic",
            ],
            "environment": [
                "arrhenius baseline multiplier 25 C vs 40 C",
                "humidity baseline suppression RH 30 vs 70 (class D model)",
            ],
            "protocol": [
                "closed-form audits: adsorption, depletion (Debye length, band "
                "bending, depletion width), power-law fit with true (A, b) "
                "recovery, first-order kinetics, mixture additivity oracle",
            ],
            "timing": [
                "static model checks, no sampling",
            ],
            "device": [
                "none (pure physics module; no hardware)",
            ],
        },
        "headline": (
            "sanity booleans all true; Ld~2nm @ 600K, psi~0.6V, W~10nm; "
            "b recovered to <0.05"),
    },
    {
        "key": "uci_drift",
        "module": "bench_uci_drift",
        "series": "Bench 1",
        "title": "UCI drift: observability + constant selection",
        "claim_class": "F",
        "claim": (
            "the six gases stay observable within a rig and across batches, "
            "and a subset of the 128 features are device-stable constants "
            "whose use helps or hurts the batch1 -> batch10 transfer"),
        "regimen": {
            "chemistry": ["six UCI drift gases (labels 1-6)"],
            "environment": [
                "uncontrolled laboratory drift across months; batch index == "
                "drift time",
            ],
            "protocol": [
                "10 batches; within-rig classification among the 6 gases",
                "drift corruption: separation between gases across batches",
                "device-stable constants: which of 128 features are stable "
                "across batches",
                "cross-batch transfer batch1 -> batch10, restricted vs full "
                "feature set",
            ],
            "timing": [
                "batch == session; 10 sessions",
            ],
            "device": [
                "16-sensor MOX array, one nominal platform, 8 features per "
                "sensor = 128 features",
            ],
        },
        "headline": (
            "same-gas sits 4.8-29.6 Mahalanobis units apart batch1->batch10; "
            "unanchored transfer baselines 0.379 raw / 0.366 stable-40"),
    },
    {
        "key": "turbulent_mixtures",
        "module": "bench_turbulent_mixtures",
        "series": "Bench 2",
        "title": "turbulent mixtures: observability, additivity, power law",
        "claim_class": "B",
        "claim": (
            "the 30 concentration configs are observable from per-channel "
            "primitives; mixture response follows the ideal-independent-"
            "adsorption null s_AB = s_A s_B rather than competitive Langmuir "
            "masking; power law rr = a C^b matches the sensors.json constants"),
        "regimen": {
            "chemistry": [
                "ethylene +- CO / methane at 4 levels; 30 concentration "
                "configs, 6 repeats each",
                "single-gas amplitudes s_A, s_B; additive null s_AB = s_A s_B",
            ],
            "environment": [
                "turbulent flow recordings; humidity arms vary across files "
                "(RH ~ 35.6-45.4)",
            ],
            "protocol": [
                "observability of 30 configs + which pairs are ambiguous",
                "mixture additivity: ideal-independent-adsorption null vs "
                "competitive Langmuir masking",
                "power law rr = a C^b fit on single-gas configs vs sensors.json "
                "(a, b) for the same models",
            ],
            "timing": [
                "per-recording sessions; repeats span sessions",
            ],
            "device": [
                "8 TGS sensors (turbulent array)",
            ],
        },
        "headline": (
            "ambiguity list across 30 configs; additivity null verdict; "
            "power-law (a, b) vs datasheet"),
    },
    {
        "key": "cross_device_primitives",
        "module": "bench_cross_device_primitives",
        "series": "Bench 3",
        "title": "cross-device primitive stability",
        "claim_class": "F",
        "claim": (
            "a small set of physical primitives (log-ratios, normalized "
            "amplitudes, directions) survives device variation better than "
            "raw features"),
        "regimen": {
            "chemistry": ["the same chemical state driven through many devices"],
            "environment": ["synthetic device-parameter draws; same headspace for the real leg"],
            "protocol": [
                "SYNTHETIC: sample 30 virtual devices via "
                "devices.device_parameter_model, drive the same chemical "
                "state, extract primitives, measure the across-device "
                "coefficient of variation of each primitive",
                "DEV_BOUND primitives should be stable; NONE-calibrated "
                "(baseline/raw) primitives should not",
                "REAL: the dynamic-mixtures file carries two nominally "
                "identical TGS2602 channels (ch0, ch8) reading the same "
                "headspace; compare their primitives for two-device "
                "reproducibility",
            ],
            "timing": ["not specified beyond the corpus streams"],
            "device": [
                "30 virtual devices (device_parameter_model)",
                "2 real TGS2602 channels of one array (ch0, ch8)",
            ],
        },
        "headline": (
            "per-primitive across-device CoV; DEV_BOUND vs NONE split; "
            "ch0-vs-ch8 two-device reproducibility"),
    },
    {
        "key": "environment_robustness",
        "module": "bench_environment_robustness",
        "series": "Bench 4",
        "title": "environment robustness: temperature and humidity",
        "claim_class": "B",
        "claim": (
            "the expected raw-baseline swing over a plausible RH 20-70% and "
            "T 20-45 C range is quantified, and the UCI turbulent data give "
            "the empirical d(log R0)/dRH and d(log R0)/dT"),
        "regimen": {
            "chemistry": ["air recordings (baseline R0, no target gas)"],
            "environment": [
                "RH 20-70% and T 20-45 C plausible range (physics leg)",
                "UCI turbulent recordings log T and RH per row",
            ],
            "protocol": [
                "PHYSICS: humidity/temperature modules quantify expected "
                "baseline swing over the plausible range; this is the "
                "raw-feature corruption environment metadata must absorb",
                "REAL UCI: regress each channel's log-baseline on (T, RH) "
                "across 180 recordings in air",
            ],
            "timing": [
                "per-row T/RH logging on the turbulent corpus",
            ],
            "device": [
                "8 TGS sensors (turbulent array)",
            ],
        },
        "headline": (
            "per-channel R2 of log-baseline on (T, RH); empirical "
            "d(log R0)/dRH and d(log R0)/dT coefficients"),
    },
    {
        "key": "datasheet_constants",
        "module": "bench_datasheet_constants",
        "series": "Bench 5",
        "title": "datasheet power-law constants: audit vs theory/data",
        "claim_class": "E",
        "claim": (
            "sensors.json (a, b) constants are trustworthy: most pass the "
            "physical sanity gate, the b distribution over reducing gases "
            "fits the ionosorption band, and the MQUnified conversion is "
            "consistent"),
        "regimen": {
            "chemistry": [
                "reducing gases; ionosorption O2- .. O- band "
                "b in [-1, -0.25]",
            ],
            "environment": ["no environment (datasheet priors only)"],
            "protocol": [
                "physical sanity gate: formula vs datasheet fixture, "
                "fraction passing",
                "b distribution over reducing gases vs ionosorption theory "
                "band and vs the empirical UCI/turbulent exponent",
                "MQUnified -> OpenSmell convention conversion consistency "
                "(rr == 1 at a^(1/b); a ~ b interplay)",
            ],
            "timing": ["no timing (static constants)"],
            "device": [
                "sensors.json dataset: the OpenSmell datasheet-prior source "
                "for the array model",
            ],
        },
        "headline": (
            "fraction passing sanity gate; b distribution vs [-1, -0.25]; "
            "convention-conversion consistency"),
    },
    {
        "key": "session_anchor",
        "module": "bench_session_anchor",
        "series": "WS1",
        "title": "session-anchor + drift deconfounding",
        "claim_class": "F",
        "claim": (
            "per-session anchoring plus environment residualization collapses "
            "the drift corruption measured in bench_uci_drift (same gas "
            "batch1->batch10, 4.8-29.6 Mahalanobis units apart)"),
        "regimen": {
            "chemistry": ["the six UCI drift gases (anchoring leg); ethylene +- CO/methane (residualization leg)"],
            "environment": [
                "UCI drift: no T/RH recorded",
                "UCI turbulent: T/RH recorded per row",
            ],
            "protocol": [
                "DRIFT ANCHOR: session anchor = self-normalization, every "
                "recording expressed relative to its own session statistics "
                "(per-feature mean/std) -- the intra-session R0 convention at "
                "feature level; remeasure batch1->batch10 same-gas separation "
                "and transfer vs unanchored baselines",
                "ENV RESIDUALIZATION: regress each channel's logR0 on "
                "(RH, T); logR0_resid = logR0 - fit; measure how much "
                "inter-repeat (same config, 6 repeats) cloud variance is "
                "environmental and whether residualization regenerates "
                "dose-level separation that the raw baseline blurred",
            ],
            "timing": [
                "batch == session; 10 sessions (anchoring leg)",
            ],
            "device": [
                "16-sensor UCI drift array (anchoring leg)",
                "8 TGS turbulent array (residualization leg)",
            ],
        },
        "headline": (
            "anchored batch1->batch10 transfer vs 0.379 raw / 0.366 "
            "stable-40; residualized inter-repeat variance share"),
    },
    {
        "key": "drift_trajectory",
        "module": "bench_drift_trajectory",
        "series": "A2",
        "title": "drift trajectory fit: smooth or regime-jumping?",
        "claim_class": "D",
        "claim": (
            "the up-to-~65-sigma-per-channel drift is either a smooth, "
            "monotone function (per-array auto-calibration from a fitted "
            "curve) or regime-jumping (humidity angle, environment events -> "
            "online adaptive baseline needed)"),
        "regimen": {
            "chemistry": ["six UCI drift gases"],
            "environment": [
                "ambient drift across 10 sessions; humidity/environment "
                "events hypothesized as regime jumps",
            ],
            "protocol": [
                "fit a 10-point per-feature drift trajectory (10 batches "
                "== 10 sessions)",
                "fit written generically so the identical code runs on the "
                "Worner corpus the moment its raw 62-channel/40-day CSV is "
                "available",
            ],
            "timing": [
                "10 batches == 10 sessions (UCI); 40 days / 62 channels "
                "(Worner, deferred)",
            ],
            "device": [
                "16-sensor UCI drift array (deployment)",
                "Worner 62-channel array (future corpus)",
            ],
        },
        "headline": (
            "smooth-vs-regime verdict on the 10-point trajectory; ~65 "
            "sigma/channel drift magnitude"),
    },
    {
        "key": "b_dynamic",
        "module": "bench_b_dynamic",
        "series": "WS4",
        "title": "per-session dynamic calibration exponent b",
        "claim_class": "B",
        "claim": (
            "track b = d log10(R/R0) / d log10(C) across sessions; the "
            "per-session trajectory is reported with agreement against the "
            "b ~ N(-0.33, 0.1) prior and an honest power-to-detect flag for "
            "the b-vs-RH slope (the O2-/O- crossover predicting |b| 0.25 -> "
            "0.5 is NOT resolvable on this corpus)"),
        "regimen": {
            "chemistry": [
                "ethylene ladder 31/46/96 ppm x 6 repeats on 8 TGS sensors",
            ],
            "environment": [
                "two interleaved humidity arms: even-indexed files RH ~ "
                "35.6-40.7, odd-indexed RH ~ 43.3-45.4",
                "RH span 35.6-45.4% cannot resolve the O2-/O- ionosorption "
                "crossover",
            ],
            "protocol": [
                "session definition 'windows': six consecutive 30-recording "
                "blocks (the L/M/H ladder cycles every 30 recordings, so "
                "every block has full concentration coverage x 6 repeats)",
                "session definition 'regimes': the two interleaved humidity "
                "arms (even/odd indexed files); b-vs-RH slope with a "
                "power-to-detect flag",
            ],
            "timing": [
                "180 recordings; per-session trajectory",
            ],
            "device": [
                "8 TGS sensors (turbulent array)",
            ],
        },
        "headline": (
            "per-session b trajectory vs N(-0.33, 0.1); b-vs-RH slope; "
            "power-to-detect flag (crossover unresolvable)"),
    },
    {
        "key": "mixture_generalization",
        "module": "bench_mixture_generalization",
        "series": "WS3",
        "title": "additive-dose mixture generalisation: hold-out R2",
        "claim_class": "B",
        "claim": (
            "on the two UCI dynamic-mixture streams the additive-dose law "
            "s_AB = (1 + s_A)(1 + s_B) - 1 (independent Langmuir adsorption, "
            "two gases adding linearly in dose) predicts held-out mixture "
            "amplitudes from single-gas amplitudes alone"),
        "regimen": {
            "chemistry": [
                "two streams: ethylene + methane and ethylene + CO on 16 "
                "Figaro TGS channels",
                "single-gas amplitude s_X = |R - R0| / R0 at its own "
                "concentration",
                "additive-dose law s_AB = (1 + s_A)(1 + s_B) - 1",
            ],
            "environment": ["lab streams; environment fixed (chemical-only)"],
            "protocol": [
                "R0 estimated from the clean-air block at the start of each "
                "stream",
                "steady-state amplitude from the tail (last 30%) of each "
                "config block, skipping onset transients",
                "train on single-gas amplitudes only; predict held-out "
                "mixture amplitudes via the law",
                "hold-out R2 and bias in log10-amplitude space",
                "stacked-identical-TGS bootstrap: the 4 replicate TGS2602 "
                "channels (ch 0, 1, 8, 9) as four independent device-level "
                "draws to bound law-deviation variance internal to a sensor "
                "family",
            ],
            "timing": [
                "config blocks: onset transient skipped, tail 30% used",
                "per-second ground-truth schedule",
            ],
            "device": [
                "16 Figaro TGS channels: 4x TGS2602 / 4x TGS2600 / 4x "
                "TGS2610 / 4x TGS2620",
            ],
        },
        "headline": (
            "hold-out R2 and bias in log10-amplitude space; family-internal "
            "law-deviation bound from the 4x TGS2602 bootstrap"),
    },
    {
        "key": "fleet_calibration",
        "module": "bench_fleet_calibration",
        "series": "WS5",
        "title": "fleet calibration: 10 drift batches as 10 device proxies",
        "claim_class": "F",
        "claim": (
            "drift harmonisation and on-site calibration ('ship it, "
            "calibrate on arrival') beat the raw cross-fleet transfer "
            "~0.38; the two-stage (a, b) concentration path is explicitly "
            "out of reach because the drift release has no concentration "
            "labels"),
        "regimen": {
            "chemistry": [
                "six gases (labels 1-6 only); no ppm labels in the drift "
                "release",
                "two-stage (a, b) concentration calibration NOT exercisable; "
                "rank-level fallback implemented, limitation stated in the "
                "report",
            ],
            "environment": [
                "each UCI batch = one device of the same nominal platform, "
                "deployed for one session, with feature-level drifting "
                "baselines",
            ],
            "protocol": [
                "within-fleet CV: train/test pooled batches 1-5 (same-device "
                "bound)",
                "raw cross-fleet: train 1-5 -> test 6-10 (naive transfer)",
                "drift-harmonised: A2-style per-feature log-drift curve "
                "fitted on the calibration fleet shifts every batch back to "
                "the fleet-anchor scale, then extrapolates into the "
                "deployment fleet",
                "on-site calibration: k = 2..30 labeled samples per gas per "
                "deployment device, recentre/Rescale toward the fleet "
                "location, then re-test",
            ],
            "timing": [
                "batch == device == one session",
            ],
            "device": [
                "a 10-array 'fleet' simulated from the UCI drift batches",
                "classifier RandomForest(200), fixed seed throughout",
            ],
        },
        "headline": (
            "headline to beat: raw cross-fleet transfer ~0.38; on-site "
            "calibration k = 2..30 sweep"),
    },
    {
        "key": "anomaly_v2",
        "module": "bench_anomaly_v2",
        "series": "WS2",
        "title": "four-axis anomaly detection on ground truth",
        "claim_class": "C",
        "claim": (
            "the residual axes detect real anomalies per axis on labelled "
            "ground truth: synthetic steps/spikes/ramps, novel SmellNet "
            "substances, dropped indoor-air activities, and per-session "
            "drift-batch health"),
        "regimen": {
            "chemistry": [
                "49 known SmellNet substances + a never-seen novel substance",
                "indoor-air activity classes + background + wine",
                "six UCI drift gases (drift-health leg)",
            ],
            "environment": [
                "mixed: synthetic injection, SmellNet offline recordings, "
                "indoor-air recordings, UCI drift sessions",
            ],
            "protocol": [
                "leg 1 synthetic: known baseline + injected step/spike/ramp "
                "anomalies; ground-truth labelled windows; per-axis AUC",
                "leg 2 SmellNet LOO: a never-seen substance must score as "
                "novel against the 49 known ones (AUC)",
                "leg 3 indoor-air drop: leave one activity class out; the "
                "dropped class is novel vs background + wine (AUC)",
                "leg 4 drift-batch health: UCI batches as coarse per-session "
                "R0-health; per-batch deviation vs batch 1 and the "
                "first-exceedance 'alarm batch' for the latent-drift axis",
                "axes: explicit EWMA (level), implicit DualKalman "
                "(drift-as-latent), physical additivity; fusion = max-|z|",
            ],
            "timing": [
                "per-second ground truth; per-session batches for drift ",
            ],
            "device": [
                "SmellNet arrays (legs 2-3); 16-sensor UCI array (leg 4)",
            ],
        },
        "headline": (
            "per-axis AUC; event-axis 0.87 headline on the synthetic leg"),
    },
    {
        "key": "separability_margin",
        "module": "bench_separability_margin",
        "series": "A3",
        "title": "stored separability margin of the deployed anomaly stack",
        "claim_class": "D",
        "claim": (
            "from stored scalar alarm counts alone, the clean-vs-event "
            "separability margin can be reconstructed in sigma units without "
            "the deployed threshold constant; it is a conservative lower "
            "bound if event noise exceeds clean noise"),
        "regimen": {
            "chemistry": [
                "u2_gas_leak gas; concentration varies by measurement file "
                "(per-day margin trends are noisy)",
                "no cross-analyte confusion recoverable (archive records "
                "event-vs-clean only)",
            ],
            "environment": ["field runs, archived in e-nose-evals/u2_gas_leak/results/*.json"],
            "protocol": [
                "from scalar alarm accumulators per staged window: "
                "s1_a/s1_t (clean/baseline stage, clean false-alarm rate) and "
                "s2_a/s2_t (exposure/event stage, event coverage), det/total "
                "(detection rate)",
                "m_clean = Phi^-1(1 - FPR_clean)",
                "q_event = Phi^-1(1 - coverage)",
                "margin = m_clean - q_event (upward shift of the response "
                "distribution during exposure, in clean-window sigma)",
                "assumptions stated: per-verdict deviation ~Gaussian per "
                "window; event-window sigma ~ clean sigma (else margin "
                "underestimated -- lower bound)",
            ],
            "timing": [
                "staged windows per measurement cycle",
            ],
            "device": [
                "archived deployment array (u2_gas_leak rig)",
            ],
        },
        "headline": (
            "m_clean ~ 3.0 if the calibrator is honest; margin in clean-window "
            "sigma as a lower bound"),
    },
    {
        "key": "sensor_memory",
        "module": "bench_sensor_memory",
        "series": "A1",
        "title": "sensor memory: exposure residue into the next baseline",
        "claim_class": "B",
        "claim": (
            "on the closest surrogate with the Worner cycle structure, an "
            "exposure's deflection partially carries into the next clean-air "
            "window (carryover) with a fitted recovery time constant tau; "
            "deferred vs surrogate results are kept visibly separate"),
        "regimen": {
            "chemistry": ["ethylene +- CO / methane exposures on the surrogate"],
            "environment": [
                "Worner corpus: the schema the product needs -- stage-1 "
                "baseline, stage-2 exposure, stage-3 recovery per cycle",
                "surrogate: UCI dynamic-mixtures lab streams (environment "
                "fixed)",
            ],
            "protocol": [
                "Worner (deferred): Cycle_Stage in {1,2,3}, ~300-550 "
                "rows/stage, 62 channels; raw CSV absent until it arrives "
                "under e-nose-evals/data/worner*/ or WORNER_CSV",
                "surrogate (live now): each air gap after an exposure is "
                "treated exactly like a Worner recovery stage; recovery tau "
                "fitted and carryover of the exposure deflection into the "
                "next baseline measured",
                "carryover + single-time-constant tau (A1 scope; W1 later "
                "tightens to multi-timescale)",
            ],
            "timing": [
                "Worner: ~300-550 rows per stage",
                "surrogate: hundreds of alternating exposure/air blocks on "
                "the 100 Hz stream",
            ],
            "device": [
                "Worner 62-channel array (future)",
                "16-channel Figaro surrogate: 4x TGS2602/2600/2610/2620",
            ],
        },
        "headline": (
            "median carryover 11% and recovery taus feed W3's residue model "
            "alpha(t)"),
    },
    {
        "key": "minimal_array",
        "module": "bench_minimal_array",
        "series": "WS6",
        "title": "minimal array spec: forward sensor selection under drift",
        "claim_class": "F",
        "claim": (
            "greedy forward selection under leave-one-batch-out drift finds a "
            "7-sensor chemoprint that beats the full 16-sensor array, so the "
            "minimal spec is a hardware BOM cut, not a research question"),
        "regimen": {
            "chemistry": ["six UCI drift gases"],
            "environment": ["drift across 10 real sessions (LOBO = the honest drift regime)"],
            "protocol": [
                "greedy forward selection over sensors (blocks of 8 features "
                "each, the UCI drift corpus layout) under leave-one-batch-out "
                "over 10 sessions",
                "scored with a fast closed-form LDA on std features",
                "selected path verified on the same LOBO protocol with the "
                "fleet-standard RandomForest (200 trees, seed 0)",
                "sensor labels are positional (indices 0..15 -> feature "
                "blocks [8i, 8i+8)); the spec is reported by index for "
                "mapping onto the deployment BOM",
            ],
            "timing": [
                "10 sessions, leave-one-batch-out",
            ],
            "device": [
                "16-sensor UCI array reduced to a 7-sensor chemoprint spec",
            ],
        },
        "headline": (
            "LDA-LOBO 0.8885 (7 sensors) vs 0.8469 (16 sensors); 3 sensors "
            "already >= 95% of the full-array score"),
    },
    {
        "key": "dynamic_memory",
        "module": "bench_dynamic_memory",
        "series": "W1",
        "title": "controlled history dependence & multi-timescale recovery",
        "claim_class": "B",
        "claim": (
            "with concentration fixed, a shorter clean gap lowers the next "
            "response (R_A(2) != R_A(1)); self-priming is gas/device "
            "specific and cannot be a single scalar; exposure recovery needs "
            "TWO time scales in ~97% of fitted tails, so h_t cannot collapse "
            "to one constant"),
        "regimen": {
            "chemistry": [
                "ethylene +- CO / methane configs from ethylene_CO.txt and "
                "ethylene_methane.txt; clean = no gas (per-second ground "
                "truth)",
            ],
            "environment": [
                "lab streams hold the environment fixed -- this is the "
                "chemical history dependence, the cleanest part of the drift "
                "stack (stated up front in the verdict)",
            ],
            "protocol": [
                "A -> clean -> A: same (gas, ppm) config repeated after "
                "clean-air gaps of different lengths; response modulation vs "
                "gap = the R_A(2) != R_A(1) test with concentration fixed",
                "A -> B -> A: same config following a different gas vs after "
                "itself; amplitude difference = history dependence not "
                "concentration",
                "multi-timescale recovery: R(t) = r_inf + a1 exp(-t/t1) + "
                "a2 exp(-t/t2), scored (AIC) against the single exponential",
                "every quantity computed twice (once per mixture file)",
            ],
            "timing": [
                "100 Hz sampled, 1 Hz resistance grid; per-second ground "
                "truth",
                "MIN_EPISODE_S = 10, MIN_CLEAN_GAP_S = 5, MIN_GAP_FOR_TAU_S = "
                "25, R0_WIN_S = 20, TAIL_WIN_S = 10",
            ],
            "device": [
                "16 channels in 4 families: 4x TGS2602 (ch 0,1,8,9), 4x "
                "TGS2600 (ch 2,3,10,11), 4x TGS2610 (ch 4,5,12,13), 4x "
                "TGS2620 (ch 6,7,14,15)",
                "the 4x TGS2602 replicates act as four device-level draws",
            ],
        },
        "headline": (
            "gap-slope -0.361/-0.414 per file; priming +0.152; 97% bi-exp "
            "tails (t_fast 15-25 s, t_slow 45-60 s)"),
    },
    {
        "key": "detector_memory_fp",
        "module": "bench_detector_memory_fp",
        "series": "W2",
        "title": "detector false-positive decomposition: chemistry vs memory",
        "claim_class": "C",
        "claim": (
            "FP(bin) rising as the gap since the last exposure shrinks shows "
            "the alarms are triggered by physical exposure residue; FP(deep "
            "clean, >> 60 s) is the algorithm's own false-alarm floor; the "
            "excess is the memory-attributable share of the FP budget"),
        "regimen": {
            "chemistry": [
                "truth gas: scheduled ppm > 0; clean air: scheduled ppm == 0 "
                "(per-second dataset schedule)",
            ],
            "environment": ["lab streams; environment fixed (chemical memory only)"],
            "protocol": [
                "production residual axes: explicit EWMA level, implicit "
                "DualKalman innovation + latent-delta, max-|z| fusion; run "
                "on the 16 channels of the UCI dynamic-mixtures streams",
                "every air-second alarm labelled by time since the previous "
                "exposure ended; FP rate as a function of that distance IS "
                "the memory footprint",
                "short-gap FP - deep-clean FP = memory-attributable share",
            ],
            "timing": [
                "BIN_EDGES = [0, 10, 30, 60, 120, 300, inf] s; "
                "REF_WIN_S = 600; per-second ground truth",
            ],
            "device": [
                "16 Figaro TGS channels of the dynamic-mixtures array",
            ],
        },
        "headline": (
            "short-gap FP 0.36 vs deep-clean FP 0.054; ~62% of detector FP "
            "is memory residue"),
    },
    {
        "key": "identifiability",
        "module": "bench_identifiability",
        "series": "W3",
        "title": "sensor identifiability audit: dose and gas identity",
        "claim_class": "D",
        "claim": (
            "the minimal window needed to tell two gas conditions apart is a "
            "closed form of replicate scatter plus memory residue; because "
            "level noise does not shrink with the window, every pair has an "
            "accuracy ceiling -- some pairs are impossible at any window"),
        "regimen": {
            "chemistry": [
                "gas-condition pairs from the 30 configs, mixtures included",
                "adversarial memory: the worst residue is the same-gas full "
                "dose just measured (a real field scenario)",
            ],
            "environment": [
                "fixed lab; per-channel scatter decomposed into level "
                "(recording-to-recording) + sample (within-window temporal)",
            ],
            "protocol": [
                "dose response vector v_i[c] = log R0[c] - log R_tail[c] "
                "(R0 = clean-air baseline, first ~12 s)",
                "window variance var_c(k) = sigma_level_c^2 + "
                "sigma_sample_c^2 / (k * fps)",
                "standardized distance d_k; 2-class Bayes accuracy = "
                "Phi(d_k / 2); ceiling d_inf at k -> inf",
                "memory residue alpha(t) = 0.11 * (0.6 e^{-t/15} + 0.4 "
                "e^{-t/55}) of the prior condition's vector, calibrated to "
                "W1 recovery taus (15, 55) and the A1 median carryover "
                "(11%); worst-case and typical residue both reported",
                "sweeps: window k in {0.5,1,2,5,10,30,60,120,300} s x "
                "residue gap t in {1,30,120,600} s; per pair: minimal window "
                "for acc >= 0.95, the ceiling, or impossible",
                "separations also stated in Mahalanobis-2 units for "
                "cross-reference with the framework's 1.0 threshold",
            ],
            "timing": [
                "FPS = 10, BASELINE_S = 12; window sweep over k",
            ],
            "device": [
                "8-channel recordings; replicate scatter measured on this "
                "array",
            ],
        },
        "headline": (
            "identifiability ceiling 0.779; impossible pairs listed at any "
            "window"),
    },
    {
        "key": "features_audit",
        "module": "bench_features_audit",
        "series": "W5",
        "title": "repository-wide feature audit (interoperability + grounding)",
        "claim_class": "F",
        "claim": (
            "of the features this repo actually computes, a transferable "
            "whitelist exists that is not calibration-bound, not "
            "protocol-confounded, and defined on the windows it is used on; "
            "the shared subset supports the interoperability claim"),
        "regimen": {
            "chemistry": ["real SmellNet offline recordings"],
            "environment": ["multi-rig / multi-recording offline corpus"],
            "protocol": [
                "audit every feature in two canonical vectors: framework "
                "187-dim (opensmell.mox.features.extract_all_framework_"
                "features) and primitives "
                "(features.physical_features.extract_physical_features, "
                "count = len(primitive_vector_names(6)))",
                "five classification axes: catalog_dimension (SDK CATALOG.md "
                "groups), category_10 (physics-semantic), invariance "
                "(INV_GAIN / INV_SCALE / DEV_BOUND / CALIBRATED / NONE), "
                "model_class (A fundamental | B derived | C empirical | D "
                "device-calib | E statistical), interop_verdict "
                "(transferable / device_bound / requires_calibration / "
                "protocol_confounded)",
                "robustness measured, not assumed: both extractors run on "
                "real SmellNet offline recordings; per-feature "
                "undefined/zero rates and dynamic range recorded",
                "transferable = not calibration-bound, not "
                "protocol-confounded, and defined on its windows",
            ],
            "timing": [
                "N_WINDOWS = 24 analysis windows from each recording",
            ],
            "device": [
                "SmellNet arrays (offline recordings)",
            ],
        },
        "headline": (
            "98 of 272 features (36%) transferable; '{pc}' placeholder bug "
            "left in the report verdict string"),
    },
]

_BY_KEY = {r["key"]: r for r in REGISTRY}
_BY_MODULE = {r["module"]: r for r in REGISTRY}


def registry():
    """Return the registry as an immutable-ish mapping keyed by benchmark key."""
    return dict(_BY_KEY)