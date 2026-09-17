"""Claim taxonomy A-F: a benchmark verifies claims, not data or algorithms.

Each claim class names one layer of the measurement chain whose story a
benchmark must (dis)prove.  The primary class per benchmark is recorded in the
regimen registry.
"""

CLAIM_CLASSES = {
    "A": {
        "name": "fundamental mechanism",
        "chain": "physics model",
        "scope": (
            "the physics model itself is sane and internally consistent: "
            "self-check chains, theoretical exponents, additivity oracles"),
        "examples": ["physics_selfcheck"],
    },
    "B": {
        "name": "transduction",
        "chain": "device behaviour",
        "scope": (
            "observable behaviour of a real sensor array: memory residue, "
            "kinetics, recovery timescales, drift magnitude, response "
            "exponents and their movement with environment"),
        "examples": [
            "turbulent_mixtures", "environment_robustness", "b_dynamic",
            "mixture_generalization", "sensor_memory", "dynamic_memory",
        ],
    },
    "C": {
        "name": "algorithm / abstract feature",
        "chain": "pipeline output",
        "scope": (
            "a pipeline computes what it claims on real data: anomaly axes "
            "against ground truth, feature computability and definition on "
            "typical windows, detector false-alarm decomposition"),
        "examples": ["anomaly_v2", "detector_memory_fp"],
    },
    "D": {
        "name": "statistical / corpus",
        "chain": "population property",
        "scope": (
            "population-level properties estimated over a corpus with stated "
            "error and bias: drift trajectory smoothness, exponent-vs-"
            "humidity slopes, identifiability accuracy ceilings, reconstructed "
            "separability margins from archives"),
        "examples": ["drift_trajectory", "separability_margin", "identifiability"],
    },
    "E": {
        "name": "calibration constant",
        "chain": "device constants",
        "scope": (
            "datasheet/device constants are trustworthy and converted "
            "consistently: sensors.json (a, b) sanity, theory band "
            "agreement, MQUnified conversion"),
        "examples": ["datasheet_constants"],
    },
    "F": {
        "name": "interoperability",
        "chain": "transfer across contexts",
        "scope": (
            "transfer across devices, rigs, sessions and features: physical "
            "primitive stability, session anchoring and environment "
            "residualization, fleet calibration, minimal array specs, "
            "feature transferability whitelists"),
        "examples": [
            "uci_drift", "cross_device_primitives", "session_anchor",
            "fleet_calibration", "minimal_array", "features_audit",
        ],
    },
}

CLAIM_TAXONOMY_VERSION = "1.0"