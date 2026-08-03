"""Legacy v2 CSV-based API: shape contract and determinism.

The framework feature extraction is intentionally heavy (multi-exponential curve
fitting per segment x channel), so this suite keeps legacy-path coverage to a
single `process()` call on one fixture.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import opensmell

FIXTURE = os.path.join(os.path.dirname(__file__), "cinnamon_6.csv")


def test_process_shapes_and_no_fabrication():
    r = opensmell.process(FIXTURE)
    assert r.features.shape == (187,), f"Expected 187 features, got {r.features.shape}"
    assert r.chemoprint.shape == (29,), f"Expected chemoprint (29,), got {r.chemoprint.shape}"
    assert r.n_windows > 0, "Expected at least one analysis window"
    assert len(r.feature_names) == 187
    assert r.substance is None, "Without a model the SDK must not fabricate a substance"
    assert r.confidence is None, "Without a model the SDK must not fabricate confidence"


def test_extract_features_finite():
    arr, fnames = opensmell.extract_features(FIXTURE)
    assert arr.shape[1] == 187
    assert len(fnames) == 187
    assert np.all(np.isfinite(arr)), "Framework features must be finite"


def test_feature_names_reproducible():
    assert len(opensmell.feature_names()) == 187


def test_determinism():
    a = opensmell.process(FIXTURE)
    b = opensmell.process(FIXTURE)
    assert np.allclose(a.features, b.features), "process() must be deterministic"
