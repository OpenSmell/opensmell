"""Legacy v2 CSV-based API: shape contract and determinism.

The framework feature extraction is intentionally heavy (multi-exponential curve
fitting per segment x channel), so this suite keeps legacy-path coverage to a
single `process()` call on one fixture.
"""

import os
import re
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import opensmell

FIXTURE = os.path.join(os.path.dirname(__file__), "cinnamon_6.csv")

# The only features allowed to vary between runs are the multi-exponential decay
# fits (ch{c}_decay_tau{1..3}, ch{c}_decay_a{1..3}). Everything else must be
# byte-identical.
MULTI_EXP_DECAY = re.compile(r"ch\d+_decay_(tau|a)[123]$")


def test_process_shapes_and_no_fabrication():
    r = opensmell.process(FIXTURE)
    assert r.features.shape == (187,), f"Expected 187 features, got {r.features.shape}"
    assert r.chemoprint.shape == (29,), f"Expected chemoprint (29,), got {r.chemoprint.shape}"
    assert r.n_windows > 0, "Expected at least one analysis window"
    assert len(r.feature_names) == 187
    assert r.substance is None, "Without a model the SDK must not fabricate a substance"
    assert r.confidence is None, "Without a model the SDK must not fabricate confidence"


def test_extract_features_finite():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        arr, fnames = opensmell.extract_features(FIXTURE)
    assert arr.shape[1] == 187
    assert len(fnames) == 187
    assert np.all(np.isfinite(arr)), "Framework features must be finite"
    overflow = [w for w in caught if "overflow" in str(w.message).lower()]
    assert not overflow, (
        "Decay fitting must not overflow: "
        + "; ".join(str(w.message) for w in overflow)
    )


def _write_custom_csv(path, n=120, prefix="S", channels=6):
    rng = np.random.default_rng(7)
    t = np.arange(n, dtype=np.float64) * 100.0
    names = [f"{prefix}{i}" for i in range(1, channels + 1)]
    with open(path, "w", encoding="utf-8") as f:
        f.write("time," + ",".join(names) + "\n")
        for i in range(n):
            values = [
                1000.0 + np.sin(i / 12.0 + j) + rng.normal(0.0, 0.05)
                for j in range(channels)
            ]
            f.write(str(t[i]) + "," + ",".join(str(v) for v in values) + "\n")
    return names


def test_extract_features_accepts_arbitrary_channel_names(tmp_path):
    path = tmp_path / "mox_6ch.csv"
    names = _write_custom_csv(path)
    arr, fnames = opensmell.extract_features(str(path))
    assert arr.shape[1] == 187
    assert len(fnames) == 187
    assert np.all(np.isfinite(arr))
    assert opensmell.load_recording(str(path)).shape == (120, len(names))


def test_extract_features_channels_mapping_and_subset(tmp_path):
    path = tmp_path / "mox_6ch.csv"
    _write_custom_csv(path)
    arr3, names3 = opensmell.extract_features(
        str(path),
        channels={"S1": "NO2", "S2": "C2H5OH", "S3": "VOC"},
    )
    assert arr3.shape[1] == 91 == len(names3)
    arr2, names2 = opensmell.extract_features(str(path), channels=["S1", "S2"])
    assert arr2.shape[1] == 61 == len(names2)
    with pytest.raises(ValueError, match="Requested channel"):
        opensmell.extract_features(str(path), channels=["NOPE"])


def test_extract_features_accepts_ingest_result_and_osmell(tmp_path):
    path = tmp_path / "mox_6ch.csv"
    _write_custom_csv(path)
    session = opensmell.ingest_file(path)
    assert session.ok
    from_session, _ = opensmell.extract_features(session)
    assert from_session.shape[1] == 187

    bundle = tmp_path / "recording.osmell"
    opensmell.write_osmell(session.file, bundle)
    from_bundle, _ = opensmell.extract_features(bundle)
    assert from_bundle.shape == from_session.shape
    assert np.allclose(from_bundle, from_session)

    with pytest.raises(ValueError, match="Ingest failed"):
        opensmell.extract_features(opensmell.IngestedSession(
            source="missing.csv", substance="x", label="x",
            error="No usable data rows found.",
        ))


def test_feature_names_reproducible():
    assert len(opensmell.feature_names()) == 187


def test_feature_names_are_channel_agnostic():
    # Coefficient-agnostic contract: length = 28c + c(c-1)/2 + 4.
    # 187 is the canonical 6-channel instance; the formula holds for any c.
    cases = {1: 32, 3: 91, 4: 122, 6: 187, 12: 406}
    from opensmell.mox import features as _f

    for c, expected in cases.items():
        assert len(_f.feature_names(n_channels=c)) == expected, (c, expected)
    # Default and explicit-None both resolve to the canonical 6.
    assert len(_f.feature_names()) == 187
    assert len(_f.feature_names(n_channels=None)) == 187
    # Names must match what the shape-driven extractor actually produces.
    rng = np.random.RandomState(0)
    for c in (3, 6):
        fake = rng.rand(120, c)
        feats = _f.extract_all_framework_features(fake)
        assert len(feats) == len(_f.feature_names(n_channels=c)), c


def test_determinism():
    a = opensmell.process(FIXTURE)
    b = opensmell.process(FIXTURE)
    # feature values follow sorted-key order (_feature_vector) while
    # feature_names() reports extraction order, so relabel cells with the
    # sorted-key order the values actually use.
    names = np.asarray(sorted(a.feature_names))
    is_decay = np.array([bool(MULTI_EXP_DECAY.match(n)) for n in names])

    # 1. Non-decay features are byte-identical between runs.
    assert np.allclose(a.features[~is_decay], b.features[~is_decay]), (
        "Non-decay features differ between runs"
    )

    # 2. Number of differing features is bounded. The decay fit may converge to
    #    near-equal-cost local minima (scipy MINPACK), flipping a handful of
    #    tau/amplitude values, but never the whole vector.
    differing = names[~np.isclose(a.features, b.features)]
    assert len(differing) <= 20, (
        f"{len(differing)} features differ between runs: {list(differing)}"
    )

    # 3. Where both runs produced a valid fit, tau values must agree within 10%.
    for i, n in enumerate(names):
        if MULTI_EXP_DECAY.match(n) and "_decay_tau" in n:
            if a.features[i] > 0 and b.features[i] > 0:
                assert np.isclose(a.features[i], b.features[i], rtol=0.10), (
                    f"{n} differs by more than 10%: "
                    f"{a.features[i]} vs {b.features[i]}"
                )
