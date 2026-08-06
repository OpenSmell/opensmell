"""Hardware sufficiency gate (§10.10 N→M limit): effective dims, implied
channels, train-time stash, and Warn-and-Proceed behavior.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import warnings

import numpy as np
import pytest

import opensmell
from opensmell.hardware import (
    HardwareInsufficiencyWarning,
    check_rig_sufficiency,
    effective_dims,
    effective_rank,
    implied_channels,
    min_effective_dimensions,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "cinnamon_6.csv")


def _synthetic_features(n_samples=120, n_features=187, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    y = (np.arange(n_samples) * n_classes // n_samples).astype(int)
    return X, y


class _DummyModel:
    """Minimal stand-in: requires more dims than any 6-channel rig provides."""

    min_effective_dimensions = 3.0

    def predict(self, X):
        return np.array(["cinnamon"] * len(X))

    def predict_proba(self, X):
        return np.array([[0.9]] * len(X))


class _FakeClassifier:
    """A model with no fitted feature count — forces the class-count fallback."""

    n_classes_ = 4


def test_effective_dims_mapping():
    assert effective_dims(1) == 0.5
    assert effective_dims(2) == 1.0
    assert effective_dims(3) == 1.5
    assert effective_dims(4) == 2.0
    assert effective_dims(5) == 2.5
    assert effective_dims(6) == 2.5


def test_implied_channels_canonical_counts():
    assert implied_channels(32) == 1
    assert implied_channels(91) == 3
    assert implied_channels(122) == 4
    assert implied_channels(154) == 5
    assert implied_channels(187) == 6


def test_implied_channels_invalid_counts():
    assert implied_channels(188) is None
    assert implied_channels(100) is None
    assert implied_channels(0) is None


def test_effective_rank():
    rng = np.random.default_rng(1)
    rank1 = rng.standard_normal((50, 10))
    rank1 = np.column_stack([rank1[:, 0]] * 10)
    assert effective_rank(rank1) == 1.0

    full = rng.standard_normal((100, 10))
    assert effective_rank(full) >= 8
    assert effective_rank(np.zeros((5, 3))) == 1.0


def test_train_stashes_min_effective_dimensions():
    X, y = _synthetic_features()
    model = opensmell.train(X, y, n_estimators=20)
    assert hasattr(model, "min_effective_dimensions")
    assert model.min_effective_dimensions == 2.5  # effective_dims(6)


def test_min_effective_dimensions_infers_training_rig():
    X, y = _synthetic_features(n_features=91)
    model = opensmell.train(X, y, n_estimators=20)
    delattr(model, "min_effective_dimensions")
    assert min_effective_dimensions(model) == 1.5  # implied 3 channels


def test_min_effective_dimensions_class_count_fallback():
    assert min_effective_dimensions(_FakeClassifier()) == pytest.approx(2.0)


def test_sufficient_rig_silent():
    X, y = _synthetic_features()
    model = opensmell.train(X, y, n_estimators=20)
    with warnings.catch_warnings():
        warnings.simplefilter("error", HardwareInsufficiencyWarning)
        assert check_rig_sufficiency(6, model) is True


def test_insufficient_rig_warns_and_proceeds():
    X, y = _synthetic_features()
    model = opensmell.train(X, y, n_estimators=20)
    with pytest.warns(HardwareInsufficiencyWarning):
        result = check_rig_sufficiency(2, model)
    assert result is True  # Warn-and-Proceed: never blocks


def test_process_warns_on_insufficient_rig():
    with pytest.warns(HardwareInsufficiencyWarning):
        r = opensmell.process(FIXTURE, _DummyModel())
    assert r.substance == "cinnamon"
    assert r.n_windows > 0


def test_process_silent_on_matching_rig():
    X, y = _synthetic_features()
    model = opensmell.train(X, y, n_estimators=20)
    with warnings.catch_warnings():
        warnings.simplefilter("error", HardwareInsufficiencyWarning)
        r = opensmell.process(FIXTURE, model)
    assert r.substance is not None
