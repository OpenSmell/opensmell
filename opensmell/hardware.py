"""Hardware sufficiency gate (§10.10 N→M limit).

Zero-shot feature transfer is only sanctioned when

    min_effective_dimensions(model) <= effective_dims(rig).

Same-family MOX channels covary (humidity and temperature are common-mode), so
the effective dimensionality of a rig grows far slower than its channel count.
Measured (§6, `research/calibration-experiments/sensor_theory_analysis.py`):
two same-family MOX ≈ 1, three ≈ 1.5–2, four ≈ 2–3. We use the conservative
lower bound so the gate errs on the side of warning.

This module is Warn-and-Proceed: an insufficient rig emits
`HardwareInsufficiencyWarning` and prediction continues. It never silently pads
missing channels with training-set means (the §8.8 `expand_channels` caveat).
"""

import math
import warnings

import numpy as np


class HardwareInsufficiencyWarning(Warning):
    """A device provides fewer effective dimensions than the target model needs.

    Zero-shot transfer across this gap is not expected to work; the prediction
    proceeds but is unsupported (§10.10, §9.4).
    """


# Per-channel feature block sizes in the canonical framework extractor
# (`feature_names()`): 6 device-agnostic + 4 absolute + 4 temporal + 4 health +
# 3 hardware + 1 saturation + 6 decay = 28 per channel, C(c, 2) selectivity
# ratios, plus 4 globals. This inverts feature count back to channel count.
_FEATURES_PER_CHANNEL = 28
_GLOBAL_FEATURES = 4


def effective_dims(n_channels: int) -> float:
    """Effective dimensionality of a same-family MOX rig.

    Empirical (§6, `sensor_theory_analysis.py`): two same-family MOX ≈ 1,
    three ≈ 1.5–2, four ≈ 2–3. Lower bounds are used so the gate warns early.
    A single channel carries ≈0.5 dims; five or more same-family channels
    saturate near 2.5 because they covary, well below the ~4–5 dims a SmellNet
    latent space needs.
    """
    n = int(n_channels)
    if n <= 1:
        return 0.5
    if n == 2:
        return 1.0
    if n == 3:
        return 1.5
    if n == 4:
        return 2.0
    return 2.5


def implied_channels(n_features: int):
    """Channel count implied by a canonical framework feature count.

    Inverts ``28c + c(c-1)/2 + 4 = n_features``. Returns the integer channel
    count for a valid canonical count, else ``None`` (e.g. hand-crafted feature
    sets, which are not subject to the hardware gate).
    """
    if n_features < 32:
        return None
    disc = 55 * 55 + 4 * (2 * n_features - 8)
    root = math.isqrt(disc)
    if root * root != disc:
        return None
    c = (-55 + root) // 2
    if c < 1:
        return None
    if _FEATURES_PER_CHANNEL * c + c * (c - 1) // 2 + _GLOBAL_FEATURES != n_features:
        return None
    return c


def effective_rank(X: np.ndarray, variance: float = 0.95) -> float:
    """Number of singular directions of X explaining ``variance`` of energy.

    Used to size ``min_effective_dimensions`` for models whose feature set is
    not a canonical framework count. Not used by the legacy predict gate.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        return 1.0
    centered = X - X.mean(axis=0)
    s = np.linalg.svd(centered, compute_uv=False)
    total = float(np.sum(s ** 2))
    if total <= 0:
        return 1.0
    cum = np.cumsum(s ** 2 / total)
    return float(np.searchsorted(cum, variance) + 1)


def min_effective_dimensions(model) -> float:
    """Minimum effective dimensions the model requires to transfer.

    Resolution order:
    1. ``model.min_effective_dimensions`` when present (set by ``train()``).
    2. The effective dims of the training rig, inferred from the canonical
       feature count the model was fitted on (``n_features_in_``).
    3. A documented heuristic from the class count: ``max(1, log2(k))``,
       capped at the ~4-dim SmellNet latent budget.
    """
    stored = getattr(model, "min_effective_dimensions", None)
    if stored is not None:
        return float(stored)

    n_features = getattr(model, "n_features_in_", None)
    if n_features is not None:
        c = implied_channels(int(n_features))
        if c is not None:
            return effective_dims(c)

    clf = getattr(model, "named_steps", {}).get("clf", model)
    n_classes = getattr(clf, "n_classes_", None)
    if n_classes is None:
        return 1.0
    return float(min(4.0, max(1.0, math.log2(int(n_classes)))))


def check_rig_sufficiency(n_channels: int, model, *, warn: bool = True) -> bool:
    """Warn-and-Proceed hardware gate (§10.10 N→M limit).

    Returns ``True`` when the rig's effective dims meet the model's requirement.
    When they do not, emits ``HardwareInsufficiencyWarning`` and still returns
    ``True`` so prediction proceeds — never silently pads or blocks.
    """
    available = effective_dims(n_channels)
    required = min_effective_dimensions(model)
    if available >= required:
        return True
    if warn:
        warnings.warn(
            f"HardwareInsufficiencyWarning: rig has {int(n_channels)} channel(s) "
            f"(≈{available:.2g} effective dims) but the model requires ≈{required:.2g} "
            f"effective dims. Zero-shot feature transfer needs "
            f"min_effective_dimensions <= effective_dims(rig) (§10.10). Prediction "
            f"proceeds unsupported; do not pad missing channels with training-set means.",
            HardwareInsufficiencyWarning,
            stacklevel=3,
        )
    return True
