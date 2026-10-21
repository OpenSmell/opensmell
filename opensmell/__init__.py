"""opensmell — digital olfaction SDK.

v3 modular framework: modality-dispatched interfaces at the top level
(`io`, `csv`, `normalize`, `quality`, `features`, `profile`) with
sensor-specific implementations in `opensmell.mox`. The MOX feature and quality
paths are implemented; `opensmell.miris` and `opensmell.electrochemical` are
explicit stubs, and `inspect_source()` reports them as unsupported rather than
silently applying MOX math. The MOX thermodynamic feasibility chain lives at
`opensmell.mox.smellability`.

Legacy v2 CSV-based API (`process`, `train`, `predict`, `extract_features`,
`load_recording`, `SmellResult`) is preserved for backwards compatibility.

The feasibility chain is re-exported at the top level as `opensmell.smellability`
so `resolve_and_run`, `chemical_from_smiles`, and the verdicts are reachable
without importing the internals.
"""

import numpy as np
from collections.abc import Mapping, Sequence
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from . import features as _features
from .calibration import (
    CalibrationError,
    build_calibration_payload,
    calibrate_precise,
    calibrate_quick,
    concentration_series,
    fit_power_law,
    invert_concentration,
    loocv_power_law,
    normed_to_rr,
    two_point_calibration,
)
from .hardware import (
    HardwareInsufficiencyWarning,
    check_rig_sufficiency,
    effective_dims,
    implied_channels,
    min_effective_dimensions,
)
from .constants import (
    UnknownGasError,
    UnknownSensorError,
    list_sensors,
    sensor_gases,
)
from .mox.preprocessing import load_csv, rs_r0_normalize, segment
from .result import SmellResult

# --- MOX thermodynamic feasibility chain (Smellability) ---
from .mox import smellability

# Register the re-export as a real dotted path so `import opensmell.smellability`
# (and `from opensmell.smellability import ...`) works, not just attribute access.
import sys as _sys

_sys.modules[__name__ + ".smellability"] = smellability
del _sys

# --- New v3 modality-aware API ---
from .csv import guess_sensor_type, parse_csv
from .features import process_mox, run_processor
from .ingest import (
    IngestedCollection,
    IngestedSession,
    build_osmell_file,
    ingest_file,
    ingest_folder,
)
from .io import (
    build_osmell,
    csv_from_file,
    default_file_name,
    parse_osmell,
    parse_osmell_file,
    write_osmell,
)
from .profile import (
    SensorProfile,
    UnsupportedSensorTypeError,
    inspect_source,
)
from .quality import compute_quality
from .presets import (
    AnomalyThresholds,
    BaselineProtocol,
    FalseAlarmBudget,
    Preset,
    PresetError,
    PresetNotFoundError,
    SensorSpec,
    list_presets,
    load_all_presets,
    load_preset,
    presets_dir,
)
from .types import (
    OSMELL_FORMAT_VERSION,
    CalibrationDescriptor,
    ChannelDescriptor,
    ChannelStats,
    OsmellFile,
    OsmellManifest,
    ParsedSample,
    QualityReport,
    SensorDescriptor,
    SessionDescriptor,
    SessionEvent,
)


def _feature_vector(feature_dict: dict) -> tuple:
    keys = sorted(feature_dict.keys())
    values = [feature_dict[k] for k in keys]
    return np.array(values, dtype=np.float32), keys


def _features_from_normed(normed: np.ndarray, n_channels: int = None) -> tuple:
    if n_channels is None:
        n_channels = normed.shape[1] if normed.ndim == 2 else 1
    segments = segment(normed)
    all_features = []
    for seg in segments:
        feats = _features.extract_all_framework_features(seg)
        vals, _ = _feature_vector(feats)
        all_features.append(vals)
    arr = np.array(all_features)
    fnames = _features.feature_names(n_channels=n_channels)
    return arr, fnames


def _array_from_osmell(file: OsmellFile, channels=None) -> np.ndarray:
    """Select channels from an OsmellFile in requested (or manifest) order."""
    available = [c.id for c in file.manifest.sensor.channels]
    if not available:
        raise ValueError("The .osmell manifest declares no sensor channels.")

    if channels is None:
        selected = available
    elif isinstance(channels, Mapping):
        missing = [c for c in channels if c not in available]
        if missing:
            raise ValueError(
                f"channels mapping refers to channels not in the .osmell: "
                f"{missing}. Available channels: {available}."
            )
        selected = list(channels.keys())
    elif isinstance(channels, Sequence) and not isinstance(channels, (str, bytes)):
        selected = list(channels)
    elif isinstance(channels, (str, bytes)):
        selected = [channels]
    else:
        raise TypeError(
            "channels must be a sequence of channel ids or a "
            "{source: output} mapping."
        )

    missing = [c for c in selected if c not in available]
    if missing:
        raise ValueError(
            f"Requested channel(s) not in the .osmell: {missing}. "
            f"Available channels: {available}."
        )
    if not selected:
        raise ValueError("No channels selected.")

    n = len(file.time)
    columns = []
    for cid in selected:
        values = file.data.get(cid)
        if values is None:
            raise ValueError(f"Channel {cid!r} has no data in the .osmell.")
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape[0] < n:
            arr = np.pad(arr, (0, n - arr.shape[0]), constant_values=np.nan)
        columns.append(arr[:n])
    return np.column_stack(columns)


def _array_from_source(source, channels=None) -> np.ndarray:
    """Raw (un-normalized) sensor array from a path, OsmellFile or ingest result."""
    if isinstance(source, OsmellFile):
        return _array_from_osmell(source, channels)
    if isinstance(source, IngestedSession):
        if not source.ok or source.file is None:
            detail = source.error or "; ".join(source.warnings) or "unknown error"
            raise ValueError(f"Ingest failed for {source.source}: {detail}")
        return _array_from_osmell(source.file, channels)
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() == ".osmell":
            return _array_from_osmell(parse_osmell_file(path), channels)
        return load_csv(path, channels=channels)
    raise TypeError(
        "source must be a CSV/.osmell path, an OsmellFile, or an "
        f"IngestedSession; got {type(source).__name__}."
    )


def _require_mox_features(source, channels=None) -> SensorProfile:
    """Refuse to apply the MOX feature framework to an unsupported family."""
    profile = inspect_source(source, channels=channels)
    if not profile.supported:
        raise UnsupportedSensorTypeError(
            f"{profile.source}: sensor type {profile.sensor_type!r} has no "
            "feature backend. The implemented framework covers MOX/chemiresistive "
            "arrays (known models or MOX-compatible numeric channels). "
            "Use opensmell.run_processor() for raw normalized data, or "
            "opensmell.inspect_source() to inspect the detected profile."
        )
    return profile


def load_recording(source, channels=None) -> np.ndarray:
    """Load and Rs/R0-normalize a recording.

    ``source`` may be a CSV or ``.osmell`` path, an ``OsmellFile``, or an
    ``IngestedSession`` returned by :func:`ingest_file`. Channel selection is
    explicit when needed:

    - omit ``channels`` to use every detected sensor channel;
    - pass a sequence such as ``["S1", "S2", "S3"]``;
    - pass a mapping such as ``{"S1": "ch0", "S2": "ch1"}``.

    This is the shared loader behind ``extract_features``, ``process`` and
    ``predict``, so custom channel names no longer have to be renamed to the
    six historical MOX columns.
    """
    raw = _array_from_source(source, channels=channels)
    return rs_r0_normalize(raw)


def extract_features(source, channels=None) -> tuple:
    """Extract the MOX framework feature matrix from a recording.

    ``source`` accepts a CSV/.osmell path, an ``OsmellFile``, or the
    ``IngestedSession`` returned by ``ingest_file``. ``channels`` selects
    arbitrary CSV columns (sequence) or renames them (mapping). The feature
    count follows the loaded channel count: ``28c + c(c-1)/2 + 4``.

    The implemented framework is MOX/chemiresistive. Unknown numeric arrays are
    treated as MOX-compatible, but declared MIRIS/electrochemical recordings
    raise :class:`UnsupportedSensorTypeError` instead of silently inheriting MOX
    feature math. Use :func:`inspect_source` to see the detected family.
    """
    _require_mox_features(source, channels=channels)
    normed = load_recording(source, channels=channels)
    return _features_from_normed(normed, n_channels=normed.shape[1])


def feature_names(n_channels=None) -> list:
    """Names of the MOX framework features (in extraction order).

    Length is a function of channel count (``28·c + c(c−1)/2 + 4``); pass
    ``n_channels`` to match a non-6 rig, or omit for the canonical 6.
    """
    return _features.feature_names(n_channels=n_channels)


def process(source, model: Pipeline = None, channels=None) -> SmellResult:
    _require_mox_features(source, channels=channels)
    normed = load_recording(source, channels=channels)
    if model is not None:
        check_rig_sufficiency(normed.shape[1], model)
    features_arr, fnames = _features_from_normed(normed)
    if features_arr.shape[0] == 0:
        return SmellResult(features=np.array([]), feature_names=fnames, n_windows=0)
    avg_features = features_arr.mean(axis=0)
    if model is not None:
        pred = model.predict([avg_features])[0]
        proba = model.predict_proba([avg_features]).max()
        warning = ""
        if proba < 0.5:
            warning = "Low confidence"
        elif proba < 0.7:
            warning = "Moderate confidence"
        return SmellResult(
            substance=str(pred),
            confidence=float(proba),
            warning=warning,
            features=avg_features,
            feature_names=fnames,
            n_windows=features_arr.shape[0],
        )
    return SmellResult(
        features=avg_features,
        feature_names=fnames,
        n_windows=features_arr.shape[0],
    )


def train(X: np.ndarray, y: np.ndarray, n_estimators: int = 200) -> Pipeline:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    model.fit(X, y)
    n_ch = implied_channels(X.shape[1])
    if n_ch is not None:
        model.min_effective_dimensions = effective_dims(n_ch)
    return model


def predict(source, model: Pipeline, channels=None) -> SmellResult:
    return process(source, model=model, channels=channels)


# Backwards-compat alias for the .osmell loader.
load_osmell = parse_osmell_file

__all__ = [
    "extract_features",
    "feature_names",
    "process",
    "train",
    "predict",
    "load_recording",
    "SmellResult",
    # Reference-point calibration (§4.6, §10.10)
    "CalibrationError",
    "two_point_calibration",
    "fit_power_law",
    "invert_concentration",
    "loocv_power_law",
    "build_calibration_payload",
    "concentration_series",
    "normed_to_rr",
    "calibrate_quick",
    "calibrate_precise",
    # Sensor constants / discoverability
    "UnknownSensorError",
    "UnknownGasError",
    "list_sensors",
    "sensor_gases",
    # Hardware sufficiency gate (§10.10 N→M limit)
    "HardwareInsufficiencyWarning",
    "check_rig_sufficiency",
    "effective_dims",
    "min_effective_dimensions",
    "implied_channels",
    # v3 modality-aware API
    "parse_csv",
    "guess_sensor_type",
    "SensorProfile",
    "UnsupportedSensorTypeError",
    "inspect_source",
    "ingest_file",
    "ingest_folder",
    "build_osmell_file",
    "IngestedSession",
    "IngestedCollection",
    "parse_osmell",
    "parse_osmell_file",
    "load_osmell",
    "build_osmell",
    "write_osmell",
    "csv_from_file",
    "default_file_name",
    "compute_quality",
    "run_processor",
    "process_mox",
    # MOX thermodynamic feasibility chain (Smellability)
    "smellability",
    "OSMELL_FORMAT_VERSION",
    "OsmellFile",
    "OsmellManifest",
    "SensorDescriptor",
    "SessionDescriptor",
    "ChannelDescriptor",
    "SessionEvent",
    "ParsedSample",
    "ChannelStats",
    "QualityReport",
    "CalibrationDescriptor",
    # Configuration presets (opensmell/presets/*.yaml)
    "Preset",
    "SensorSpec",
    "BaselineProtocol",
    "AnomalyThresholds",
    "FalseAlarmBudget",
    "PresetError",
    "PresetNotFoundError",
    "presets_dir",
    "list_presets",
    "load_preset",
    "load_all_presets",
]
