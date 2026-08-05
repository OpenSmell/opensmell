"""opensmell — digital olfaction SDK.

v3 modular framework: sensor-agnostic interfaces at the top level
(`io`, `csv`, `normalize`, `quality`, `features`) with sensor-specific
implementations in `opensmell.mox` (and future `opensmell.miris`,
`opensmell.electrochemical`). The MOX thermodynamic feasibility chain lives at
`opensmell.mox.smellability`.

Legacy v2 CSV-based API (`process`, `train`, `predict`, `extract_features`,
`load_recording`, `SmellResult`) is preserved for backwards compatibility.
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from . import features as _features
from .mox.preprocessing import load_csv, rs_r0_normalize, segment
from .result import SmellResult

# --- New v3 sensor-agnostic API ---
from .csv import guess_sensor_type, parse_csv
from .features import process_mox, run_processor
from .io import (
    build_osmell,
    csv_from_file,
    default_file_name,
    parse_osmell,
    parse_osmell_file,
    write_osmell,
)
from .quality import compute_quality
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


def load_recording(filepath: str) -> np.ndarray:
    raw = load_csv(filepath)
    return rs_r0_normalize(raw)


def extract_features(filepath: str) -> tuple:
    normed = load_recording(filepath)
    segments = segment(normed)
    all_features = []
    for seg in segments:
        feats = _features.extract_all_framework_features(seg)
        vals, _ = _feature_vector(feats)
        all_features.append(vals)
    arr = np.array(all_features)
    fnames = _features.feature_names()
    return arr, fnames


def feature_names() -> list:
    """Names of the MOX framework features (in extraction order)."""
    return _features.feature_names()


def process(filepath: str, model: Pipeline = None) -> SmellResult:
    features_arr, fnames = extract_features(filepath)
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
    return model


def predict(filepath: str, model: Pipeline) -> SmellResult:
    return process(filepath, model=model)


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
    # v3 sensor-agnostic API
    "parse_csv",
    "guess_sensor_type",
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
]
