import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

SENSOR_NAMES = ["NO2", "C2H5OH", "VOC", "CO", "Alcohol", "LPG"]
SEGMENT_LEN = 100
STRIDE = 50

SENSOR_MEAN = np.array([108.76061248779297, 151.33633422851562, 216.8444366455078,
                        799.3080444335938, 3.354598045349121, 31.686960220336914])
SENSOR_STD = np.array([126.42085266113281, 152.64385986328125, 205.7685546875,
                       63.45093536376953, 3.118058919906616, 39.982574462890625])


def rs_r0_normalize(arr: np.ndarray, r0_frac: float = 0.15) -> np.ndarray:
    n_baseline = max(5, int(len(arr) * r0_frac))
    r0 = np.median(arr[:n_baseline], axis=0, keepdims=True)
    r0 = np.where(r0 < 1.0, 1.0, r0)
    return (arr - r0) / r0


def detect_sensor_columns(df: pd.DataFrame):
    cols = []
    for expected in SENSOR_NAMES:
        found = [c for c in df.columns if c.lower() == expected.lower()]
        cols.append(found[0] if found else None)
    if any(c is None for c in cols):
        fallback = [c for c in df.columns if c.lower().startswith("sensor_")]
        if len(fallback) >= 6:
            return fallback[:6]
    return cols


def load_csv(filepath: str, sensor_map: Optional[dict] = None):
    df = pd.read_csv(filepath)
    if sensor_map is not None:
        df = df.rename(columns=sensor_map)
    cols = detect_sensor_columns(df)
    if any(c is None for c in cols):
        missing = [SENSOR_NAMES[i] for i, c in enumerate(cols) if c is None]
        raise ValueError(
            f"Could not detect sensor columns in CSV. "
            f"Missing after mapping: {missing}. "
            f"Expected one of {SENSOR_NAMES}. "
            f"Found columns: {list(df.columns)}. "
            f"If using non-standard sensor names, provide a sensor_map "
            f"dict mapping your column names to the standard names."
        )
    raw = df[cols].values.astype(np.float32)
    return raw


def expand_channels(arr: np.ndarray, mapping: Optional[list] = None, n_target: int = 6) -> np.ndarray:
    """Map N-channel sensor data to M-channel encoder input.

    Args:
        arr: (T, N) raw sensor readings.
        mapping: List of (from_ch, to_ch) pairs. If None, uses default
                 3-channel mapping: [(0,0), (1,1), (0,2), (2,3), (1,4)].
        n_target: Number of output channels (default 6 for SmellNet encoder).

    Returns:
        (T, n_target) expanded array.

    Examples:
        # 3 sensors: MQ-135, MQ-3, MQ-7 -> 6 encoder channels
        expanded = expand_channels(arr_3ch)

        # 4 sensors: MQ-135, MQ-3, MQ-6, MQ-7 -> 6 channels
        expanded = expand_channels(arr_4ch, mapping=[(0,0), (1,1), (2,2), (3,3), (1,4)])

        # 6 sensors: direct passthrough
        expanded = expand_channels(arr_6ch, mapping=[(i,i) for i in range(6)])
    """
    if mapping is None:
        mapping = [(0, 0), (1, 1), (0, 2), (2, 3), (1, 4)]
    out = np.zeros((arr.shape[0], n_target), dtype=np.float32)
    for src, dst in mapping:
        if src < arr.shape[1] and dst < n_target:
            out[:, dst] = arr[:, src]
    # Fill unmapped channels with their training-set means
    unmapped_means = {5: SENSOR_MEAN[5]}  # LPG mean
    for ch, val in unmapped_means.items():
        if ch < n_target and np.all(out[:, ch] == 0):
            out[:, ch] = val
    return out


def per_recording_zscore(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-recording z-score normalization.

    Device-agnostic: removes per-channel mean and variance from each
    recording independently. Use instead of global z-score when
    the recording's sensor baseline is unknown.

    Args:
        arr: (T, N) sensor readings.
        eps: Small constant to avoid division by zero.

    Returns:
        (T, N) normalized array.
    """
    return (arr - arr.mean(axis=0, keepdims=True)) / (arr.std(axis=0, keepdims=True) + eps)


def segment(sensor_array: np.ndarray):
    N = sensor_array.shape[0]
    if N >= SEGMENT_LEN:
        segments = [
            sensor_array[i : i + SEGMENT_LEN]
            for i in range(0, N - SEGMENT_LEN + 1, STRIDE)
        ]
    else:
        pad_width = ((0, SEGMENT_LEN - N), (0, 0))
        segments = [np.pad(sensor_array, pad_width, mode="edge")]
    return np.stack(segments)


def normalize(segments: np.ndarray):
    return (segments - SENSOR_MEAN) / SENSOR_STD


def segment_and_normalize(sensor_array: np.ndarray):
    segs = segment(sensor_array)
    return normalize(segs)


def export_for_contribution(filepath: str, result, output_dir: str = "./"):
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(filepath).stem
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    metadata = {
        "format_version": "1.0",
        "encoder_version": "v1",
        "substance": result.substance,
        "confidence": float(result.confidence),
        "timestamp": timestamp,
        "chemoprint": result.chemoprint.tolist(),
        "latent": result.latent.tolist(),
    }
    meta_path = os.path.join(output_dir, f"{stem}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    import shutil
    shutil.copy2(filepath, os.path.join(output_dir, os.path.basename(filepath)))
    return meta_path
