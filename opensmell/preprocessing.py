import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

SENSOR_NAMES = ["NO2", "C2H5OH", "VOC", "CO", "Alcohol", "LPG"]
SEGMENT_LEN = 100
STRIDE = 50

SENSOR_MEAN = np.array([108.76061248779297, 151.33633422851562, 216.8444366455078,
                        799.3080444335938, 3.354598045349121, 31.686960220336914])
SENSOR_STD = np.array([126.42085266113281, 152.64385986328125, 205.7685546875,
                       63.45093536376953, 3.118058919906616, 39.982574462890625])


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


def load_csv(filepath: str):
    df = pd.read_csv(filepath)
    cols = detect_sensor_columns(df)
    if any(c is None for c in cols):
        raise ValueError(
            f"Could not detect sensor columns in CSV. "
            f"Expected one of {SENSOR_NAMES} or sensor_N prefix. "
            f"Found columns: {list(df.columns)}"
        )
    raw = df[cols].values.astype(np.float32)
    return raw


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
