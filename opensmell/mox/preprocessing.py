import numpy as np
import pandas as pd
from typing import Optional

SENSOR_NAMES = ["NO2", "C2H5OH", "VOC", "CO", "Alcohol", "LPG"]

MQ6_COLS = ["MQ135", "MQ3", "MQ6", "MQ7", "MQ4", "MQ8"]

WINDOW_SIZE = 100
WINDOW_STRIDE = 10


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
        mq = [c for c in df.columns if c.lower() in [m.lower() for m in MQ6_COLS]]
        if len(mq) >= 6:
            return mq[:6]
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
        )
    return df[cols].values.astype(np.float64)


def expand_channels(arr: np.ndarray, n_target: int = 6, mapping=None) -> np.ndarray:
    """Expand an N-channel array to ``n_target`` channels.

    Without ``mapping`` this pads/truncates channel-wise. The legacy firmware
    contract passes ``mapping`` as a list of ``(source, target)`` pairs (e.g.
    ``FW_MAPPING``) so a 3-sensor rig's columns are copied into the 6-channel
    layout the framework expects.
    """
    out = np.zeros((arr.shape[0], n_target), dtype=np.float64)
    if mapping is not None:
        for source, target in mapping:
            if source < arr.shape[1] and target < n_target:
                out[:, target] = arr[:, source]
        return out
    for i in range(min(arr.shape[1], n_target)):
        out[:, i] = arr[:, i]
    return out


def segment(sensor_array: np.ndarray, window_size: int = WINDOW_SIZE,
            stride: int = WINDOW_STRIDE):
    N = sensor_array.shape[0]
    if N >= window_size:
        segments = [
            sensor_array[i: i + window_size]
            for i in range(0, N - window_size + 1, stride)
        ]
    else:
        pad_width = ((0, window_size - N), (0, 0))
        segments = [np.pad(sensor_array, pad_width, mode="edge")]
    return np.stack(segments)
