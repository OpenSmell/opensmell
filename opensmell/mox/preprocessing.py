import numpy as np
import pandas as pd
from collections.abc import Mapping, Sequence
from typing import Optional

from ..csv import detect_time_column, is_context_column

SENSOR_NAMES = ["NO2", "C2H5OH", "VOC", "CO", "Alcohol", "LPG"]

MQ6_COLS = ["MQ135", "MQ3", "MQ6", "MQ7", "MQ4", "MQ8"]

WINDOW_SIZE = 100
WINDOW_STRIDE = 10


def rs_r0_normalize(arr: np.ndarray, r0_frac: float = 0.15) -> np.ndarray:
    n_baseline = max(5, int(len(arr) * r0_frac))
    r0 = np.median(arr[:n_baseline], axis=0, keepdims=True)
    r0 = np.where(r0 < 1.0, 1.0, r0)
    return (arr - r0) / r0


def _is_time_column(name) -> bool:
    return detect_time_column([str(name)]) is not None


def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.apply(pd.to_numeric, errors="coerce")
    if data.shape[1] == 0:
        raise ValueError("No sensor columns were selected.")
    values = data.to_numpy(dtype=np.float64)
    if not np.isfinite(values).any():
        raise ValueError(
            "The selected sensor columns contain no numeric values. "
            f"Found columns: {list(frame.columns)}."
        )
    return data


def detect_sensor_columns(df: pd.DataFrame):
    """Return every numeric, non-time, non-context column in ``df``.

    Channel count is intentionally not fixed at six. If the canonical six MOX
    names are all present they keep their historical order and any extra
    numeric sensors are appended; otherwise columns stay in CSV order so
    arbitrary names (``S1``…``S6``) work unchanged.
    """
    candidates = [
        c for c in df.columns
        if not _is_time_column(c) and not is_context_column(str(c))
    ]
    numeric = [
        c for c in candidates
        if pd.to_numeric(df[c], errors="coerce").notna().any()
    ]
    if not numeric:
        raise ValueError(
            "Could not detect any numeric sensor columns in CSV. "
            f"Found columns: {list(df.columns)}. "
            "Pass channels=[...] or channels={source: output} to select them "
            "explicitly."
        )
    if all(name in numeric for name in SENSOR_NAMES):
        extras = [c for c in numeric if c not in SENSOR_NAMES]
        return [name for name in SENSOR_NAMES if name in numeric] + extras
    return numeric


def _select_mapping(df: pd.DataFrame, channels: Mapping) -> pd.DataFrame:
    sources = list(channels.keys())
    outputs = list(channels.values())
    missing = [c for c in sources if c not in df.columns]
    if missing:
        raise ValueError(
            f"channels mapping refers to columns not in the CSV: {missing}. "
            f"Found columns: {list(df.columns)}."
        )
    if len(set(outputs)) != len(outputs):
        raise ValueError(
            f"channels mapping must use unique output names; got {outputs}."
        )
    data = _numeric_frame(df[sources])
    data.columns = outputs
    return data


def _select_sequence(df: pd.DataFrame, channels: Sequence) -> pd.DataFrame:
    if isinstance(channels, (str, bytes)):
        channels = [channels]
    selected = list(channels)
    missing = [c for c in selected if c not in df.columns]
    if missing:
        raise ValueError(
            f"Requested channel(s) not in the CSV: {missing}. "
            f"Found columns: {list(df.columns)}."
        )
    return _numeric_frame(df[selected])


def load_csv(filepath: str, sensor_map: Optional[dict] = None,
             channels=None):
    """Load numeric sensor channels from a CSV.

    ``channels`` may be:

    - omitted: detect every numeric, non-time, non-context column;
    - a sequence of column names, in the order features should use them;
    - a mapping ``{source_column: output_name}``, in output order.

    ``sensor_map`` is the legacy rename-then-detect form. The returned array is
    ``(n_rows, n_channels)`` and is not normalized; use ``rs_r0_normalize`` for
    that.
    """
    if sensor_map is not None and channels is not None:
        raise ValueError("Pass either sensor_map or channels, not both.")

    df = pd.read_csv(filepath)
    if sensor_map is not None:
        df = df.rename(columns=sensor_map)

    if channels is None:
        selected = detect_sensor_columns(df)
        return _numeric_frame(df[selected]).to_numpy(dtype=np.float64)

    if isinstance(channels, Mapping):
        data = _select_mapping(df, channels)
    elif isinstance(channels, Sequence):
        data = _select_sequence(df, channels)
    else:
        raise TypeError(
            "channels must be a sequence of column names or a "
            "{source: output} mapping."
        )
    return data.to_numpy(dtype=np.float64)


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
