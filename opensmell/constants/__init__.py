"""Offline sensor constants for MOX power-law response (§4.6, §10.10).

Power-law channels. Every ``(a, b)`` pair satisfies the SDK power law

    rr = a · C^b      (C in ppm, rr = R/R0)

with concentration inverted as ``C = (rr / a) ^ (1 / b)``. Values are the
**converted** datasheet-derived constants (source ``MQSensorsLib``, referenced
by each sensor's ``sources``), so they are directly consumable by the
reference-point calibration machinery. They are single-reference-substance
estimates; real quantification still needs per-rig, per-channel measured
reference points (§4.6).

Relative-response channels. Some MEMS entries (``SGP30``, ``SGP40``,
``TGS8100``, ``BME680``) carry no ``gases`` table and
``power_law_calibratable = False``: there is no authoritative, publicly
available power-law mapping to a target gas concentration for them, so the
SDK deliberately refuses to fabricate one. They are usable as relative
response/VOC-index channels, but ``power_law``/``clean_air_ratio``/etc. raise
``UnknownSensorError`` for them. Use ``is_power_law_sensor`` to distinguish.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

_SENSORS_PATH = Path(__file__).with_name("sensors.json")

_load = None


def _load_sensors() -> dict:
    global _load
    if _load is None:
        with _SENSORS_PATH.open("r", encoding="utf-8") as f:
            _load = json.load(f)
    return _load


class UnknownSensorError(KeyError):
    """The requested sensor model is not in the offline constants table."""


def sensor_models() -> List[str]:
    """Sorted list of sensor models present in the constants table."""
    return sorted(_load_sensors()["sensors"].keys())


def _gases_for(sensor: str) -> Dict[str, Dict[str, float]]:
    """Gas-to-constants map for ``sensor`` ([] if none tabulated).

    A ``KeyError`` here means either the sensor is unknown, or it is a known
    non-power-law channel (e.g. relative-response MEMS like ``TGS8100`` /
    ``SGP30`` / ``SGP40`` / ``BME680``) that has no ``gases`` table. Both cases
    surface as ``UnknownSensorError`` so callers never see a raw ``KeyError``.
    """
    try:
        gases = _load_sensors()["sensors"][sensor]["gases"]
    except KeyError:
        raise UnknownSensorError(sensor) from None
    return gases


def sensor_gases(sensor: str) -> List[str]:
    """Gases (substances) a sensor model has response constants for."""
    return list(_gases_for(sensor).keys())


def clean_air_ratio(sensor: str) -> float:
    """Clean-air resistance ratio ``Rs/R0`` reported for the sensor model."""
    try:
        return float(_load_sensors()["sensors"][sensor]["clean_air_ratio"])
    except KeyError:
        raise UnknownSensorError(sensor) from None


def power_law(sensor: str, gas: str) -> Dict[str, float]:
    """Return ``{"a": a, "b": b}`` for ``sensor`` responding to ``gas``.

    Raises ``UnknownSensorError`` for an unknown sensor (or a known
    non-power-law channel with no ``gases`` table) and ``KeyError`` for a gas
    the sensor has no tabulated response for.
    """
    gases = _gases_for(sensor)
    try:
        return dict(gases[gas])
    except KeyError:
        raise KeyError(
            f"'{sensor}' has no tabulated response for gas '{gas}'. "
            f"Known gases: {sorted(gases)}") from None


def all_power_laws(sensor: str) -> Dict[str, Dict[str, float]]:
    """All ``{gas: {"a": a, "b": b}}`` responses for a sensor model."""
    return {g: dict(v) for g, v in _gases_for(sensor).items()}


def is_power_law_sensor(sensor: str) -> bool:
    """Whether ``sensor`` has a tabulated power-law ``gases`` table.

    Power-law channels (e.g. the MQ family) return True and can feed
    ``power_law``/``clean_air_ratio``. Relative-response MEMS entries
    (``TGS8100``, ``SGP30``, ``SGP40``, ``BME680``) and unknown sensors
    return False -- they cannot be quantified from offline constants.
    """
    try:
        return "gases" in _load_sensors()["sensors"][sensor]
    except KeyError:
        return False


def sensor_sources(sensor: str) -> List[str]:
    """Verification/reference URLs for ``sensor`` (datasheets, source code).

    These are the provenance links backing the tabulated constants (or, for
    non-power-law MEMS entries, the official documentation confirming no
    authoritative power-law exists). Raises ``UnknownSensorError`` if the
    sensor is not in the table.
    """
    try:
        return list(_load_sensors()["sensors"][sensor].get("sources", []))
    except KeyError:
        raise UnknownSensorError(sensor) from None


__all__ = [
    "UnknownSensorError",
    "sensor_models",
    "sensor_gases",
    "clean_air_ratio",
    "power_law",
    "all_power_laws",
    "is_power_law_sensor",
    "sensor_sources",
]
