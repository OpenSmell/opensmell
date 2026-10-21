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

Sensor and gas lookup is deliberately tolerant of the spellings people copy
from datasheets and READMEs: ``MQ-135``, ``MQ135``, ``mq_135`` and ``MQ 135``
all resolve to the same entry, and gas names are matched case-insensitively.
Common synonyms are only mapped when the table labels the same measurement
(for example ``C2H5OH`` -> ``Alcohol``); no gas is invented for a sensor that
does not tabulate it.
"""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

_SENSORS_PATH = Path(__file__).with_name("sensors.json")

_load = None
_sensor_index = None


def _load_sensors() -> dict:
    global _load
    if _load is None:
        with _SENSORS_PATH.open("r", encoding="utf-8") as f:
            _load = json.load(f)
    return _load


def _key(name: object) -> str:
    """Case/punctuation-insensitive lookup key for a sensor or gas name."""
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


def _sensor_lookup() -> Dict[str, str]:
    global _sensor_index
    if _sensor_index is None:
        _sensor_index = {
            _key(name): name for name in _load_sensors()["sensors"]
        }
    return _sensor_index


class UnknownSensorError(KeyError):
    """The requested sensor model is not in the offline constants table."""

    def __init__(self, sensor: object, reason: Optional[str] = None):
        self.sensor = str(sensor)
        self.reason = reason
        if reason:
            message = f"Sensor {self.sensor!r} {reason}."
        else:
            known = sensor_models()
            message = (
                f"Unknown sensor {self.sensor!r}. "
                f"Known sensors: {', '.join(known)}."
            )
            matches = difflib.get_close_matches(self.sensor, known, n=3)
            if matches:
                message += (
                    " Did you mean "
                    + ", ".join(repr(m) for m in matches)
                    + "?"
                )
        super().__init__(message)

    def __str__(self) -> str:
        return self.args[0]


class UnknownGasError(KeyError):
    """The sensor exists but has no tabulated response for this gas."""

    def __init__(self, sensor: str, gas: object, known: List[str]):
        self.sensor = sensor
        self.gas = str(gas)
        self.known_gases = list(known)
        message = (
            f"Sensor {sensor!r} has no tabulated response for gas {self.gas!r}. "
            f"Known gases: {', '.join(self.known_gases)}."
        )
        matches = difflib.get_close_matches(self.gas, self.known_gases, n=3)
        if matches:
            message += (
                " Did you mean "
                + ", ".join(repr(m) for m in matches)
                + "?"
            )
        super().__init__(message)

    def __str__(self) -> str:
        return self.args[0]


def sensor_models() -> List[str]:
    """Sorted list of sensor models present in the constants table."""
    return sorted(_load_sensors()["sensors"].keys())


def list_sensors() -> List[str]:
    """Public alias for :func:`sensor_models` (``opensmell.list_sensors()``)."""
    return sensor_models()


def normalize_sensor_name(sensor: object) -> str:
    """Resolve a user spelling to the canonical constants-table model name.

    ``MQ-135``, ``MQ135``, ``mq_135`` and ``MQ 135`` all resolve to ``MQ-135``.
    Raises :class:`UnknownSensorError` with the valid options when no model
    matches.
    """
    canonical = _sensor_lookup().get(_key(sensor))
    if canonical is None:
        raise UnknownSensorError(sensor)
    return canonical


def _gases_for(sensor: str) -> Dict[str, Dict[str, float]]:
    """Gas-to-constants map for ``sensor``.

    A known non-power-law channel (e.g. relative-response MEMS like
    ``TGS8100`` / ``SGP30`` / ``SGP40`` / ``BME680``) has no ``gases`` table.
    That is reported as :class:`UnknownSensorError` with an explicit reason so
    callers never see a raw ``KeyError`` or mistake it for a typo.
    """
    canonical = normalize_sensor_name(sensor)
    entry = _load_sensors()["sensors"][canonical]
    gases = entry.get("gases")
    if not gases:
        raise UnknownSensorError(
            canonical,
            reason=(
                "is a relative-response channel with no tabulated "
                "power-law gases"
            ),
        )
    return gases


# Synonyms that name the same measurement as a tabulated gas. These are
# intentionally narrow: a missing gas must stay missing rather than be
# silently mapped onto a chemically different curve.
_GAS_ALIASES = {
    "C2H5OH": "Alcohol",
    "ETHANOL": "Alcohol",
    "ETHYLALCOHOL": "Alcohol",
}


def normalize_gas_name(sensor: str, gas: object) -> str:
    """Resolve a gas spelling to the canonical name for ``sensor``.

    Matching is case- and punctuation-insensitive, with a small set of
    documented synonyms. Raises :class:`UnknownGasError` listing the sensor's
    tabulated gases when no match exists.
    """
    canonical_sensor = normalize_sensor_name(sensor)
    gases = _gases_for(canonical_sensor)
    raw = str(gas)

    if raw in gases:
        return raw

    key = _key(raw)
    for canonical in gases:
        if _key(canonical) == key:
            return canonical

    alias = _GAS_ALIASES.get(key)
    if alias is not None:
        for canonical in gases:
            if _key(canonical) == _key(alias):
                return canonical

    raise UnknownGasError(canonical_sensor, raw, sorted(gases))


def sensor_gases(sensor: str) -> List[str]:
    """Gases (substances) a sensor model has response constants for."""
    return list(_gases_for(sensor).keys())


def clean_air_ratio(sensor: str) -> float:
    """Clean-air resistance ratio ``Rs/R0`` reported for the sensor model."""
    canonical = normalize_sensor_name(sensor)
    entry = _load_sensors()["sensors"][canonical]
    try:
        return float(entry["clean_air_ratio"])
    except KeyError:
        raise UnknownSensorError(
            canonical, reason="has no tabulated clean-air ratio"
        ) from None


def power_law(sensor: str, gas: str) -> Dict[str, float]:
    """Return ``{"a": a, "b": b}`` for ``sensor`` responding to ``gas``.

    Raises :class:`UnknownSensorError` for an unknown sensor (or a known
    non-power-law channel with no ``gases`` table) and :class:`UnknownGasError`
    for a gas the sensor has no tabulated response for. Both subclasses of
    ``KeyError`` carry the valid options in their message.
    """
    canonical_sensor = normalize_sensor_name(sensor)
    canonical_gas = normalize_gas_name(canonical_sensor, gas)
    return dict(_gases_for(canonical_sensor)[canonical_gas])


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
        canonical = normalize_sensor_name(sensor)
    except UnknownSensorError:
        return False
    return bool(_load_sensors()["sensors"][canonical].get("gases"))


def sensor_sources(sensor: str) -> List[str]:
    """Verification/reference URLs for ``sensor`` (datasheets, source code).

    These are the provenance links backing the tabulated constants (or, for
    non-power-law MEMS entries, the official documentation confirming no
    authoritative power-law exists). Raises ``UnknownSensorError`` if the
    sensor is not in the table.
    """
    canonical = normalize_sensor_name(sensor)
    return list(_load_sensors()["sensors"][canonical].get("sources", []))


__all__ = [
    "UnknownSensorError",
    "UnknownGasError",
    "sensor_models",
    "list_sensors",
    "normalize_sensor_name",
    "normalize_gas_name",
    "sensor_gases",
    "clean_air_ratio",
    "power_law",
    "all_power_laws",
    "is_power_law_sensor",
    "sensor_sources",
]
