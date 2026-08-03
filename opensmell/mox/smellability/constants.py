"""Smellability constants and band tables.

Mirrors `osmograph-web/lib/smellability/constants.ts` 1:1. The volatility and
headspace band tables drive the chain's grading; the array-capacity table
(MAX_SUBSTANCES) reconciles with the canonical spec table.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import List, NamedTuple, Optional

from .types import SignalBand

AMBIENT_TEMP_C = 25
AMBIENT_TEMP_K = 298.15

DEFAULT_SENSOR_COUNT = 6
DEFAULT_DISTANCE_M = 0.1

MOX_FLOOR_PPM = 1

REFERENCE_CHEMICAL_ID = "ethanol"

MAX_SUBSTANCES = {
    3: 6,
    4: 12,
    5: 20,
    6: 40,
    12: 200,
    24: 10000,
}

SENSOR_COUNT_OPTIONS = [3, 4, 5, 6, 12, 24]


@dataclass(frozen=True)
class Band:
    min: float
    max: float
    label: str


VOLATILITY_BANDS: List[Band] = [
    Band(10000, inf, "very high"),
    Band(1000, 10000, "high"),
    Band(100, 1000, "moderate"),
    Band(1, 100, "low"),
    Band(0, 1, "negligible"),
]

HEADSPACE_PPM_BANDS: List[Band] = [
    Band(1000, inf, "strong"),
    Band(100, 1000, "moderate"),
    Band(10, 100, "weak"),
    Band(MOX_FLOOR_PPM, 10, "marginal"),
    Band(0, MOX_FLOOR_PPM, "none"),
]

SIGNAL_RATIO_BANDS: List[Band] = [
    Band(1, inf, "strong"),
    Band(0.1, 1, "moderate"),
    Band(0.01, 0.1, "weak"),
    Band(0.001, 0.01, "marginal"),
    Band(0, 0.001, "none"),
]


def volatility_label(p_vap_pa: Optional[float]) -> str:
    if p_vap_pa is None:
        return "unknown"
    for band in VOLATILITY_BANDS:
        if band.min <= p_vap_pa < band.max:
            return band.label
    return "unknown"


def headspace_ppm_band(ppm: float) -> SignalBand:
    for band in HEADSPACE_PPM_BANDS:
        if band.min <= ppm < band.max:
            return band.label
    return "none"


def signal_band_label(ratio: float) -> SignalBand:
    for band in SIGNAL_RATIO_BANDS:
        if band.min <= ratio < band.max:
            return band.label
    return "none"


class ClassTerm(NamedTuple):
    label: str
    functional_groups: List[str]


CLASS_TERMS = {
    "alcohol": ClassTerm("Alcohols", ["alcohol"]),
    "aldehyde": ClassTerm("Aldehydes", ["aldehyde"]),
    "ketone": ClassTerm("Ketones", ["ketone"]),
    "ester": ClassTerm("Esters", ["ester"]),
    "carboxylic acid": ClassTerm("Carboxylic acids", ["carboxylic acid"]),
    "alkane": ClassTerm("Alkanes", ["alkane"]),
    "alkene": ClassTerm("Alkenes", ["alkene"]),
    "terpene": ClassTerm("Terpenes", ["terpene"]),
    "thiol": ClassTerm("Thiols / mercaptans", ["thiol"]),
    "sulfide": ClassTerm("Organic sulfides", ["thioether", "sulfur"]),
    "amine": ClassTerm("Amines", ["amine"]),
    "phenol": ClassTerm("Phenols", ["phenol"]),
    "aromatic": ClassTerm("Aromatic hydrocarbons", ["aromatic"]),
    "ether": ClassTerm("Ethers", ["ether"]),
}
