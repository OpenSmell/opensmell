"""Provisional chemicals built from live PubChem enrichment.

Mirrors `osmograph-web/lib/smellability/provisional.ts` 1:1. Everything derived
here is flagged `estimated`: molecular weight and boiling point come from
PubChem, vapor pressure is back-computed from the boiling point via
Clausius-Clapeyron + Trouton (or left unknown when no boiling point was fetched),
and functional groups are inferred structurally from SMILES.
"""

from __future__ import annotations

import math
import re
from typing import Optional

from .enrichment import EnrichedBoilingPoint, EnrichedChemical
from .groups import infer_functional_groups
from .types import Chemical, ChemicalProperties, Property

_INORGANIC_FULL = re.compile(r"^(n2|o2|co2|ar|he|ne)$")
_INORGANIC_NAMES = re.compile(
    r"^(n|o2?|co2|argon|helium|neon|nitrogen|oxygen|carbon dioxide)$"
)
_REDUCING = re.compile(
    r"^(h2|co|h2s|nh3|hydrogen|hydrogen sulfide|carbon monoxide|ammonia)$"
)


def build_provisional_chemical(
    enriched: EnrichedChemical,
    bp: Optional[EnrichedBoilingPoint],
) -> Chemical:
    id_ = f"prov-{slug(enriched.name)}"

    functional_groups = infer_functional_groups(enriched.smiles)
    name = enriched.name.strip().lower()
    # Reducing gases (H2, CO, H2S, NH3) and organic molecules are redox-active at
    # MOX operating temperature; true inerts (N2, O2, CO2, noble gases) are not.
    inorganic = _INORGANIC_FULL.match(name) is not None or _INORGANIC_NAMES.match(name) is not None
    redox_active = len(functional_groups) > 0 or _REDUCING.match(name) is not None

    return Chemical(
        id=id_,
        name=enriched.name,
        synonyms=[enriched.name],
        smiles=enriched.smiles,
        props=ChemicalProperties(
            molecular_weight=Property(
                value=enriched.molecular_weight,
                source="measured" if enriched.molecular_weight is not None else "unknown",
                note="PubChem" if enriched.molecular_weight is not None else None,
            ),
            boiling_point=(
                Property(value=bp.value_c, source="measured", note=bp.note)
                if bp
                else Property(value=None, source="unknown")
            ),
            vapor_pressure_25=(
                Property(
                    value=estimate_vapor_pressure_from_boiling_point(bp.value_c),
                    source="estimated",
                    note="Clausius–Clapeyron + Trouton from PubChem boiling point",
                )
                if bp
                else Property(value=None, source="unknown")
            ),
            functional_groups=functional_groups,
            redox_active=redox_active,
            non_redox=True if inorganic else None,
            odor_descriptor=None,
        ),
        source_refs=["PubChem (live lookup)"] if enriched.source == "pubchem" else [],
    )


def slug(name: str) -> str:
    return re.sub(r"^-+|-+$", "", re.sub(r"[^a-z0-9]+", "-", name.lower()))[:48]


def estimate_vapor_pressure_from_boiling_point(boil_c: float) -> Optional[float]:
    # Clausius-Clapeyron from the normal boiling point with Trouton's-rule ΔH_vap.
    # Same path as the estimated branch in chain.effective_vapor_pressure.
    r = 8.314
    t_boil_k = boil_c + 273.15
    delta_h_vap = 88 * t_boil_k
    p_pa = 101325 * math.exp(-(delta_h_vap / r) * (1 / 298.15 - 1 / t_boil_k))
    return p_pa if p_pa > 0 else None
