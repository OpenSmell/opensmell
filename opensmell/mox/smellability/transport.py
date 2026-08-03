"""Transport physics: vapor pressure, diffusion, flux.

Mirrors `osmograph-web/lib/smellability/transport.ts` 1:1. Antoine constants are
the NIST-corrected values; the Clausius-Clapeyron + Trouton path is the honest
estimation branch used when only a boiling point is known.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

R = 8.314
N_A = 6.022e23
P_ATM = 101325

MMHG_TO_PA = 133.322


def vapor_pressure_antoine(a: float, b: float, c: float, temp_c: float) -> float:
    p_mmhg = 10 ** (a - b / (temp_c + c))
    return p_mmhg * MMHG_TO_PA


def vapor_pressure_clausius_clapeyron(temp_k: float, t_boil_k: float, delta_h_vap: float) -> float:
    return P_ATM * math.exp(-(delta_h_vap / R) * (1 / temp_k - 1 / t_boil_k))


def evaporation_flux(p_vap: float, mol_weight_kg: float, temp_k: float) -> float:
    return p_vap / math.sqrt(2 * math.pi * mol_weight_kg * R * temp_k)


def diffusion_coefficient_fuller(
    mol_weight: float,
    diffusion_volume: float,
    temp_k: float,
    pressure_atm: float = 1.0,
) -> float:
    m_air = 28.97
    v_air = 20.1
    d_cm2 = (
        (0.00143 * math.pow(temp_k, 1.75))
        / (pressure_atm * math.pow(math.pow(v_air, 1 / 3) + math.pow(diffusion_volume, 1 / 3), 2))
        * math.sqrt(1 / m_air + 1 / mol_weight)
    )
    return d_cm2 * 1e-4


def concentration_at_distance(evap_rate: float, d: float, distance_m: float) -> float:
    return evap_rate / (4 * math.pi * d * distance_m)


def incident_flux(concentration: float, mol_weight_kg: float, temp_k: float) -> float:
    return concentration * math.sqrt((R * temp_k) / (2 * math.pi * mol_weight_kg))


def diffusion_volume_from_mw(mol_weight: float) -> float:
    return 1.1 * mol_weight


def delta_h_vap_trouton(t_boil_k: float) -> float:
    return 88 * t_boil_k


@dataclass
class IncidentFluxInput:
    vapor_pressure_pa: float
    mol_weight_kg: float
    diffusion_volume_cm3: float


def incident_flux_proportional(input_: IncidentFluxInput) -> float:
    d = diffusion_coefficient_fuller(input_.mol_weight_kg * 1000, input_.diffusion_volume_cm3, 298.15)
    return input_.vapor_pressure_pa / (input_.mol_weight_kg * d)


def signal_ratio_vs_ref(compound: IncidentFluxInput, reference: IncidentFluxInput) -> float:
    return incident_flux_proportional(compound) / incident_flux_proportional(reference)
