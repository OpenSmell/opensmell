"""Perceptual ontology and MOX capability boundaries.

Mirrors `osmograph-web/lib/smellability/ontology.ts` 1:1. PERCEPTS map chemistry
to the "hat" a user perceives (fruity esters, citrus terpenes, ...). MOX_BOUNDARIES
mirrors the spec's Table 3: 4 capabilities / 5 limitations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .types import Chemical, FeasibilityVerdict


@dataclass
class Percept:
    id: str
    label: str
    description: str
    groups: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


@dataclass
class MoxBoundary:
    id: str
    domain: str
    capability: bool
    statement: str
    implication: str


PERCEPTS: List[Percept] = [
    Percept(
        id="fruity-ester",
        label="Fruity / sweet esters",
        description="Volatile esters and small ketones — the chemistry behind ripe fruit.",
        groups=["ester"],
        keywords=["fruity", "banana", "pineapple", "sweet", "apricot", "apple", "pear"],
    ),
    Percept(
        id="citrus-terpenic",
        label="Citrus / terpenic",
        description="Terpenes (limonene, pinene, myrcene) — peel oils and conifers.",
        groups=["terpene"],
        keywords=["citrus", "terpene", "lemon", "pine", "orange"],
    ),
    Percept(
        id="green-leafy",
        label="Green / leafy",
        description="C6 aldehydes and alcohols (hexanal, hexenals) — freshly cut foliage.",
        groups=[],
        keywords=["green", "grassy", "leaf", "leafy", "tallow"],
    ),
    Percept(
        id="floral",
        label="Floral",
        description="Terpene alcohols and linalool-type aromatics.",
        groups=[],
        keywords=["floral", "lavender", "rose", "violet"],
    ),
    Percept(
        id="minty",
        label="Minty / cooling",
        description="Menthol-type cyclic alcohols and menthone ketones.",
        groups=[],
        keywords=["mint", "menthol", "cooling"],
    ),
    Percept(
        id="spicy-balsamic",
        label="Spicy / balsamic",
        description="Cinnamaldehyde and phenolic spices — low volatility, slow release.",
        groups=["phenol"],
        keywords=["cinnamon", "spicy", "clove"],
    ),
    Percept(
        id="roasted-caramel",
        label="Roasted / caramel",
        description="Maillard products (furfural, diacetyl) — baked and roasted notes.",
        groups=[],
        keywords=["roast", "caramel", "bakey", "butter", "toasted"],
    ),
    Percept(
        id="smoky-phenolic",
        label="Smoky / phenolic",
        description="Phenol and guaiacol — pyrolysis signatures.",
        groups=["phenol"],
        keywords=["smoke", "smoky", "phenolic", "campfire", "creosote"],
    ),
    Percept(
        id="sulfurous",
        label="Sulfurous / rotten",
        description="Thiols, sulfides, and H2S — the most potent MOX-reducing agents.",
        groups=["thiol", "thioether", "sulfur"],
        keywords=["sulfurous", "garlic", "sewer", "rotten", "skunk", "gas"],
    ),
    Percept(
        id="ammoniacal",
        label="Ammoniacal",
        description="Ammonia and amines — sharp, basic headspace.",
        groups=["amine"],
        keywords=["ammonia", "fishy", "urine"],
    ),
    Percept(
        id="solvent-industrial",
        label="Solvent / industrial",
        description="Aromatics (BTX) and alkanes — fuels, thinners, cleaning products.",
        groups=[],
        keywords=["solvent", "gasoline", "paint", "fuel", "aromatic"],
    ),
    Percept(
        id="alcoholic",
        label="Alcoholic",
        description="Small-chain alcohols — ethanol and relatives.",
        groups=["alcohol"],
        keywords=["alcohol", "alcoholic"],
    ),
    Percept(
        id="sour-acidic",
        label="Sour / acidic",
        description="Carboxylic acids — vinegar and rancid notes.",
        groups=["carboxylic acid"],
        keywords=["vinegar", "sour", "rancid", "acidic"],
    ),
    Percept(
        id="neutral-gas",
        label="Odorless combustibles",
        description="Methane, propane — low odor but strong MOX reducers when odorized.",
        groups=[],
        keywords=["methane", "propane", "odorless"],
    ),
]

LOW_VOLATILITY = {"spicy-balsamic", "smoky-phenolic"}


def percepts_for(chemical: Chemical) -> List[Percept]:
    groups = set(chemical.props.functional_groups or [])
    text = f"{chemical.name} {chemical.props.odor_descriptor or ''}".lower()

    return [
        p
        for p in PERCEPTS
        if any(g in groups for g in p.groups) or any(k in text for k in p.keywords)
    ]


def top_percepts(chemical: Chemical, max_percepts: int = 3) -> List[Percept]:
    return percepts_for(chemical)[:max_percepts]


def dominant_percept(chemical: Chemical) -> Optional[Percept]:
    ps = percepts_for(chemical)
    if not ps:
        return None
    fg = chemical.props.functional_groups or []
    group_hit = [p for p in ps if any(g in fg for g in p.groups)]
    return (group_hit[0] if group_hit else ps[0]) or None


def is_low_volatility_percept(percept: Optional[Percept]) -> bool:
    return percept is not None and percept.id in LOW_VOLATILITY


MOX_BOUNDARIES: List[MoxBoundary] = [
    MoxBoundary(
        id="functional-groups",
        domain="Identity",
        capability=True,
        statement="Rough chemical family (esters, aldehydes, terpenes, thiols…)",
        implication="You can read the *kind* of chemistry — the hat — but not the exact molecule.",
    ),
    MoxBoundary(
        id="molecular-size",
        domain="Identity",
        capability=True,
        statement="Small vs large volatile molecules",
        implication="Size ordering is visible in kinetics; exact mass is not.",
    ),
    MoxBoundary(
        id="vapor-pressure",
        domain="Identity",
        capability=True,
        statement="Volatility / how readily it reaches the sensor",
        implication="The engine's headspace estimate is the physical upper bound, not a reading.",
    ),
    MoxBoundary(
        id="redox",
        domain="Reactivity",
        capability=True,
        statement="Redox activity — will it reduce the sensor surface",
        implication="Reducing VOCs respond; the response is proportional to total reducing power.",
    ),
    MoxBoundary(
        id="structure",
        domain="Identity",
        capability=False,
        statement="Exact molecular structure (isomers, chirality)",
        implication="Limonene vs pinene, L- vs D-carvone: indistinguishable to MOX.",
    ),
    MoxBoundary(
        id="concentration",
        domain="Concentration",
        capability=False,
        statement="Absolute concentration (ppm)",
        implication="No calibration → relative response only. Treat any ppm as an estimate.",
    ),
    MoxBoundary(
        id="non-redox",
        domain="Reactivity",
        capability=False,
        statement="Non-redox-active gases (N2, O2, CO2, noble gases)",
        implication="CO2 is abundant in headspace yet invisible to a MOX array.",
    ),
    MoxBoundary(
        id="trace",
        domain="Sensitivity",
        capability=False,
        statement="Trace concentrations below ~1 ppm",
        implication="Below the practical MOX floor, regardless of how strong the smell is.",
    ),
    MoxBoundary(
        id="mixture",
        domain="Composition",
        capability=False,
        statement="Decomposing complex mixtures into components",
        implication="A 50/50 blend can look like a pure substance to the array.",
    ),
]


def describe_boundaries() -> dict:
    can = [b for b in MOX_BOUNDARIES if b.capability]
    cannot = [b for b in MOX_BOUNDARIES if not b.capability]
    return {"can": can, "cannot": cannot}


def relevant_boundaries(verdict: FeasibilityVerdict) -> List[str]:
    hits: List[str] = []
    reactivity = next((s for s in verdict.steps if s.id == "reactivity"), None)
    signal = next((s for s in verdict.steps if s.id == "signal"), None)
    if reactivity is not None and reactivity.verdict == "red":
        hits.append("non-redox")
    if verdict.signal_strength == "none" and signal is not None and any("ppm" in v.value for v in signal.values):
        hits.append("trace")
    if verdict.kind == "composite":
        hits.append("mixture")
    if verdict.kind == "class":
        hits.append("structure")
    return hits


def perceptual_summary(verdict: FeasibilityVerdict, percepts: List[Percept]) -> str:
    hat = "; ".join(p.label.lower() for p in percepts) if percepts else "unclassified chemistry"
    base = f"{verdict.entity_name} reads as {hat}."
    if verdict.verdict == "red":
        if any(s.id == "reactivity" and s.verdict == "red" for s in verdict.steps):
            return f"{base} The array cannot confirm it: the chemistry is not redox-active at MOX operating temperatures (beyond-MOX boundary)."
        return f"{base} The array is unlikely to register a usable signal under normal conditions."
    if verdict.confidence == "low":
        return f"{base} The array should respond, but key properties are unknown — treat the strength estimate as a guess until verified."
    if is_low_volatility_percept(percepts[0] if percepts else None):
        return f"{base} Expect a weak, slow signal — this family is low-volatility, so the headspace builds slowly and stays small. The exact identity and ppm are beyond MOX."
    return f"{base} Expect a clear reducing response — the family is volatile and redox-active. The exact molecule and ppm are beyond MOX; what you get is the kind."
