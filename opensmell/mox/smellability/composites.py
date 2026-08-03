"""Composite (mixture) dictionary.

Mirrors `osmograph-web/lib/smellability/composites.ts` 1:1. Weight fractions are
literature estimates (GC-MS / FlavorDB) — the `estimated` source in the web
build. All fractions sum toward 1; the chain renormalizes.
"""

from __future__ import annotations

from typing import List

from .types import Composite, CompositeConstituent, Property

COMPOSITES: List[Composite] = []


def _wf(chemical_id: str, fraction: float, source: str = "estimated") -> CompositeConstituent:
    return CompositeConstituent(
        chemical_id=chemical_id,
        weight_fraction=Property(value=fraction, source=source),
    )


def _add(
    id_: str,
    name: str,
    kind: str,
    synonyms: List[str],
    constituents: List[CompositeConstituent],
    source_refs: List[str],
    notes: str | None = None,
) -> None:
    COMPOSITES.append(
        Composite(
            id=id_,
            name=name,
            kind=kind,  # type: ignore[arg-type]
            synonyms=synonyms,
            constituents=constituents,
            source_refs=source_refs,
            notes=notes,
        )
    )


_add(
    "banana",
    "Banana",
    "food",
    ["ripe banana", "banana fruit", "banana aroma"],
    [
        _wf("isoamyl-acetate", 0.5),
        _wf("isoamyl-butyrate", 0.15),
        _wf("butyl-acetate", 0.1),
        _wf("isoamyl-isovalerate", 0.1),
        _wf("hexanal", 0.05),
        _wf("e2-hexenal", 0.05),
        _wf("hexanol", 0.05),
    ],
    ["Zhu et al. 2018 (Molecules)", "Pino & Febles 2013", "Sutikdja et al. 2012"],
    "Character impact is isoamyl acetate; esters dominate ripe-fruit headspace. Profile varies with ripeness (green stage is aldehyde-heavy).",
)
_add(
    "cinnamon",
    "Cinnamon",
    "spice",
    ["cinnamon bark", "cinnamon powder", "ceylon cinnamon"],
    [
        _wf("cinnamaldehyde", 0.65),
        _wf("eugenol", 0.2),
        _wf("linalool", 0.05),
        _wf("limonene", 0.05),
    ],
    ["PubChem", "FlavorDB ingredient profiles"],
    "Simplified profile dominated by cinnamaldehyde (bark).",
)
_add(
    "coffee",
    "Coffee",
    "beverage",
    ["roasted coffee", "coffee beans", "espresso"],
    [
        _wf("diacetyl", 0.25),
        _wf("furfural", 0.2),
        _wf("acetaldehyde", 0.15),
        _wf("acetic-acid", 0.15),
        _wf("guaiacol", 0.1),
    ],
    ["FlavorDB2", "Chambers & Koppel 2013 (Molecules)"],
    "Highly complex headspace (200+ VOCs); this is a simplified representative subset of potent roast volatiles.",
)
_add(
    "garlic",
    "Garlic",
    "food",
    ["raw garlic", "garlic clove", "garlic powder"],
    [
        _wf("diallyl-disulfide", 0.6),
        _wf("methanethiol", 0.15),
        _wf("dimethyl-sulfide", 0.1),
    ],
    ["FlavorDB ingredient profiles", "Lanzotti 2006"],
    "Organosulfur compounds dominate and are potent MOX-reducing agents.",
)
_add(
    "peppermint",
    "Peppermint",
    "spice",
    ["mint", "spearmint", "peppermint oil"],
    [
        _wf("menthol", 0.55),
        _wf("menthone", 0.3),
        _wf("limonene", 0.15),
    ],
    ["FlavorDB ingredient profiles"],
    "Menthol + menthone dominate; limonene adds citrus lift.",
)
_add(
    "lemon",
    "Lemon",
    "food",
    ["lemon peel", "lemon zest", "citrus lemon"],
    [
        _wf("limonene", 0.6),
        _wf("linalool", 0.15),
        _wf("alpha-pinene", 0.1),
        _wf("myrcene", 0.05),
    ],
    ["PubChem", "FlavorDB ingredient profiles"],
    "Peel oil dominated by limonene.",
)
_add(
    "orange",
    "Orange",
    "food",
    ["orange peel", "sweet orange"],
    [
        _wf("limonene", 0.7),
        _wf("linalool", 0.1),
        _wf("alpha-pinene", 0.05),
    ],
    ["FlavorDB ingredient profiles"],
    "Peel oil dominated by limonene.",
)
_add(
    "strawberry",
    "Strawberry",
    "food",
    ["ripe strawberry", "strawberry aroma"],
    [
        _wf("methyl-butyrate", 0.35),
        _wf("ethyl-butyrate", 0.3),
        _wf("isoamyl-acetate", 0.15),
        _wf("hexanal", 0.1),
    ],
    ["FlavorDB ingredient profiles"],
    "Esters dominate fresh strawberry aroma.",
)
_add(
    "apple",
    "Apple",
    "food",
    ["fresh apple", "apple aroma", "granny smith"],
    [
        _wf("hexanal", 0.3),
        _wf("e2-hexenal", 0.25),
        _wf("butyl-acetate", 0.2),
        _wf("ethyl-butyrate", 0.1),
    ],
    ["FlavorDB ingredient profiles"],
    "Green aldehydes + fruity esters; ripening shifts toward esters.",
)
_add(
    "tomato",
    "Tomato",
    "food",
    ["fresh tomato", "tomato leaf", "ripe tomato"],
    [
        _wf("hexanal", 0.4),
        _wf("e2-hexenal", 0.3),
        _wf("hexanol", 0.15),
        _wf("limonene", 0.05),
    ],
    ["FlavorDB ingredient profiles"],
    "Green-leaf volatiles (C6 aldehydes) dominate.",
)
_add(
    "vinegar",
    "Vinegar",
    "food",
    ["white vinegar", "apple cider vinegar", "acetic acid vinegar"],
    [
        _wf("acetic-acid", 0.85),
        _wf("acetaldehyde", 0.05),
    ],
    [],
    "Essentially an aqueous acetic acid solution.",
)
_add(
    "wine",
    "Wine",
    "beverage",
    ["red wine", "white wine", "grape wine"],
    [
        _wf("ethanol", 0.7),
        _wf("ethyl-acetate", 0.15),
        _wf("ethyl-butyrate", 0.05),
        _wf("acetic-acid", 0.05),
    ],
    ["FlavorDB ingredient profiles"],
    "Ethanol-dominated with fruity fermentation esters.",
)
_add(
    "beer",
    "Beer",
    "beverage",
    ["lager", "ale", "draft beer"],
    [
        _wf("ethanol", 0.55),
        _wf("myrcene", 0.1),
        _wf("linalool", 0.1),
        _wf("diacetyl", 0.05),
        _wf("acetic-acid", 0.05),
    ],
    ["FlavorDB ingredient profiles"],
    "Ethanol plus hop and fermentation aromatics; diacetyl gives buttery off-note.",
)
_add(
    "bread",
    "Bread",
    "food",
    ["fresh bread", "baked bread", "bread crust"],
    [
        _wf("diacetyl", 0.3),
        _wf("furfural", 0.2),
        _wf("acetaldehyde", 0.15),
        _wf("ethanol", 0.1),
    ],
    ["FlavorDB ingredient profiles"],
    "Maillard/bakey aromatics; simplified subset.",
)
_add(
    "spoiled-milk",
    "Spoiled milk",
    "food",
    ["sour milk", "off milk", "spoiled dairy"],
    [
        _wf("butyric-acid", 0.45),
        _wf("dimethyl-sulfide", 0.2),
        _wf("hydrogen-sulfide", 0.15),
        _wf("acetaldehyde", 0.1),
    ],
    ["Food spoilage literature (Dairy Science)"],
    "Lipolysis + protein degradation produce butyric acid and sulfides.",
)
_add(
    "gasoline",
    "Gasoline",
    "material",
    ["petrol", "fuel", "unleaded gasoline"],
    [
        _wf("toluene", 0.35),
        _wf("xylene", 0.25),
        _wf("hexane", 0.15),
        _wf("benzene", 0.1),
        _wf("isopentane", 0.1),
    ],
    ["ASTM fuel composition data"],
    "BTX + alkanes dominate light-fuel headspace; highly variable by blend.",
)
_add(
    "wood-smoke",
    "Wood smoke",
    "activity",
    ["smoke", "fire smoke", "campfire", "burning wood"],
    [
        _wf("guaiacol", 0.35),
        _wf("phenol", 0.25),
        _wf("eugenol", 0.1),
        _wf("acetaldehyde", 0.1),
        _wf("furfural", 0.05),
    ],
    ["Smoke chemistry literature"],
    "Phenolic signature of pyrolysis; simplified subset.",
)
_add(
    "paint-thinner",
    "Paint thinner",
    "product",
    ["mineral spirits", "white spirit", "solvent"],
    [
        _wf("toluene", 0.4),
        _wf("xylene", 0.3),
        _wf("ethyl-acetate", 0.15),
        _wf("acetone", 0.1),
    ],
    ["MSDS / solvent formulation data"],
    "Aromatic + ester solvents dominate.",
)
_add(
    "nail-polish-remover",
    "Nail polish remover",
    "product",
    ["acetone remover", "polish remover"],
    [_wf("acetone", 0.9)],
    [],
    "Essentially acetone.",
)
_add(
    "hand-sanitizer",
    "Hand sanitizer",
    "product",
    ["sanitizer gel", "hand rub", "alcohol gel"],
    [
        _wf("ethanol", 0.7),
        _wf("isopropanol", 0.2),
    ],
    ["CDC formulation guidance"],
    "Alcohol gel; residue is ethanol/IPA.",
)
_add(
    "natural-gas",
    "Natural gas",
    "activity",
    ["gas leak", "methane leak", "utility gas"],
    [
        _wf("methane", 0.9),
        _wf("methanethiol", 0.05),
        _wf("butane", 0.05),
    ],
    ["Gas utility odorization standards"],
    "Methane plus added sulfur odorant (mercaptan).",
)
_add(
    "propane-leak",
    "Propane / LPG",
    "activity",
    ["propane tank", "grill gas", "LPG leak"],
    [
        _wf("propane", 0.95),
        _wf("methanethiol", 0.05),
    ],
    ["LPG odorization standards"],
    "Propane plus added sulfur odorant.",
)
_add(
    "sewer",
    "Sewer odor",
    "activity",
    ["drain smell", "septic", "wastewater"],
    [
        _wf("hydrogen-sulfide", 0.45),
        _wf("ammonia", 0.25),
        _wf("methanethiol", 0.15),
        _wf("dimethyl-sulfide", 0.1),
    ],
    ["Wastewater odor literature"],
    "Hydrogen sulfide is the dominant signature.",
)
_add(
    "rotten-egg",
    "Rotten egg",
    "food",
    ["sulfurous egg", "bad egg"],
    [_wf("hydrogen-sulfide", 0.95)],
    [],
    "H2S from protein sulfur decomposition.",
)
_add(
    "car-exhaust",
    "Car exhaust",
    "activity",
    ["vehicle exhaust", "tailpipe", "engine emissions"],
    [
        _wf("carbon-monoxide", 0.8),
        _wf("acetaldehyde", 0.05),
        _wf("benzene", 0.05),
    ],
    ["Combustion emission literature"],
    "CO-dominated for gasoline cold-start; simplified subset.",
)
_add(
    "ripe-fruit-gas",
    "Ripe fruit (ethylene)",
    "activity",
    ["ethylene ripening", "fruit ripening", "climacteric fruit gas"],
    [_wf("ethylene", 0.9)],
    ["Plant physiology literature"],
    "Ethylene is the climacteric ripening hormone emitted by ripening fruit.",
)

COMPOSITE_BY_ID = {c.id: c for c in COMPOSITES}
