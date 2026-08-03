"""Curated compound dictionary.

Mirrors `osmograph-web/lib/smellability/compounds.ts` 1:1 — the same ~46 seed
compounds with PubChem/NIST-sourced properties. `m()` / `e()` / `u()` are the
measured / estimated / unknown property shorthands.
"""

from __future__ import annotations

from typing import List

from .types import AntoineCoeffs, Chemical, ChemicalProperties, Property

COMPOUNDS: List[Chemical] = []


def m(value: float) -> Property[float]:
    return Property(value=value, source="measured")


def e(value: float) -> Property[float]:
    return Property(value=value, source="estimated")


def u() -> Property[float]:
    return Property(value=None, source="unknown")


def _antoine(a: float, b: float, c: float) -> AntoineCoeffs:
    return AntoineCoeffs(a=a, b=b, c=c)


def _add(
    id_: str,
    name: str,
    synonyms: List[str],
    props: ChemicalProperties,
    source_refs: List[str],
    cas: str | None = None,
    smiles: str | None = None,
) -> None:
    COMPOUNDS.append(
        Chemical(
            id=id_,
            name=name,
            synonyms=synonyms,
            cas=cas,
            smiles=smiles,
            props=props,
            source_refs=source_refs,
        )
    )


_add(
    "ethanol",
    "Ethanol",
    ["ethyl alcohol", "alcohol", "grain alcohol", "drinking alcohol"],
    ChemicalProperties(
        molecular_weight=m(46.07),
        boiling_point=m(78.37),
        vapor_pressure_25=m(7870),
        antoine=_antoine(8.20417, 1642.89, 230.3),
        functional_groups=["alcohol"],
        redox_active=True,
        odor_descriptor="alcoholic, solvent",
    ),
    ["PubChem CID 702", "NIST Webbook"],
    cas="64-17-5",
    smiles="CCO",
)
_add(
    "methanol",
    "Methanol",
    ["methyl alcohol", "wood alcohol"],
    ChemicalProperties(
        molecular_weight=m(32.04),
        boiling_point=m(64.7),
        vapor_pressure_25=m(16900),
        antoine=_antoine(8.08097, 1582.271, 239.726),
        functional_groups=["alcohol"],
        redox_active=True,
        odor_descriptor="alcoholic, pungent",
    ),
    ["PubChem CID 887", "NIST Webbook"],
    cas="67-56-1",
    smiles="CO",
)
_add(
    "isopropanol",
    "Isopropanol",
    ["isopropyl alcohol", "rubbing alcohol", "propan-2-ol", "IPA"],
    ChemicalProperties(
        molecular_weight=m(60.1),
        boiling_point=m(82.6),
        vapor_pressure_25=m(6020),
        antoine=_antoine(8.87829, 2010.33, 252.636),
        functional_groups=["alcohol"],
        redox_active=True,
        odor_descriptor="alcoholic, sweet",
    ),
    ["PubChem CID 3776", "NIST Webbook"],
    cas="67-63-0",
    smiles="CC(C)O",
)
_add(
    "acetone",
    "Acetone",
    ["propanone", "dimethyl ketone", "nail polish remover"],
    ChemicalProperties(
        molecular_weight=m(58.08),
        boiling_point=m(56.05),
        vapor_pressure_25=m(30600),
        antoine=_antoine(7.11714, 1210.595, 229.664),
        functional_groups=["ketone"],
        redox_active=True,
        odor_descriptor="sweet, fruity, solvent",
    ),
    ["PubChem CID 180", "NIST Webbook"],
    cas="67-64-1",
    smiles="CC(=O)C",
)
_add(
    "acetaldehyde",
    "Acetaldehyde",
    ["ethanal", "acetaldehyde (in foods)"],
    ChemicalProperties(
        molecular_weight=m(44.05),
        boiling_point=m(20.2),
        vapor_pressure_25=m(120000),
        functional_groups=["aldehyde"],
        redox_active=True,
        odor_descriptor="pungent, fruity",
    ),
    ["PubChem CID 177", "NIST Webbook"],
    cas="75-07-0",
    smiles="CC=O",
)
_add(
    "acetic-acid",
    "Acetic acid",
    ["ethanoic acid", "vinegar", "glacial acetic acid"],
    ChemicalProperties(
        molecular_weight=m(60.05),
        boiling_point=m(118),
        vapor_pressure_25=m(2090),
        functional_groups=["carboxylic acid"],
        redox_active=True,
        odor_descriptor="vinegar, sour",
    ),
    ["PubChem CID 176", "NIST Webbook"],
    cas="64-19-7",
    smiles="CC(=O)O",
)
_add(
    "butyric-acid",
    "Butyric acid",
    ["butanoic acid", "n-butyric acid"],
    ChemicalProperties(
        molecular_weight=m(88.11),
        boiling_point=m(163.5),
        vapor_pressure_25=e(110),
        functional_groups=["carboxylic acid"],
        redox_active=True,
        odor_descriptor="rancid butter, vomit",
    ),
    ["PubChem CID 264", "NIST Webbook"],
    cas="107-92-6",
    smiles="CCCC(=O)O",
)
_add(
    "benzene",
    "Benzene",
    ["benzin"],
    ChemicalProperties(
        molecular_weight=m(78.11),
        boiling_point=m(80.1),
        vapor_pressure_25=m(12700),
        antoine=_antoine(6.90565, 1211.033, 220.79),
        functional_groups=["aromatic"],
        redox_active=True,
        odor_descriptor="aromatic, solvent",
    ),
    ["PubChem CID 241", "NIST Webbook"],
    cas="71-43-2",
    smiles="c1ccccc1",
)
_add(
    "toluene",
    "Toluene",
    ["methylbenzene", "toluol"],
    ChemicalProperties(
        molecular_weight=m(92.14),
        boiling_point=m(110.6),
        vapor_pressure_25=m(3790),
        antoine=_antoine(6.95464, 1344.8, 219.48),
        functional_groups=["aromatic"],
        redox_active=True,
        odor_descriptor="aromatic, paint-like",
    ),
    ["PubChem CID 1140", "NIST Webbook"],
    cas="108-88-3",
    smiles="Cc1ccccc1",
)
_add(
    "xylene",
    "Xylene",
    ["dimethylbenzene", "m-xylene", "xylol"],
    ChemicalProperties(
        molecular_weight=m(106.17),
        boiling_point=m(139),
        vapor_pressure_25=e(1100),
        functional_groups=["aromatic"],
        redox_active=True,
        odor_descriptor="sweet, solvent",
    ),
    ["PubChem CID 7809", "NIST Webbook"],
    cas="108-38-3",
    smiles="Cc1cccc(C)c1",
)
_add(
    "hexane",
    "n-Hexane",
    ["hexane", "normal hexane"],
    ChemicalProperties(
        molecular_weight=m(86.18),
        boiling_point=m(68.7),
        vapor_pressure_25=m(20000),
        functional_groups=["alkane"],
        redox_active=True,
        odor_descriptor="gasoline-like, faint",
    ),
    ["PubChem CID 8100", "NIST Webbook"],
    cas="110-54-3",
    smiles="CCCCCC",
)
_add(
    "isopentane",
    "Isopentane",
    ["2-methylbutane"],
    ChemicalProperties(
        molecular_weight=m(72.15),
        boiling_point=m(27.8),
        vapor_pressure_25=e(95000),
        functional_groups=["alkane"],
        redox_active=True,
        odor_descriptor="gasoline-like",
    ),
    ["PubChem CID 6556", "NIST Webbook"],
    cas="78-78-4",
    smiles="CCC(C)C",
)
_add(
    "isoamyl-acetate",
    "Isoamyl acetate",
    ["isopentyl acetate", "3-methylbutyl acetate", "banana oil"],
    ChemicalProperties(
        molecular_weight=m(130.18),
        boiling_point=m(142),
        vapor_pressure_25=e(700),
        functional_groups=["ester"],
        redox_active=True,
        odor_descriptor="banana, pear",
    ),
    ["PubChem CID 31276", "NIST Webbook"],
    cas="123-92-2",
    smiles="CC(C)CCOC(C)=O",
)
_add(
    "ethyl-acetate",
    "Ethyl acetate",
    ["acetic ester", "ethyl ethanoate"],
    ChemicalProperties(
        molecular_weight=m(88.11),
        boiling_point=m(77.1),
        vapor_pressure_25=m(12600),
        functional_groups=["ester"],
        redox_active=True,
        odor_descriptor="sweet, fruity, solvent",
    ),
    ["PubChem CID 8857", "NIST Webbook"],
    cas="141-78-6",
    smiles="CCOC(C)=O",
)
_add(
    "butyl-acetate",
    "Butyl acetate",
    ["n-butyl acetate", "butyl ethanoate"],
    ChemicalProperties(
        molecular_weight=m(116.16),
        boiling_point=m(126),
        vapor_pressure_25=e(1330),
        functional_groups=["ester"],
        redox_active=True,
        odor_descriptor="sweet, fruity, banana",
    ),
    ["PubChem CID 31272", "NIST Webbook"],
    cas="123-86-4",
    smiles="CCCCOC(C)=O",
)
_add(
    "isoamyl-butyrate",
    "Isoamyl butyrate",
    ["isopentyl butyrate", "3-methylbutyl butanoate"],
    ChemicalProperties(
        molecular_weight=m(158.24),
        boiling_point=e(179),
        vapor_pressure_25=e(60),
        functional_groups=["ester"],
        redox_active=True,
        odor_descriptor="fruity, apricot",
    ),
    ["PubChem CID 7794"],
    cas="106-27-4",
    smiles="CCCC(=O)OCCC(C)C",
)
_add(
    "isoamyl-isovalerate",
    "Isoamyl isovalerate",
    ["isopentyl 3-methylbutanoate", "isopentyl isovalerate"],
    ChemicalProperties(
        molecular_weight=m(172.26),
        boiling_point=e(193),
        vapor_pressure_25=e(40),
        functional_groups=["ester"],
        redox_active=True,
        odor_descriptor="apple, fruity",
    ),
    ["PubChem CID 69021"],
    cas="659-70-1",
    smiles="CC(C)CC(=O)OCCC(C)C",
)
_add(
    "methyl-butyrate",
    "Methyl butyrate",
    ["methyl butanoate"],
    ChemicalProperties(
        molecular_weight=m(102.13),
        boiling_point=m(102.8),
        vapor_pressure_25=e(3000),
        functional_groups=["ester"],
        redox_active=True,
        odor_descriptor="apple, pineapple",
    ),
    ["PubChem CID 12180"],
    cas="623-42-7",
    smiles="CCCC(=O)OC",
)
_add(
    "ethyl-butyrate",
    "Ethyl butyrate",
    ["ethyl butanoate"],
    ChemicalProperties(
        molecular_weight=m(116.16),
        boiling_point=m(121.5),
        vapor_pressure_25=e(2000),
        functional_groups=["ester"],
        redox_active=True,
        odor_descriptor="fruity, pineapple",
    ),
    ["PubChem CID 7762"],
    cas="105-54-4",
    smiles="CCCC(=O)OCC",
)
_add(
    "hexanal",
    "Hexanal",
    ["n-hexanal", "caproaldehyde", "hexyl aldehyde"],
    ChemicalProperties(
        molecular_weight=m(100.16),
        boiling_point=m(131),
        vapor_pressure_25=e(1330),
        functional_groups=["aldehyde"],
        redox_active=True,
        odor_descriptor="grassy, green, tallow",
    ),
    ["PubChem CID 6184"],
    cas="66-25-1",
    smiles="CCCCCC=O",
)
_add(
    "e2-hexenal",
    "(E)-2-Hexenal",
    ["trans-2-hexenal", "leaf aldehyde", "hex-2-enal"],
    ChemicalProperties(
        molecular_weight=m(98.14),
        boiling_point=m(147),
        vapor_pressure_25=e(600),
        functional_groups=["aldehyde", "alkene"],
        redox_active=True,
        odor_descriptor="green leaf, apple",
    ),
    ["PubChem CID 5281168"],
    cas="6728-26-3",
    smiles="CCCC=CC=O",
)
_add(
    "hexanol",
    "1-Hexanol",
    ["n-hexanol", "hexan-1-ol"],
    ChemicalProperties(
        molecular_weight=m(102.17),
        boiling_point=m(157),
        vapor_pressure_25=e(133),
        functional_groups=["alcohol"],
        redox_active=True,
        odor_descriptor="green, grassy, fruity",
    ),
    ["PubChem CID 8103"],
    cas="111-27-3",
    smiles="CCCCCCO",
)
_add(
    "eugenol",
    "Eugenol",
    ["clove oil", "4-allyl-2-methoxyphenol"],
    ChemicalProperties(
        molecular_weight=m(164.2),
        boiling_point=m(254),
        vapor_pressure_25=e(2.7),
        functional_groups=["phenol", "ether", "alkene"],
        redox_active=True,
        odor_descriptor="clove, spicy",
    ),
    ["PubChem CID 3314"],
    cas="97-53-0",
    smiles="COc1cc(CC=C)ccc1O",
)
_add(
    "cinnamaldehyde",
    "Cinnamaldehyde",
    ["cinnamic aldehyde", "3-phenylpropenal", "cinnamon aldehyde"],
    ChemicalProperties(
        molecular_weight=m(132.16),
        boiling_point=m(248),
        vapor_pressure_25=e(1.3),
        functional_groups=["aldehyde", "alkene", "aromatic"],
        redox_active=True,
        odor_descriptor="cinnamon, spicy",
    ),
    ["PubChem CID 637511"],
    cas="104-55-2",
    smiles="O=C/C=C/c1ccccc1",
)
_add(
    "linalool",
    "Linalool",
    ["linalol", "coriandrol"],
    ChemicalProperties(
        molecular_weight=m(154.25),
        boiling_point=m(198),
        vapor_pressure_25=e(26),
        functional_groups=["alcohol", "terpene", "alkene"],
        redox_active=True,
        odor_descriptor="floral, lavender, citrus",
    ),
    ["PubChem CID 6549"],
    cas="78-70-6",
    smiles="CC(C)=CCCC(C)(O)C=C",
)
_add(
    "limonene",
    "Limonene",
    ["d-limonene", "citrus terpene"],
    ChemicalProperties(
        molecular_weight=m(136.23),
        boiling_point=m(176),
        vapor_pressure_25=e(270),
        functional_groups=["terpene", "alkene"],
        redox_active=True,
        odor_descriptor="citrus, orange",
    ),
    ["PubChem CID 440917"],
    cas="5989-27-5",
    smiles="CC1=CCC(CC1)C(C)=C",
)
_add(
    "alpha-pinene",
    "alpha-Pinene",
    ["alpha-pinene", "pinene"],
    ChemicalProperties(
        molecular_weight=m(136.23),
        boiling_point=m(155),
        vapor_pressure_25=e(627),
        functional_groups=["terpene", "alkene"],
        redox_active=True,
        odor_descriptor="pine, resinous",
    ),
    ["PubChem CID 6654"],
    cas="80-56-8",
    smiles="CC1=CCC2CC1C2(C)C",
)
_add(
    "myrcene",
    "beta-Myrcene",
    ["myrcene", "beta-myrcene"],
    ChemicalProperties(
        molecular_weight=m(136.23),
        boiling_point=e(167),
        vapor_pressure_25=e(300),
        functional_groups=["terpene", "alkene"],
        redox_active=True,
        odor_descriptor="resinous, balsamic",
    ),
    ["PubChem CID 31253"],
    cas="123-35-3",
    smiles="CC(=C)CCC=C(C)C",
)
_add(
    "menthol",
    "Menthol",
    ["l-menthol", "peppermint camphor"],
    ChemicalProperties(
        molecular_weight=m(156.27),
        boiling_point=m(212),
        vapor_pressure_25=e(7),
        functional_groups=["alcohol", "terpene"],
        redox_active=True,
        odor_descriptor="mint, cooling",
    ),
    ["PubChem CID 165675"],
    cas="2216-51-5",
    smiles="CC1CCC(C(C1)O)C(C)C",
)
_add(
    "menthone",
    "Menthone",
    ["l-menthone", "peppermint ketone"],
    ChemicalProperties(
        molecular_weight=m(154.25),
        boiling_point=e(207),
        vapor_pressure_25=e(26),
        functional_groups=["ketone", "terpene"],
        redox_active=True,
        odor_descriptor="mint, peppermint",
    ),
    ["PubChem CID 165834"],
    cas="10458-14-7",
    smiles="CC1CCC(C(=O)C1)C(C)C",
)
_add(
    "guaiacol",
    "Guaiacol",
    ["2-methoxyphenol", "o-methoxyphenol"],
    ChemicalProperties(
        molecular_weight=m(124.14),
        boiling_point=m(205),
        vapor_pressure_25=e(20),
        functional_groups=["phenol", "ether"],
        redox_active=True,
        odor_descriptor="smoky, medicinal",
    ),
    ["PubChem CID 460"],
    cas="90-05-1",
    smiles="COc1ccccc1O",
)
_add(
    "phenol",
    "Phenol",
    ["carbolic acid", "hydroxybenzene"],
    ChemicalProperties(
        molecular_weight=m(94.11),
        boiling_point=m(181.8),
        vapor_pressure_25=e(47),
        functional_groups=["phenol", "aromatic"],
        redox_active=True,
        odor_descriptor="phenolic, tar-like",
    ),
    ["PubChem CID 996"],
    cas="108-95-2",
    smiles="Oc1ccccc1",
)
_add(
    "furfural",
    "Furfural",
    ["furfuraldehyde", "2-furaldehyde"],
    ChemicalProperties(
        molecular_weight=m(96.08),
        boiling_point=m(161.7),
        vapor_pressure_25=e(260),
        functional_groups=["aldehyde", "furan", "aromatic"],
        redox_active=True,
        odor_descriptor="almond, bready, caramel",
    ),
    ["PubChem CID 7362"],
    cas="98-01-1",
    smiles="O=Cc1ccco1",
)
_add(
    "diacetyl",
    "2,3-Butanedione",
    ["diacetyl", "dimethyl diketone"],
    ChemicalProperties(
        molecular_weight=m(86.09),
        boiling_point=m(88),
        vapor_pressure_25=e(7000),
        functional_groups=["ketone", "diketone"],
        redox_active=True,
        odor_descriptor="buttery, creamy",
    ),
    ["PubChem CID 650"],
    cas="431-03-8",
    smiles="CC(=O)C(C)=O",
)
_add(
    "diallyl-disulfide",
    "Diallyl disulfide",
    ["garlic oil", "4,5-dithia-1,7-octadiene"],
    ChemicalProperties(
        molecular_weight=m(146.27),
        boiling_point=e(177),
        vapor_pressure_25=e(130),
        functional_groups=["thioether", "sulfur", "alkene"],
        redox_active=True,
        odor_descriptor="garlic, pungent",
    ),
    ["PubChem CID 16590"],
    cas="2179-57-9",
    smiles="C=CCSSCC=C",
)
_add(
    "ammonia",
    "Ammonia",
    ["NH3", "ammonium gas"],
    ChemicalProperties(
        molecular_weight=m(17.03),
        boiling_point=m(-33.3),
        vapor_pressure_25=u(),
        gas=True,
        functional_groups=["amine"],
        redox_active=True,
        odor_descriptor="pungent, sharp",
    ),
    ["PubChem CID 222"],
    cas="7664-41-7",
    smiles="N",
)
_add(
    "methane",
    "Methane",
    ["natural gas", "CH4"],
    ChemicalProperties(
        molecular_weight=m(16.04),
        boiling_point=m(-161.5),
        vapor_pressure_25=u(),
        gas=True,
        functional_groups=["alkane"],
        redox_active=True,
        odor_descriptor="odorless (odorized in supply)",
    ),
    ["PubChem CID 297"],
    cas="74-82-8",
    smiles="C",
)
_add(
    "propane",
    "Propane",
    ["LPG", "liquefied petroleum gas"],
    ChemicalProperties(
        molecular_weight=m(44.1),
        boiling_point=m(-42.1),
        vapor_pressure_25=u(),
        gas=True,
        functional_groups=["alkane"],
        redox_active=True,
        odor_descriptor="odorless (odorized)",
    ),
    ["PubChem CID 6334"],
    cas="74-98-6",
    smiles="CCC",
)
_add(
    "butane",
    "n-Butane",
    ["butane", "liquefied gas"],
    ChemicalProperties(
        molecular_weight=m(58.12),
        boiling_point=m(-0.5),
        vapor_pressure_25=e(240000),
        gas=True,
        functional_groups=["alkane"],
        redox_active=True,
        odor_descriptor="odorless (odorized)",
    ),
    ["PubChem CID 7843"],
    cas="106-97-8",
    smiles="CCCC",
)
_add(
    "hydrogen-sulfide",
    "Hydrogen sulfide",
    ["H2S", "sulfane", "sewer gas"],
    ChemicalProperties(
        molecular_weight=m(34.08),
        boiling_point=m(-60.3),
        vapor_pressure_25=u(),
        gas=True,
        functional_groups=["thiol", "sulfur"],
        redox_active=True,
        odor_descriptor="rotten eggs",
    ),
    ["PubChem CID 402"],
    cas="7783-06-4",
    smiles="S",
)
_add(
    "methanethiol",
    "Methanethiol",
    ["methyl mercaptan", "mercaptomethane"],
    ChemicalProperties(
        molecular_weight=m(48.11),
        boiling_point=m(5.95),
        vapor_pressure_25=u(),
        gas=True,
        functional_groups=["thiol", "sulfur"],
        redox_active=True,
        odor_descriptor="rotten cabbage, sulfurous",
    ),
    ["PubChem CID 855"],
    cas="74-93-1",
    smiles="CS",
)
_add(
    "dimethyl-sulfide",
    "Dimethyl sulfide",
    ["DMS", "methylthiomethane"],
    ChemicalProperties(
        molecular_weight=m(62.13),
        boiling_point=m(37.3),
        vapor_pressure_25=e(64000),
        functional_groups=["thioether", "sulfur"],
        redox_active=True,
        odor_descriptor="seafood, cabbage, sulfurous",
    ),
    ["PubChem CID 1068"],
    cas="75-18-3",
    smiles="CSC",
)
_add(
    "ethylene",
    "Ethylene",
    ["ethene", "fruit ripening gas"],
    ChemicalProperties(
        molecular_weight=m(28.05),
        boiling_point=m(-103.7),
        vapor_pressure_25=u(),
        gas=True,
        functional_groups=["alkene"],
        redox_active=True,
        odor_descriptor="sweet, ethereal",
    ),
    ["PubChem CID 6325"],
    cas="74-85-1",
    smiles="C=C",
)
_add(
    "carbon-monoxide",
    "Carbon monoxide",
    ["CO", "exhaust gas"],
    ChemicalProperties(
        molecular_weight=m(28.01),
        boiling_point=m(-191.5),
        vapor_pressure_25=u(),
        gas=True,
        functional_groups=["inorganic"],
        redox_active=True,
        odor_descriptor="odorless",
    ),
    ["PubChem CID 281"],
    cas="630-08-0",
    smiles="[C-]#[O+]",
)
_add(
    "carbon-dioxide",
    "Carbon dioxide",
    ["CO2", "carbonic anhydride"],
    ChemicalProperties(
        molecular_weight=m(44.01),
        boiling_point=m(-78.5),
        vapor_pressure_25=u(),
        gas=True,
        functional_groups=["inorganic"],
        redox_active=False,
        non_redox=True,
        odor_descriptor="odorless",
    ),
    ["PubChem CID 280"],
    cas="124-38-9",
    smiles="O=C=O",
)
_add(
    "nitrogen",
    "Nitrogen",
    ["N2", "dinitrogen"],
    ChemicalProperties(
        molecular_weight=m(28.01),
        boiling_point=m(-195.8),
        vapor_pressure_25=u(),
        gas=True,
        functional_groups=["inorganic"],
        redox_active=False,
        non_redox=True,
        odor_descriptor="odorless",
    ),
    ["PubChem CID 947"],
    cas="7727-37-9",
    smiles="N#N",
)
_add(
    "oxygen",
    "Oxygen",
    ["O2", "dioxygen"],
    ChemicalProperties(
        molecular_weight=m(32.0),
        boiling_point=m(-183),
        vapor_pressure_25=u(),
        gas=True,
        functional_groups=["inorganic"],
        redox_active=False,
        non_redox=False,
        odor_descriptor="odorless",
    ),
    ["PubChem CID 977"],
    cas="7782-44-7",
    smiles="O=O",
)
_add(
    "water",
    "Water",
    ["H2O", "water vapor", "humidity"],
    ChemicalProperties(
        molecular_weight=m(18.02),
        boiling_point=m(100),
        vapor_pressure_25=m(3170),
        functional_groups=["inorganic"],
        redox_active=False,
        non_redox=False,
        odor_descriptor="odorless (humidity response)",
    ),
    ["PubChem CID 962"],
    cas="7732-18-5",
    smiles="O",
)

COMPOUND_BY_ID = {c.id: c for c in COMPOUNDS}
REFERENCE_COMPOUND = COMPOUND_BY_ID.get("ethanol")

if REFERENCE_COMPOUND is None:
    raise RuntimeError("Reference compound 'ethanol' missing from bundled dataset")
