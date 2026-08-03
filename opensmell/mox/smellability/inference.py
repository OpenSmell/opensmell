"""Offline structure-in chemical inference (SMILES -> Chemical, no live lookups).

The public entry point is `chemical_from_smiles()`, which builds a `Chemical`
with `estimated` molecular weight, boiling point, functional groups, and redox
classification straight from a SMILES string. It never touches the network:

* molecular weight is computed from the SMILES atom graph (exact, offline);
* boiling point uses the Joback group-contribution table (vendored 40-group);
* functional groups reuse `groups.infer_functional_groups`;
* redox activity follows the rule used by the OpenSmell validation scratch: any
  carbon plus a recognized functional group is redox-active at MOX operating
  temperature, while true inerts (N2, O2, CO2, noble gases, water, ammonia) are
  not.

RDKit is optional. When it is importable it canonicalizes the SMILES and runs
the exact SMARTS group matching used by the `thermo` package (MIT, Caleb Bell;
SMARTS by Jason Biggs). Without it, a pure-Python decomposition of the shared
SMILES scanner (see `groups.py`) approximates the same Joback counts, so the
SDK's structure-in path works in any environment that can import the package.

Error bounds: Joback boiling-point estimates carry a mean absolute error of
roughly 27 °C versus NIST values on the 716-VOC validation set (inorganic
outliers excluded). The chain back-computes vapor pressure from Tb via
Clausius-Clapeyron + Trouton, so headspace estimates inherit that log-scale
uncertainty — use these numbers for triage, not calibration.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from .groups import _scan_smiles, infer_functional_groups, kekule_to_aromatic
from .types import Chemical, ChemicalProperties, Property

JOBACK_BASE_K = 198.2

# (name, RDKit SMARTS, Tb contribution in K) — Joback 1984, 40 groups.
# SMARTS from the `thermo` package (MIT, Caleb Bell 2017-2020; SMARTS by Jason
# Biggs). Vendored from the opensmell validation work and cross-checked against
# NIST boiling points (MAE ~= 27 K on 716 VOCs, inorganics excluded).
JOBACK_GROUPS: List[Tuple[str, str, float]] = [
    ("-CH3", "[CX4H3]", 23.58),
    ("-CH2-", "[!R;CX4H2]", 22.88),
    (">CH-", "[!R;CX4H]", 21.74),
    (">C<", "[!R;CX4H0]", 18.25),
    ("=CH2", "[CX3H2]", 18.18),
    ("=CH-", "[!R;CX3H1;!$([CX3H1](=O))]", 24.96),
    ("=C<", "[$([!R;#6X3H0]);!$([!R;#6X3H0]=[#8])]", 24.14),
    ("=C=", "[$([CX2H0](=*)=*)]", 26.15),
    ("C#CH", "[$([CX2H1]#[!#7])]", 9.2),
    ("C#C-", "[$([CX2H0]#[!#7])]", 27.38),
    ("-CH2- (ring)", "[R;CX4H2]", 27.15),
    (">CH- (ring)", "[R;CX4H]", 21.78),
    (">C< (ring)", "[R;CX4H0]", 21.32),
    ("=CH- (ring)", "[R;CX3H1,cX3H1]", 26.73),
    ("=C< (ring)", "[$([R;#6X3H0]);!$([R;#6X3H0]=[#8])]", 31.01),
    ("-F", "[F]", -0.03),
    ("-Cl", "[Cl]", 38.13),
    ("-Br", "[Br]", 66.86),
    ("-I", "[I]", 93.84),
    ("-OH (alcohol)", "[OX2H;!$([OX2H]-[#6]=[O]);!$([OX2H]-a)]", 92.88),
    ("-OH (phenol)", "[$([OX2H]-a)]", 76.34),
    ("-O- (nonring)", "[OX2H0;!R;!$([OX2H0]-[#6]=[#8])]", 22.42),
    ("-O- (ring)", "[#8X2H0;R;!$([#8X2H0]~[#6]=[#8])]", 31.22),
    (">C=O (nonring)", "[$([CX3H0](=[OX1]));!$([CX3](=[OX1])-[OX2]);!R]=O", 76.75),
    (">C=O (ring)", "[$([#6X3H0](=[OX1]));!$([#6X3](=[#8X1])~[#8X2]);R]=O", 94.97),
    ("O=CH- (aldehyde)", "[CX3H1](=O)[#6]", 72.24),
    ("-COOH (acid)", "[OX2H]-[C]=O", 169.09),
    ("-COO- (ester)", "[#6X3H0;!$([#6X3H0](~O)(~O)(~O))](=[#8X1])[#8X2H0]", 81.1),
    ("=O (other)", "[OX1H0;!$([OX1H0]~[#6X3]);!$([OX1H0]~[#7X3]~[#8])]", -10.5),
    ("-NH2", "[NX3H2]", 73.23),
    (">NH (nonring)", "[NX3H1;!R]", 50.17),
    (">NH (ring)", "[#7X3H1;R]", 52.82),
    (">N- (nonring)", "[#7X3H0;!$([#7](~O)~O)]", 11.74),
    ("-N= (nonring)", "[#7X2H0;!R]", 74.6),
    ("-N= (ring)", "[#7X2H0;R]", 57.55),
    ("=NH", "[#7X2H1]", 83.08),
    ("-CN", "[#6X2]#[#7X1H0]", 125.66),
    ("-NO2", "[$([#7X3,#7X3+][!#8])](=[O])~[O-]", 152.54),
    ("-SH", "[SX2H]", 63.56),
    ("-S- (nonring)", "[#16X2H0;!R]", 68.78),
    ("-S- (ring)", "[#16X2H0;R]", 52.1),
]

JOBACK_TB_BY_GROUP: Dict[str, float] = {name: tb for name, _, tb in JOBACK_GROUPS}

_ATOMIC_WEIGHTS: Dict[str, float] = {
    "H": 1.008,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "Na": 22.99,
    "Mg": 24.305,
    "Al": 26.982,
    "Si": 28.085,
    "P": 30.974,
    "S": 32.06,
    "Cl": 35.45,
    "K": 39.098,
    "Ca": 40.078,
    "Fe": 55.845,
    "Se": 78.971,
    "Br": 79.904,
    "I": 126.904,
}

_BRACKET = re.compile(
    r"^\[(\d+)?([A-Z][a-z]?|[a-z])(?:@|@@)*(?:H(\d*))?(?:[+-]\d*)?\]$"
)


def _element_h(el_str: str) -> Tuple[str, int, bool]:
    """(element, explicit_H, aromatic) from a scanner atom element string."""
    if el_str.startswith("["):
        m = _BRACKET.match(el_str)
        if m:
            h_raw = m.group(3)
            if h_raw is None:
                explicit_h = 0
            elif h_raw == "":
                explicit_h = 1
            else:
                explicit_h = int(h_raw)
            el = m.group(2)
            return (el.upper(), explicit_h, el.islower())
        body = el_str[1:-1]
        h_m = re.search(r"H(\d*)", body)
        if h_m:
            explicit_h = 1 if h_m.group(1) == "" else int(h_m.group(1))
        else:
            explicit_h = 0
        el_m = re.match(r"[A-Z][a-z]?|[a-z]", body)
        el = el_m.group(0) if el_m else "?"
        return (el.upper(), explicit_h, el.islower())
    if el_str.islower():
        return (el_str.upper(), 0, True)
    return (el_str, 0, False)


def _valence_h(el: str, aromatic: bool, bond_sum: int, bond_count: int) -> int:
    if aromatic:
        if el == "C":
            return max(0, 3 - bond_count)
        return 0
    if el == "C":
        return max(0, 4 - bond_sum)
    if el == "N":
        return max(0, 3 - bond_sum)
    if el == "O":
        return max(0, 2 - bond_sum)
    if el == "S":
        return max(0, 2 - bond_sum)
    if el == "P":
        return max(0, 3 - bond_sum)
    return 0


class _AtomInfo:
    __slots__ = ("el", "aromatic", "in_ring", "h", "heavy", "all_orders")

    def __init__(self, el, aromatic, in_ring, h, heavy, all_orders):
        self.el = el
        self.aromatic = aromatic
        self.in_ring = in_ring
        self.h = h
        self.heavy = heavy
        self.all_orders = all_orders


def _parse_graph(smiles: str) -> List[_AtomInfo]:
    """Atom graph with implicit-H counts and ring membership, pure Python."""
    norm = kekule_to_aromatic(smiles.strip())
    atoms, _ = _scan_smiles(norm)
    if not atoms:
        raise ValueError(f"could not parse SMILES: {smiles!r}")

    # Ring membership by bridge detection (undirected). Atoms on a simple cycle
    # are exactly those incident to a non-bridge edge. A DFS shortest-path per
    # closure pair is not enough (it can wander through the *other* ring of a
    # spiro/fused system and miss true ring atoms), and leaf removal wrongly
    # keeps degree-2 bridges between rings (S-S, aryl-O-aryl, Ar-CH2-Ar).
    adj: Dict[int, set] = {i: set() for i in range(len(atoms))}
    for i, a in enumerate(atoms):
        for e in a.edges:
            adj[i].add(e.to)
            adj[e.to].add(i)
    ring_membership = _ring_membership(adj)

    parsed = [_element_h(a.el) for a in atoms]
    infos: List[_AtomInfo] = []
    for i, a in enumerate(atoms):
        el, explicit_h, arom = parsed[i]
        all_bonds = [(e.to, e.bond) for e in a.edges]
        heavy = [(e.to, e.bond) for e in a.edges if parsed[e.to][0] != "H"]
        bond_sum = sum(o for _, o in all_bonds)
        if el == "H":
            h = 1
        elif a.el.startswith("["):
            h = explicit_h
        else:
            h = _valence_h(el, arom, bond_sum, len(heavy))
        infos.append(
            _AtomInfo(
                el=el,
                aromatic=arom,
                in_ring=i in ring_membership,
                h=h,
                heavy=heavy,
                all_orders=[o for _, o in all_bonds],
            )
        )
    return infos


def _ring_membership(adj: Dict[int, set]) -> set:
    """Indices of atoms that lie on at least one simple cycle."""
    n = len(adj)
    all_edges = {(i, j) for i in range(n) for j in adj[i] if i < j}

    def component_count(skip: Optional[set] = None) -> int:
        seen: set = set()
        count = 0
        for s in range(n):
            if s in seen:
                continue
            count += 1
            stack = [s]
            seen.add(s)
            while stack:
                cur = stack.pop()
                for nb in adj[cur]:
                    if skip is not None and (
                        (cur, nb) in skip or (nb, cur) in skip
                    ):
                        continue
                    if nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        return count

    base = component_count()
    bridges: set = set()
    for i, j in all_edges:
        if component_count(skip={(i, j)}) > base:
            bridges.add((i, j))
    ring_atoms: set = set()
    for i, j in all_edges - bridges:
        ring_atoms.add(i)
        ring_atoms.add(j)
    return ring_atoms


def estimate_molecular_weight(smiles: str) -> Optional[float]:
    """Average molecular weight (g/mol) from the atom graph, pure Python."""
    infos = _parse_graph(smiles)
    total = 0.0
    hydrogens = 0
    for info in infos:
        if info.el == "H":
            hydrogens += 1
            continue
        weight = _ATOMIC_WEIGHTS.get(info.el)
        if weight is None:
            return None
        total += weight
        hydrogens += info.h
    return total + hydrogens * 1.008


def _joback_counts_pure(smiles: str) -> Dict[str, int]:
    infos = _parse_graph(smiles)
    counts: Counter = Counter()
    used: set[int] = set()

    for i, info in enumerate(infos):
        if i in used or info.el != "C":
            continue
        if info.aromatic and not any(
            o == 2 and infos[to].el == "O" for to, o in info.heavy
        ):
            # An "aromatic" carbon carrying an exocyclic C=O (e.g. the lactone
            # carbonyl written `O=c1` in coumarin) is an sp2 carbonyl in RDKit,
            # excluded from `=C< (ring)`; fall through to the carbonyl logic.
            counts["=CH- (ring)" if info.h == 1 else "=C< (ring)"] += 1
            used.add(i)
            continue
        orders = info.all_orders
        if 3 in orders:
            nitrile = any(
                infos[to].el == "N" for to, o in info.heavy if o == 3
            )
            if nitrile:
                counts["-CN"] += 1
            else:
                counts["C#CH" if info.h == 1 else "C#C-"] += 1
            used.add(i)
            continue
        dbl_o = [to for to, o in info.heavy if o == 2 and infos[to].el == "O"]
        if dbl_o:
            single_o = [to for to, o in info.heavy if o == 1 and infos[to].el == "O"]
            oh_o = [to for to in single_o if infos[to].h >= 1]
            est_o = [
                to
                for to in single_o
                if infos[to].h == 0 and len(infos[to].heavy) >= 2
            ]
            carbon_single = [
                to for to, o in info.heavy if o == 1 and infos[to].el == "C"
            ]
            if oh_o:
                counts["-COOH (acid)"] += 1
                used.update([i] + dbl_o + oh_o)
                continue
            if info.h == 0 and len(single_o) == 1 and est_o:
                counts["-COO- (ester)"] += 1
                used.update([i] + dbl_o + est_o)
                continue
            # A carbonate (two single-bonded O) matches none of the carbonyl
            # SMARTS, and a formate (O=C-O with an H-bearing carbonyl carbon)
            # matches neither the ester (H0 required) nor the aldehyde (a
            # carbon neighbor required) patterns. RDKit contributes nothing for
            # either, so neither does this decomposition.
            if len(single_o) >= 2:
                continue
            if info.h == 1 and len(info.heavy) == 2 and carbon_single:
                counts["O=CH- (aldehyde)"] += 1
                used.update([i] + dbl_o)
                continue
            if info.h == 0 and len(info.heavy) == 3:
                counts[">C=O (ring)" if info.in_ring else ">C=O (nonring)"] += 1
                used.update([i] + dbl_o)
                continue
            continue
        dbl_any = [to for to, o in info.heavy if o == 2]
        if len(dbl_any) >= 2:
            # `[$([CX2H0](=*)=*)]`: cumulated double bonds to any atoms, e.g.
            # isothiocyanate S=C=N- (RDKit does not exclude these from `=C=`).
            counts["=C="] += 1
            used.add(i)
            continue
        if dbl_any:
            # One double bond to a non-carbonyl atom (C, N, S, ...). RDKit's
            # `=C<`/`=CH-`/`=CH2` groups only exclude carbonyl carbons
            # (`=[#8]`), so e.g. C=N imines and C=S xanthates count as
            # `=C< (ring)` / `=C<` rather than plain `>C<`.
            if info.in_ring:
                counts["=CH- (ring)" if info.h == 1 else "=C< (ring)"] += 1
            else:
                counts["=CH2" if info.h == 2 else ("=CH-" if info.h == 1 else "=C<")] += 1
            used.add(i)
            continue
        if info.in_ring:
            if info.h == 2:
                counts["-CH2- (ring)"] += 1
            elif info.h == 1:
                counts[">CH- (ring)"] += 1
            elif info.h == 0:
                counts[">C< (ring)"] += 1
        else:
            if info.h == 3:
                counts["-CH3"] += 1
            elif info.h == 2:
                counts["-CH2-"] += 1
            elif info.h == 1:
                counts[">CH-"] += 1
            elif info.h == 0:
                counts[">C<"] += 1
        used.add(i)

    for i, info in enumerate(infos):
        if i in used:
            continue
        el, h, heavy, in_ring = info.el, info.h, info.heavy, info.in_ring
        if el == "O":
            if len(heavy) == 0:
                used.add(i)
                continue
            if h >= 1 and len(heavy) == 1:
                nbr = infos[heavy[0][0]]
                counts["-OH (phenol)" if nbr.aromatic else "-OH (alcohol)"] += 1
                used.add(i)
                continue
            if h == 0 and len(heavy) == 2:
                # Mirror `!$([OX2H0]-[#6]=[#8])` (and its ring variant
                # `!$([#8X2H0]~[#6]=[#8])`): an ether O whose carbon neighbor
                # is a carbonyl carbon (e.g. a formate or anhydride O) is not
                # an ether and contributes nothing in RDKit.
                if any(
                    infos[to].el == "C"
                    and any(
                        o == 2 and infos[ot].el == "O"
                        for ot, o in infos[to].heavy
                    )
                    for to, _ in heavy
                ):
                    used.add(i)
                    continue
                counts["-O- (ring)" if in_ring else "-O- (nonring)"] += 1
                used.add(i)
                continue
            used.add(i)
            continue
        if el == "N":
            o_doubles = [to for to, o in heavy if o == 2 and infos[to].el == "O"]
            if o_doubles and len(heavy) >= 2:
                counts["-NO2"] += 1
                used.add(i)
                continue
            if any(o == 3 for _, o in heavy):
                used.add(i)
                continue
            if h == 2:
                counts["-NH2"] += 1
            elif h == 1:
                counts[">NH (ring)" if in_ring else ">NH (nonring)"] += 1
            elif h == 0:
                if len(heavy) == 3:
                    counts[">N- (nonring)"] += 1
                elif info.aromatic or any(o == 2 for _, o in heavy):
                    counts["-N= (ring)" if in_ring else "-N= (nonring)"] += 1
                else:
                    counts[">N- (nonring)"] += 1
            used.add(i)
            continue
        if el == "S":
            if h >= 1:
                counts["-SH"] += 1
            elif len(heavy) == 2:
                counts["-S- (ring)" if in_ring else "-S- (nonring)"] += 1
            used.add(i)
            continue
        if el in ("F", "Cl", "Br", "I"):
            counts[f"-{el}"] += 1
            used.add(i)
            continue
        used.add(i)

    for i, info in enumerate(infos):
        if i in used or info.el != "O" or info.h != 0 or len(info.heavy) != 1:
            continue
        nbr = infos[info.heavy[0][0]]
        if nbr.el == "N" and any(
            infos[to].el == "O" for to, o in nbr.heavy if o == 2
        ):
            continue
        # A terminal O on a CX3 carbonyl carbon is part of a classified or
        # skipped carbonyl (e.g. a carbonate), never "=O (other)".
        if nbr.el == "C" and len(nbr.heavy) == 3:
            continue
        counts["=O (other)"] += 1
        used.add(i)

    return dict(counts)


def joback_boiling_point_k(smiles: str) -> Optional[float]:
    """Joback Tb (K) via the pure-Python group decomposition."""
    try:
        counts = _joback_counts_pure(smiles)
    except ValueError:
        return None
    return JOBACK_BASE_K + sum(
        n * JOBACK_TB_BY_GROUP[name] for name, n in counts.items()
    )


def _rdkit() -> Optional[Tuple]:
    try:
        from rdkit import Chem
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
        return (Chem,)
    except Exception:
        return None


_RDKIT_CACHE = {}


def _rdkit_boiling_point_k(canonical: str) -> Optional[float]:
    """Joback Tb (K) using RDKit SMARTS matching (the `thermo`-derived path)."""
    rdk = _rdkit()
    if rdk is None:
        return None
    (Chem,) = rdk
    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        return None
    total = JOBACK_BASE_K
    for name, smarts, tb in JOBACK_GROUPS:
        patt = _RDKIT_CACHE.get(smarts)
        if patt is None:
            patt = Chem.MolFromSmarts(smarts)
            _RDKIT_CACHE[smarts] = patt
        if patt is None:
            continue
        total += len(mol.GetSubstructMatches(patt)) * tb
    return total


def _canonical_smiles(smiles: str) -> str:
    rdk = _rdkit()
    if rdk is not None:
        (Chem,) = rdk
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
    return smiles.strip()


def _has_carbon(infos: List[_AtomInfo]) -> bool:
    return any(info.el == "C" for info in infos)


def chemical_from_smiles(
    smiles: str,
    name: Optional[str] = None,
    cas: Optional[str] = None,
    source_refs: Optional[List[str]] = None,
    prefer_rdkit: bool = True,
) -> Chemical:
    """Build an `estimated` `Chemical` from a SMILES string, fully offline.

    Raises `ValueError` when the SMILES cannot be parsed.
    """
    canonical = _canonical_smiles(smiles) if prefer_rdkit else smiles.strip()

    mw = estimate_molecular_weight(canonical)
    if mw is None:
        raise ValueError(f"could not parse SMILES: {smiles!r}")

    tb_k = None
    used_rdkit = False
    if prefer_rdkit:
        tb_k = _rdkit_boiling_point_k(canonical)
        used_rdkit = tb_k is not None
    if tb_k is None:
        tb_k = joback_boiling_point_k(canonical)
    tb_c = tb_k - 273.15

    infos = _parse_graph(canonical)
    groups = infer_functional_groups(canonical)

    # Redox is decided from the structural Joback decomposition, not the
    # string-heuristic groups: CO2 ("O=C=O") looks like an aldehyde to the
    # heuristic (`C=O$`), but structurally it is an inert oxide with no
    # carbon-bearing functional group. "=O (other)" and the halogens are the
    # only groups that never imply carbon chemistry.
    joback_counts = _joback_counts_pure(canonical)
    organic = any(
        k not in ("=O (other)", "-F", "-Cl", "-Br", "-I") for k in joback_counts
    )
    if _has_carbon(infos) and organic:
        redox_active = True
        non_redox = None
    else:
        redox_active = False
        if canonical in ("O", "O=O", "N#N", "[NH3]"):
            non_redox = False
        else:
            non_redox = True

    note = (
        "Joback group contributions (RDKit SMARTS); MAE ~27 °C vs NIST"
        if used_rdkit
        else "Joback group contributions (pure-Python decomposition); MAE ~27 °C vs NIST"
    )

    return Chemical(
        id=f"inf-{hashlib.sha1(canonical.encode()).hexdigest()[:10]}",
        name=name or canonical,
        synonyms=[name or canonical],
        cas=cas,
        smiles=canonical,
        source_refs=source_refs or ["RDKit/Joback (inferred)"],
        props=ChemicalProperties(
            molecular_weight=Property(
                value=round(mw, 2),
                source="estimated",
                note="computed from the SMILES atom graph",
            ),
            boiling_point=Property(
                value=round(tb_c, 1),
                source="estimated",
                note=note,
            ),
            vapor_pressure_25=Property(
                value=None,
                source="unknown",
                note="back-computed by the chain from Tb (Clausius-Clapeyron + Trouton)",
            ),
            functional_groups=groups,
            redox_active=redox_active,
            non_redox=non_redox,
            gas=bool(tb_c < 25.0),
        ),
    )
