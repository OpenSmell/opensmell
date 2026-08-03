"""Structural functional-group inference from a SMILES string.

Mirrors `osmograph-web/lib/smellability/groups.ts` 1:1. A deliberately
conservative heuristic, not a full SMILES parser. Aromatic rings arrive in two
notations — lowercase `c1ccccc1` (curated dictionary) and Kekulé `C1=CC=CC=C1`
(the form PubChem's property endpoint returns) — and are normalized so a
live-fetched provisional chemical runs through the same group inference as a
curated one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class _Edge:
    to: int
    bond: int
    edge: int
    dpos: Optional[int] = None


@dataclass
class _Atom:
    el: str
    aromatic: bool
    start: int
    length: int
    edges: List[_Edge] = field(default_factory=list)


def _scan_smiles(smiles: str) -> Tuple[List[_Atom], List[Tuple[int, int]]]:
    atoms: List[_Atom] = []
    digit_to_atoms: dict[int, List[int]] = {}
    edge_records: List[Tuple[int, int, int, Optional[int]]] = []
    stack: List[int] = []
    last = -1
    pending_bond = 1
    pending_dpos: Optional[int] = None

    def push_edge(frm: int, to: int, bond: int, dpos: Optional[int] = None) -> None:
        edge = len(edge_records)
        edge_records.append((frm, to, bond, dpos))
        atoms[frm].edges.append(_Edge(to=to, bond=bond, edge=edge, dpos=dpos))
        atoms[to].edges.append(_Edge(to=frm, bond=bond, edge=edge, dpos=dpos))

    i = 0
    n = len(smiles)
    while i < n:
        ch = smiles[i]
        if ch == "(":
            stack.append(last)
            i += 1
            pending_bond = 1
            pending_dpos = None
            continue
        if ch == ")":
            last = stack.pop() if stack else -1
            i += 1
            pending_bond = 1
            pending_dpos = None
            continue
        if ch == "=":
            pending_bond = 2
            pending_dpos = i
            i += 1
            continue
        if ch == "#":
            pending_bond = 3
            pending_dpos = None
            i += 1
            continue
        if ch in "-/\\":
            pending_bond = 1
            pending_dpos = None
            i += 1
            continue
        if ch == ".":
            # Disconnected fragment (e.g. `[S-].[K+]` salts): no bond is formed
            # with the previous atom, so the "last" pointer must reset.
            last = -1
            pending_bond = 1
            pending_dpos = None
            i += 1
            continue
        if ch == "%":
            d = int(smiles[i + 1 : i + 3])
            if last >= 0:
                digit_to_atoms.setdefault(d, []).append(last)
            i += 3
            continue
        if ch.isdigit():
            d = int(ch)
            if last >= 0:
                digit_to_atoms.setdefault(d, []).append(last)
            i += 1
            continue
        if ch == "[":
            end = smiles.index("]", i)
            idx = len(atoms)
            bracketed = smiles[i : end + 1]
            atoms.append(_Atom(el=bracketed, aromatic=bool(re.match(r"^\[[a-z]", bracketed)), start=i, length=end + 1 - i, edges=[]))
            if last >= 0:
                push_edge(last, idx, pending_bond, pending_dpos)
            last = idx
            pending_bond = 1
            pending_dpos = None
            i = end + 1
            continue
        if ch.isalpha():
            el = ch
            j = i + 1
            if (ch == "C" and j < n and smiles[j] == "l") or (ch == "B" and j < n and smiles[j] == "r"):
                el += smiles[j]
                j += 1
            idx = len(atoms)
            atoms.append(_Atom(el=el, aromatic=bool(re.search(r"[cnosp]", el)), start=i, length=j - i, edges=[]))
            if last >= 0:
                push_edge(last, idx, pending_bond, pending_dpos)
            last = idx
            pending_bond = 1
            pending_dpos = None
            i = j
            continue
        i += 1

    # Ring closures: an atom written `C1` and a later `C1` (or `=C1`) close the
    # same ring. The closure bond is single in Kekulé notation. A digit may be
    # reused for a later ring (e.g. dibenzofuran `c1ccc2c(c1)oc1ccccc12`), so
    # occurrences pair up sequentially: 1st-2nd, 3rd-4th, and so on.
    ring_pairs: List[Tuple[int, int]] = []
    for occ in digit_to_atoms.values():
        for k in range(1, len(occ), 2):
            x, y = occ[k - 1], occ[k]
            ring_pairs.append((x, y))
            push_edge(x, y, 1)

    return atoms, ring_pairs


def _find_path_between(from_: int, to: int, atoms: List[_Atom], skip_edge: int) -> Optional[List[int]]:
    visited = {from_}
    prev: dict[int, int] = {}
    queue: List[int] = [from_]
    while queue:
        cur = queue.pop()
        if cur == to:
            break
        for e in atoms[cur].edges:
            if e.edge == skip_edge:
                continue
            if e.to not in visited:
                visited.add(e.to)
                prev[e.to] = cur
                queue.append(e.to)
    if to not in visited:
        return None
    path: List[int] = []
    cur: Optional[int] = to
    while cur is not None:
        path.append(cur)
        if cur == from_:
            break
        cur = prev.get(cur)
    return list(reversed(path))


def _edge_bond_between(a: int, b: int, atoms: List[_Atom]) -> Optional[_Edge]:
    for e in atoms[a].edges:
        if e.to == b:
            return e
    return None


def _is_kekule_aromatic_ring(path: List[int], atoms: List[_Atom]) -> bool:
    if len(path) != 6 and len(path) != 5:
        return False

    els = [atoms[idx].el for idx in path]
    if len(path) == 6:
        if not all(el == "C" for el in els):
            return False
    else:
        carbons = sum(1 for el in els if el == "C")
        heteros = sum(1 for el in els if re.fullmatch(r"[NOSnos]", el))
        if carbons != 4 or heteros != 1:
            return False

    bonds: List[int] = []
    for k in range(len(path)):
        edge = _edge_bond_between(path[k], path[(k + 1) % len(path)], atoms)
        if edge is None:
            return False
        bonds.append(edge.bond)

    doubles = [k for k, b in enumerate(bonds) if b == 2]
    want = 3 if len(path) == 6 else 2
    if len(doubles) != want:
        return False
    for d in doubles:
        prev = (d - 1 + len(bonds)) % len(bonds)
        nxt = (d + 1) % len(bonds)
        if bonds[prev] == 2 or bonds[nxt] == 2:
            return False
    return True


def _analyze(smiles: str) -> Tuple[str, bool, bool]:
    atoms, ring_pairs = _scan_smiles(smiles)

    ring_atoms: set[int] = set()
    ring_hetero: set[str] = set()
    has_lowercase_aromatic = any(a.aromatic for a in atoms)
    dropped_equals: set[int] = set()

    if has_lowercase_aromatic:
        for idx, a in enumerate(atoms):
            if a.aromatic:
                ring_atoms.add(idx)
    else:
        for x, y in ring_pairs:
            closure_edge = next((e for e in atoms[x].edges if e.to == y), None)
            if closure_edge is None:
                continue
            path = _find_path_between(x, y, atoms, closure_edge.edge)
            if path is None:
                continue
            if not _is_kekule_aromatic_ring(path, atoms):
                continue
            for idx in path:
                ring_atoms.add(idx)
            for idx in path:
                if re.fullmatch(r"[NOSnos]", atoms[idx].el):
                    ring_hetero.add(atoms[idx].el)
            for k in range(len(path)):
                edge = _edge_bond_between(path[k], path[(k + 1) % len(path)], atoms)
                if edge is not None and edge.bond == 2 and edge.dpos is not None:
                    dropped_equals.add(edge.dpos)

    phenol = False
    for idx, a in enumerate(atoms):
        if a.el != "O":
            continue
        singles = [e for e in a.edges if e.bond == 1]
        if len(singles) != 1:
            continue
        if singles[0].to in ring_atoms:
            phenol = True
            break

    if len(ring_atoms) == 0:
        return smiles, phenol, False

    out = ""
    for pos in range(len(smiles)):
        if pos in dropped_equals:
            continue
        replaced = False
        for idx in ring_atoms:
            a = atoms[idx]
            if a.el == "C" and pos == a.start:
                out += "c"
                replaced = True
                break
        if not replaced:
            out += smiles[pos]
    return out, phenol, ("O" in ring_hetero or "o" in ring_hetero)


def kekule_to_aromatic(smiles: str) -> str:
    return _analyze(smiles)[0]


def infer_functional_groups(smiles: Optional[str]) -> List[str]:
    if not smiles:
        return []
    s, phenol, furan = _analyze(smiles.strip())

    groups: set[str] = set()

    has_aromatic = re.search(r"c[1-9]", s) is not None
    if has_aromatic:
        groups.add("aromatic")

    # Hetero-only small molecules that the chain treats specially.
    if re.fullmatch(r"S", s):
        groups.add("sulfur")
        return list(groups)
    if re.fullmatch(r"N", s):
        groups.add("amine")
        return list(groups)

    is_acid = re.search(r"C\(=O\)O", s) is not None
    if is_acid:
        groups.add("carboxylic acid")

    is_ester = re.search(r"[Cc]O[Cc]\(=O\)", s) is not None
    if is_ester:
        groups.add("ester")

    is_ketone = not is_acid and not is_ester and re.search(r"[Cc]C\(=O\)[Cc]", s) is not None
    if is_ketone:
        groups.add("ketone")
        if re.search(r"(=O).*\(C\)=O|\(=O\).*=O|\(C\)=O", s) is not None:
            groups.add("diketone")

    is_aldehyde = not is_acid and (
        re.search(r"C=O$", s) is not None
        or re.search(r"^O=C(/|\[|[Cc])", s) is not None
        or re.search(r"\(C=O\)$", s) is not None
        or re.search(r"[cC]\(C=O\)", s) is not None
        or re.search(r"\)C=O", s) is not None
    )
    if is_aldehyde:
        groups.add("aldehyde")

    if has_aromatic and phenol:
        groups.add("phenol")

    if not has_aromatic and (re.search(r"[Cc]O$", s) is not None or re.search(r"\([CcH]\)O$", s) is not None) and not is_acid and not is_ester:
        groups.add("alcohol")

    # Ether: an O bonded to two carbons — COc (anisole/methoxy), CCOC, COC.
    # A phenol's O-H is single-carbon-bonded so it never matches these patterns;
    # the structural `phenol` check is what separates Ar-OH from Ar-O-CH3.
    if re.search(r"CO[cC]|CCO[cC]|O[Cc][CcH]|\)O[Cc]", s) is not None and not is_ester and not is_acid:
        groups.add("ether")

    if re.search(r"\[NH[0-9]\]|\(N\)|[Cc]N[Cc]|N[Cc]", s) is not None:
        groups.add("amine")

    is_thiol = re.search(r"\[SH\]|S[Cc]?H|H[Ss]", s) is not None or re.search(r"^[Cc][Ss]$", s) is not None
    if is_thiol:
        groups.add("thiol")
    if re.search(r"[Ss][Cc]|S[Ss]|\(S\)", s) is not None:
        groups.add("thioether")
    if is_thiol or re.search(r"[Ss][Cc]|S[Ss]|\(S\)", s) is not None:
        groups.add("sulfur")

    # Alkene: a carbon-carbon double bond. Kekulé ring bonds are already removed
    # by normalization, so this only fires on real alkenes.
    if re.search(r"[Cc]\d*=[Cc]\d*", s) is not None:
        groups.add("alkene")

    # Furan ring: a five-membered O heterocycle — either the aromatic `o1cccc1`
    # form or a Kekulé `C1=COC=C1` ring normalised to a digit-less `o`.
    if furan or re.search(r"o[1-9]", s) is not None:
        groups.add("furan")

    # Alkane: only C/H/ring/bond characters, no hetero atoms, no double bonds, no aromatic.
    if re.fullmatch(r"[CcH0-9/\\()\[\]%]+", s) and "=" not in s and not has_aromatic:
        groups.add("alkane")

    # Terpene is not reliably inferable from SMILES alone; leave it to keyword/odor matching.
    return list(groups)
