"""Substance search and exact resolution.

Mirrors `osmograph-web/lib/smellability/search.ts` 1:1. Scoring: exact 100,
prefix 75, contains 55, all-tokens 65; CAS and SMILES exact hits score 95/90.
User-dictionary entries join the surface marked provisional (score - 5).
"""

from __future__ import annotations

import re
from typing import List, Optional

from .composites import COMPOSITES
from .compounds import COMPOUNDS
from .constants import CLASS_TERMS
from .types import Chemical, Composite, ResolvedEntity, SearchCandidate
from .user_dictionary import read_user_dictionary


def normalize_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.lower()).strip()


def _norm(s: str) -> str:
    return s.lower().strip()


def _score_field(field: str, query: str) -> int:
    f = _norm(field)
    if f == query:
        return 100
    if f.startswith(query) and len(query) >= 2:
        return 75
    if query in f and len(query) >= 2:
        return 55
    tokens = query.split(" ")
    if len(tokens) > 1 and all(t in f for t in tokens):
        return 65
    return 0


def _score_chemical(c: Chemical, query: str) -> int:
    best = _score_field(c.name, query)
    for s in c.synonyms:
        best = max(best, _score_field(s, query))
    if c.cas and _norm(c.cas) == query:
        best = max(best, 95)
    if c.smiles and _norm(c.smiles) == query:
        best = max(best, 90)
    return best


def _score_composite(c: Composite, query: str) -> int:
    best = _score_field(c.name, query)
    for s in c.synonyms:
        best = max(best, _score_field(s, query))
    return best


def search_substances(query: str, limit: int = 8) -> List[SearchCandidate]:
    q = normalize_query(query)
    if len(q) < 1:
        return []

    results: List[SearchCandidate] = []

    for c in COMPOUNDS:
        score = _score_chemical(c, q)
        if score >= 40:
            results.append(
                SearchCandidate(
                    kind="chemical",
                    id=c.id,
                    name=c.name,
                    display_name=c.name,
                    match_hint=f"chemical · CAS {c.cas}" if c.cas else "chemical",
                    score=score,
                )
            )

    # User-resolved compounds join the search surface, clearly marked provisional.
    for c in read_user_dictionary():
        score = _score_chemical(c, q)
        if score >= 40:
            results.append(
                SearchCandidate(
                    kind="chemical",
                    id=c.id,
                    name=c.name,
                    display_name=c.name,
                    match_hint="my dictionary · estimated",
                    score=score - 5,
                )
            )

    for c in COMPOSITES:
        score = _score_composite(c, q)
        if score >= 40:
            results.append(
                SearchCandidate(
                    kind="composite",
                    id=c.id,
                    name=c.name,
                    display_name=c.name,
                    match_hint=f"{c.kind} · mixture profile",
                    score=score,
                )
            )

    for key, term in CLASS_TERMS.items():
        score = _score_field(term.label, q) or _score_field(key, q)
        if score >= 40:
            results.append(
                SearchCandidate(
                    kind="class",
                    id=f"class:{key}",
                    name=key,
                    display_name=term.label,
                    match_hint="functional class",
                    score=score,
                )
            )

    results.sort(key=lambda c: -c.score)
    return results[:limit]


def exact_resolve(query: str) -> Optional[ResolvedEntity]:
    q = normalize_query(query)
    if len(q) < 1:
        return None
    candidates = search_substances(query, 1)
    if not candidates:
        return None
    top = candidates[0]
    if top.score >= 100:
        return ResolvedEntity(
            kind=top.kind,
            id=top.id,
            name=top.name,
            display_name=top.display_name,
            match_hint=top.match_hint,
        )
    return None
