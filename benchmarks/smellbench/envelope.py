"""Report envelope: a score never travels without its claim and regimen.

Every report written through record_benchmark is annotated with the claim
declaration it provides evidence for (`__claim__`) and the package versions
(`__meta__`).  The injection is additive and non-breaking: all computed
evidence keys are preserved.
"""

from __future__ import annotations

from .claims import CLAIM_TAXONOMY_VERSION
from .regimens import REGIMEN_VERSION, REGISTRY
from .schema import SCHEMA_VERSION

CLAIM_KEYS = (
    "key", "module", "series", "title", "claim_class", "claim", "regimen",
    "headline",
)


def claim_for(name: str) -> dict | None:
    """Resolve the claim declaration for a report name (with/without 'bench_')."""
    key = name[6:] if name.startswith("bench_") else name
    return next((r for r in REGISTRY if r["key"] == key), None)


def meta() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "taxonomy_version": CLAIM_TAXONOMY_VERSION,
        "regimen_version": REGIMEN_VERSION,
    }


def wrap(name: str, results: dict) -> dict:
    """Return `results` with `__claim__` (if any) and `__meta__` added."""
    payload = dict(results)
    claim = claim_for(name)
    if claim is not None:
        payload["__claim__"] = {k: claim[k] for k in CLAIM_KEYS if k in claim}
    payload["__meta__"] = meta()
    return payload