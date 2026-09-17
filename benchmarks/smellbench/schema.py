"""Portable schema for an OpenSmell claim declaration.

The ruler is only as useful as it is reusable: any lab can declare a claim by
filling the same fields, and the same validator and gate apply.  The canonical
form is a plain dict (JSON-serialisable) so a declaration can be exchanged
between rigs without importing this package.

A declaration records:

    key         stable short id (also the report stem: reports/bench_<key>.json)
    module      the module that computes the evidence
    series      the benchmark's own family label (Bench n / Wn / WS n / An)
    title       one-line human title
    claim_class primary A-F chain layer (see claims.py)
    claim       the falsifiable statement, in words
    regimen     {chemistry, environment, protocol, timing, device}
    headline    the quantitative evidence the report must carry

A benchmark may honestly bear several claim classes; the declaration records
its *primary* one so the ruler can be grouped without ambiguity.
"""

from .claims import CLAIM_CLASSES, CLAIM_TAXONOMY_VERSION

SCHEMA_VERSION = "0.1.0"
REGIMEN_VERSION = "0.1.0"

REGIMEN_FIELDS = ("chemistry", "environment", "protocol", "timing", "device")
CLAIM_KEYS = (
    "key", "module", "series", "title", "claim_class", "claim", "regimen",
    "headline",
)


def validate_claim(claim: dict) -> list[str]:
    """Return a list of schema violations for one claim declaration (empty = valid)."""
    issues: list[str] = []
    if not isinstance(claim, dict):
        return [f"claim is {type(claim).__name__}, expected dict"]
    for key in CLAIM_KEYS:
        if key not in claim:
            issues.append(f"missing field {key!r}")
    if claim.get("claim_class") not in CLAIM_CLASSES:
        issues.append(f"claim_class {claim.get('claim_class')!r} not in taxonomy")
    for field in REGIMEN_FIELDS:
        value = claim.get("regimen", {}).get(field) if isinstance(claim.get("regimen"), dict) else None
        if not value or not any(str(x).strip() for x in value):
            issues.append(f"empty regimen field {field!r}")
    return issues