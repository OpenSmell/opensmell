"""Build and write the ruler card: a shareable manifest of every claim.

The manifest is the artifact the wider field consumes: it carries the schema
and taxonomy versions, the class definitions, and every claim declaration with
its regimen, so a third party can read (or re-run) the ruler without importing
the package.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .claims import CLAIM_CLASSES, CLAIM_TAXONOMY_VERSION
from .paths import reports_dir
from .regimens import REGIMEN_VERSION, REGISTRY
from .schema import CLAIM_KEYS, REGIMEN_FIELDS, SCHEMA_VERSION

MANIFEST_NAME = "smellbench_manifest.json"


def build_manifest() -> dict:
    return {
        "name": "opensmell-bench",
        "framing": "one ruler for digital olfaction",
        "schema_version": SCHEMA_VERSION,
        "taxonomy_version": CLAIM_TAXONOMY_VERSION,
        "regimen_version": REGIMEN_VERSION,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fields": {
            "claim_keys": list(CLAIM_KEYS),
            "regimen_fields": list(REGIMEN_FIELDS),
        },
        "claim_classes": CLAIM_CLASSES,
        "claims": REGISTRY,
    }


def write_manifest(path=None, reports_path=None) -> Path:
    target = Path(path) if path is not None else reports_dir(reports_path) / MANIFEST_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build_manifest(), indent=2, sort_keys=False, default=str))
    return target


def main() -> None:
    out = write_manifest()
    print(f"Wrote {out}")
    print(f"{len(REGISTRY)} claims, schema {SCHEMA_VERSION}, taxonomy "
          f"{CLAIM_TAXONOMY_VERSION}, regimen {REGIMEN_VERSION}")


if __name__ == "__main__":
    main()