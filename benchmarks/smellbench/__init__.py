"""smellbench - the OpenSmell benchmark ruler.

"One ruler for digital olfaction."  A pluggable harness that qualifies every
score with its regimen (chemistry x environment x protocol x timing x
device-generation) and verifies claims, not data or algorithms.

CLI verbs: check, verify, manifest, bench.
"""

from .claims import CLAIM_CLASSES, CLAIM_TAXONOMY_VERSION
from .regimens import REGIMEN_VERSION, REGISTRY

__all__ = [
    "CLAIM_CLASSES", "CLAIM_TAXONOMY_VERSION", "REGIMEN_VERSION", "REGISTRY",
    "SCHEMA_VERSION", "validate_claim", "verify_reports", "build_manifest",
    "write_manifest", "annotate_reports",
]
__version__ = "0.1.0"

_LAZY = {
    "SCHEMA_VERSION": (".schema", "SCHEMA_VERSION"),
    "validate_claim": (".schema", "validate_claim"),
    "verify_reports": (".verify", "verify_reports"),
    "build_manifest": (".manifest", "build_manifest"),
    "write_manifest": (".manifest", "write_manifest"),
    "annotate_reports": (".annotate", "annotate_reports"),
}


def __getattr__(name):
    if name in _LAZY:
        module, attr = _LAZY[name]
        import importlib
        return getattr(importlib.import_module(module, __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")