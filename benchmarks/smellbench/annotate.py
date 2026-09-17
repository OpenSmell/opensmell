"""Upgrade existing reports to the self-describing envelope, in place.

New runs are wrapped automatically by record_benchmark.  This verb annotates
reports that were produced before the envelope existed, without re-running the
benchmarks (the envelope only adds declaration metadata; computed evidence is
untouched).
"""

from __future__ import annotations

import json
from pathlib import Path

from .envelope import claim_for, wrap
from .paths import reports_dir
from .regimens import REGISTRY


def annotate_reports(reports_path=None, verbose: bool = True) -> list[str]:
    reports = reports_dir(reports_path)
    keys = {r["key"] for r in REGISTRY}
    annotated: list[str] = []
    for key in sorted(keys):
        path = reports / f"bench_{key}.json"
        if not path.is_file():
            if verbose:
                print(f"  - {path.name}: missing, skipped")
            continue
        data = json.loads(path.read_text())
        if isinstance(data, dict) and data.get("__claim__", {}).get("key") == key:
            if verbose:
                print(f"  = {path.name}: already annotated")
            continue
        path.write_text(json.dumps(wrap(f"bench_{key}", data), indent=2,
                                   sort_keys=True, default=str))
        annotated.append(key)
        if verbose:
            print(f"  + {path.name}: annotated ({claim_for(key)['series']}, "
                  f"class {claim_for(key)['claim_class']})")
    if verbose:
        print(f"annotated {len(annotated)} report(s)")
    return annotated


def main() -> None:
    annotate_reports()


if __name__ == "__main__":
    main()