"""Command line for the OpenSmell benchmark ruler.

Verbs:

  check      validate every claim declaration against the portable schema
  verify     cross-check declarations against the reports on disk (the gate)
  manifest   write the shareable ruler-card JSON
  annotate   upgrade existing reports to the self-describing envelope
  bench      run the reference benchmarks (subprocess, per-benchmark timeout)

Exit code is 0 when the verb's gate passes, non-zero otherwise.
"""

from __future__ import annotations

import argparse

from .annotate import annotate_reports
from .manifest import write_manifest
from .regen import DEFAULT_TIMEOUT_S, run_benchmarks
from .regimens import REGISTRY
from .schema import SCHEMA_VERSION, validate_claim
from .verify import verify_reports


def _cmd_check(_args: argparse.Namespace) -> int:
    problems: list[str] = []
    seen: set[str] = set()
    for entry in REGISTRY:
        key = entry.get("key", "<missing>")
        if key in seen:
            problems.append(f"{key}: duplicate registry key")
        seen.add(key)
        for problem in validate_claim(entry):
            problems.append(f"{key}: {problem}")
    if problems:
        print(f"[ISSUES] {len(problems)} schema problem(s)")
        for problem in problems:
            print("  !", problem)
        return 1
    print(f"[OK] {len(REGISTRY)} claim declarations valid (schema {SCHEMA_VERSION})")
    return 0


def _cmd_verify(_args: argparse.Namespace) -> int:
    return 0 if verify_reports(verbose=True)["ok"] else 1


def _cmd_manifest(args: argparse.Namespace) -> int:
    out = write_manifest(path=args.output)
    print(f"Wrote {out}")
    return 0


def _cmd_annotate(_args: argparse.Namespace) -> int:
    annotate_reports(verbose=True)
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    result = run_benchmarks(keys=args.keys or None, timeout_s=args.timeout,
                            skip_existing=args.skip_existing, verbose=True)
    return 0 if all(s.get("ok") for s in result.values()) else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="opensmell-bench", description=__doc__)
    sub = parser.add_subparsers(dest="verb", required=True)
    sub.add_parser("check", help="validate claim declarations against the schema")
    sub.add_parser("verify", help="cross-check declarations against reports on disk")
    p_manifest = sub.add_parser("manifest", help="write the shareable ruler card")
    p_manifest.add_argument("output", nargs="?", default=None)
    sub.add_parser("annotate", help="upgrade existing reports to the envelope")
    p_bench = sub.add_parser("bench", help="run reference benchmarks")
    p_bench.add_argument("keys", nargs="*", help="benchmark keys (default: all)")
    p_bench.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S,
                         help=f"per-benchmark wall-clock timeout in seconds "
                              f"(default {DEFAULT_TIMEOUT_S})")
    p_bench.add_argument("--skip-existing", action="store_true",
                         help="trust reports already on disk (cheap cache path)")

    args = parser.parse_args(argv)
    handler = {
        "check": _cmd_check,
        "verify": _cmd_verify,
        "manifest": _cmd_manifest,
        "annotate": _cmd_annotate,
        "bench": _cmd_bench,
    }[args.verb]
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())