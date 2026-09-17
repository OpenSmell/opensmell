"""verify verb: cross-check the regimen registry against the reports on disk.

Checks:

  1. registry keys == metrics_summary benchmark keys (no orphan either way)
  2. every entry passes the portable schema (validate_claim)
  3. every reports/bench_<key>.json exists, parses, and carries a matching
     self-describing envelope (__claim__.key / claim_class, __meta__)
  4. every summary entry carries ok=True (data quality is a PASS/FLAG gate,
     never a score)
"""

from __future__ import annotations

import json

from .paths import reports_dir
from .regimens import REGIMEN_VERSION, REGISTRY
from .schema import validate_claim


def verify_reports(reports_path=None, verbose: bool = True) -> dict:
    reports = reports_dir(reports_path)
    issues: list[str] = []
    summary_path = reports / "metrics_summary.json"

    summary = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text()).get("benchmarks", {})
    else:
        issues.append(f"missing {summary_path}")

    reg_keys = {r["key"] for r in REGISTRY}
    sum_keys = set(summary)
    orphans_in_registry = sorted(reg_keys - sum_keys)
    orphans_in_summary = sorted(sum_keys - reg_keys)
    if orphans_in_registry:
        issues.append(f"registry keys with no summary entry: {orphans_in_registry}")
    if orphans_in_summary:
        issues.append(f"summary entries with no registry entry: {orphans_in_summary}")

    missing_reports: list[str] = []
    bad_ok: list[str] = []
    regimen_issues: list[str] = []
    missing_envelope: list[str] = []
    envelope_issues: list[str] = []

    for entry in REGISTRY:
        key = entry["key"]
        for problem in validate_claim(entry):
            regimen_issues.append(f"{key}: {problem}")

        path = reports / f"bench_{key}.json"
        if not path.is_file():
            missing_reports.append(f"bench_{key}.json")
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            missing_reports.append(f"bench_{key}.json (bad JSON: {exc})")
            continue

        claim = data.get("__claim__") if isinstance(data, dict) else None
        if not claim:
            missing_envelope.append(f"bench_{key}.json")
        else:
            if claim.get("key") != key:
                envelope_issues.append(f"{key}: __claim__.key={claim.get('key')!r}")
            if claim.get("claim_class") != entry["claim_class"]:
                envelope_issues.append(
                    f"{key}: __claim__.claim_class={claim.get('claim_class')!r} "
                    f"!= registry {entry['claim_class']!r}")
        if not (isinstance(data, dict) and data.get("__meta__")):
            envelope_issues.append(f"{key}: missing __meta__")

        if key in summary and not summary[key].get("ok"):
            bad_ok.append(key)

    if regimen_issues:
        issues.append(f"{len(regimen_issues)} regimen schema issue(s)")
    if missing_reports:
        issues.append(f"{len(missing_reports)} unreadable report(s)")
    if missing_envelope:
        issues.append(f"{len(missing_envelope)} report(s) without an envelope "
                      f"(run `annotate`)")
    if envelope_issues:
        issues.append(f"{len(envelope_issues)} envelope mismatch(es)")
    if bad_ok:
        issues.append(f"summary entries not ok: {bad_ok}")

    class_counts: dict[str, int] = {}
    for entry in REGISTRY:
        cls = entry.get("claim_class", "?")
        class_counts[cls] = class_counts.get(cls, 0) + 1

    result = {
        "ok": not issues,
        "regimen_version": REGIMEN_VERSION,
        "reports_dir": str(reports),
        "n_registry": len(REGISTRY),
        "n_summary": len(summary),
        "claim_class_counts": class_counts,
        "orphans_in_registry": orphans_in_registry,
        "orphans_in_summary": orphans_in_summary,
        "missing_reports": missing_reports,
        "missing_envelope": missing_envelope,
        "envelope_issues": envelope_issues,
        "summary_not_ok": bad_ok,
        "regimen_issues": regimen_issues,
        "issues": issues,
    }

    if verbose:
        status = "OK" if result["ok"] else "ISSUES"
        print(f"[{status}] regimen {REGIMEN_VERSION}: "
              f"{result['n_registry']} registry / {result['n_summary']} summary "
              f"entries")
        print("claim classes:", class_counts)
        for key in sorted(reg_keys):
            entry = next(r for r in REGISTRY if r["key"] == key)
            print(f"  {entry['series']:<9} {key:<26} class {entry['claim_class']}")
        for issue in issues:
            print("  !", issue)
    return result


def main() -> None:
    raised = verify_reports()
    raise SystemExit(0 if raised["ok"] else 1)


if __name__ == "__main__":
    main()