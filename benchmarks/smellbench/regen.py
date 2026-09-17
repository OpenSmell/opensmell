"""Robust benchmark regeneration.

Each benchmark runs in its own subprocess with a wall-clock timeout, so one
slow or hanging leg can never take down the whole run.  The summary is written
incrementally after every benchmark, so partial progress survives an
interruption.  Real per-benchmark timings are recorded in the summary.

Selection: pass keys to run a subset; pass skip_existing to trust reports that
are already on disk (the cheap cache path).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

from .envelope import wrap
from .paths import benchmarks_dir, reports_dir
from .regimens import REGISTRY

DEFAULT_TIMEOUT_S = 600

_RUNNER = (
    "import importlib, sys\n"
    "from _common import record_benchmark\n"
    "key, module = sys.argv[1], sys.argv[2]\n"
    "res = importlib.import_module(module).run()\n"
    "record_benchmark('bench_' + key, res)\n"
)


def _summary_payload(reports) -> dict:
    path = reports / "metrics_summary.json"
    if path.is_file():
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            data = {}
        if isinstance(data, dict) and "benchmarks" in data:
            return data["benchmarks"]
    return {}


def _write_summary(reports, benchmarks: dict) -> None:
    payload = wrap("metrics_summary", {
        "__generated__": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmarks": benchmarks,
    })
    (reports / "metrics_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str))


def _run_one(key: str, module: str, bench_dir, timeout_s: int) -> dict:
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _RUNNER, key, module],
            cwd=str(bench_dir), capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"ok": False, "seconds": round(time.time() - t0, 2),
                "error": f"timeout after {timeout_s}s"}
    dur = round(time.time() - t0, 2)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        return {"ok": False, "seconds": dur, "error": detail[-2000:]}
    return {"ok": True, "seconds": dur}


def run_benchmarks(keys=None, timeout_s: int = DEFAULT_TIMEOUT_S,
                   reports_path=None, skip_existing: bool = False,
                   verbose: bool = True) -> dict:
    bench_dir = benchmarks_dir()
    reports = reports_dir(reports_path)
    reports.mkdir(parents=True, exist_ok=True)

    by_key = {r["key"]: r for r in REGISTRY}
    selected = list(keys) if keys else list(by_key)
    unknown = sorted(set(selected) - set(by_key))
    if unknown:
        raise SystemExit(f"unknown benchmark keys: {unknown}")

    benchmarks = _summary_payload(reports)
    for key in selected:
        module = by_key[key]["module"]
        report_path = reports / f"bench_{key}.json"
        if skip_existing and report_path.is_file():
            status = benchmarks.get(key) or {"ok": True}
            status = {"ok": status.get("ok", True), "seconds": "cached"}
            benchmarks[key] = status
            _write_summary(reports, benchmarks)
            if verbose:
                print(f"[cached] {key:<24} (report present)")
            continue
        status = _run_one(key, module, bench_dir, timeout_s)
        benchmarks[key] = status
        _write_summary(reports, benchmarks)
        if verbose:
            tag = "ok" if status["ok"] else "FAILED"
            print(f"[{tag:<6}] {key:<24} {status['seconds']:>7}s"
                  + ("" if status["ok"] else f"  {status.get('error', '')[:80]}"))

    if verbose:
        selected_ok = sum(1 for k in selected if benchmarks.get(k, {}).get("ok"))
        total_ok = sum(1 for s in benchmarks.values() if s.get("ok"))
        print(f"\n{selected_ok}/{len(selected)} selected ok; "
              f"{total_ok}/{len(benchmarks)} total ok; wrote "
              f"{reports / 'metrics_summary.json'}")
    return benchmarks


def main(argv=None) -> None:
    args = sys.argv[1:] if argv is None else argv
    run_benchmarks(keys=args or None)


if __name__ == "__main__":
    main()