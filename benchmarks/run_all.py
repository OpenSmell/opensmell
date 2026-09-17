"""Run all benchmarks and aggregate the metrics for reports/.

Thin wrapper over smellbench.regen: each benchmark runs in its own
subprocess with a wall-clock timeout, and metrics_summary.json is written
incrementally so partial progress survives an interruption.
"""

from __future__ import annotations

from smellbench.regen import run_benchmarks
from smellbench.regimens import REGISTRY

BENCHMARKS = [(entry["key"], entry["module"]) for entry in REGISTRY]


def main() -> None:
    run_benchmarks()


if __name__ == "__main__":
    main()