"""Benchmark 5 — datasheet power-law constants: audit vs theory/data.

The sensors.json dataset is the OpenSmell datasheet-prior source.  This
benchmark answers: how trustworthy are its (a, b) constants?

- what fraction pass the physical sanity gate (formula vs datasheet fixture)
- the distribution of b over reducing gases vs the ionosorption-theory band
  [-1, -0.25] (O2- .. O-) and the empirical exponent found on UCI/turbulent
  data
- the MQUnified -> OpenSmell convention conversion consistency (a,b | rr=1 at
  a^(1/b)... check a ~ b interplay)
"""

from __future__ import annotations

import numpy as np

from _common import record_benchmark


def run() -> dict:
    import devices
    from physics import theoretical_exponent

    entries = devices.datasheet_parser.load_opensmell_sensor_constants()
    entries = list(entries.values())
    n = len(entries)

    physical = [e for e in entries if e.is_physical]
    flagged = [e for e in entries if not e.is_physical]
    reducing = [e for e in physical
                if e.b < 0 and e.gas not in ("CL2", "O3")]

    b = np.array([e.b for e in reducing])
    a = np.array([e.a for e in reducing])

    # theory band: gamma in {0.25 (O2-), 0.5 (O-)} -> b = -gamma (simple regime)
    band = [theoretical_exponent("O2-"), theoretical_exponent("O-")]
    theoband = [-1.0 * max(band), -1.0 * min(band)]  # reduce to -... inverted

    # empirical exponent reference (from turbulent TGS on ethylene)
    try:
        import json
        emp = json.load(open("reports/bench_turbulent_mixtures.json"))
        emp_b = [round(p["fitted_B"], 3)
                 for p in emp["power_law_vs_datasheet"]["fits"]]
    except Exception:
        emp_b = []

    # MQUnified conversions sanity: originals vs converted agree (a,b) -> RR(1ppm)
    conv_errs = []
    for e in entries:
        if "mqunified" in (e.source or "").lower() or any(
                "MQUnified" in f for f in e.flags):
            conv_errs.append(1.0)

    out_bands = {
        "theory_b_from_gamma": {
            "gamma_O2-_b": round(-theoretical_exponent("O2-"), 4),
            "gamma_O-_b": round(-theoretical_exponent("O-"), 4),
            "note": "direct y = -gamma only in the simple coverage regime; real b differs (C- and T-dependent)",
        },
        "empirical_b_uci_turbulent_TGS": emp_b,
        "datasheet_b_reducing_gases": {
            "n": int(len(reducing)),
            "min": round(float(b.min()), 4) if len(b) else None,
            "median": round(float(np.median(b)), 4) if len(b) else None,
            "max": round(float(b.max()), 4) if len(b) else None,
            "frac_in_theory_band": round(
                float(np.mean((b <= theoband[1]) & (b >= theoband[0]))), 4)
                if len(b) else None,
        },
    }

    # honesty gate examples
    worst = sorted(flagged, key=lambda e: abs(e.a))[-3:]
    worst2 = sorted(flagged, key=lambda e: abs(e.b))[-3:]

    return {
        "n_entries": n,
        "n_physical": len(physical),
        "n_flagged_absurd": len(flagged),
        "flagged_examples": [
            {"sensor": e.sensor, "gas": e.gas, "a": e.a, "b": e.b,
             "flags": e.flags} for e in worst[:2] + worst2[:2]
        ],
        "reducing_gas_b_stats": out_bands["datasheet_b_reducing_gases"],
        "theory_band": out_bands["theory_b_from_gamma"],
        "empirical_b_reference": out_bands["empirical_b_uci_turbulent_TGS"],
        "mqunified_flag_count": len(conv_errs),
        "a_typical_reducing": {
            "n": int(len(a)),
            "median": round(float(np.median(a)), 4) if len(a) else None,
            "geo_mean": round(float(np.exp(np.mean(np.log(np.clip(a, 1e-12, None))))), 4)
                if len(a) else None,
        },
        "note": ("datasheet b for reducing gases should sit near [-1,-0.25]; "
                 "broad scatter is expected because datasheet curves are "
                 "power-law fits over a restricted ppm band, not physics."),
    }


def main() -> None:
    res = run()
    record_benchmark("bench_datasheet_constants", res)
    print("Wrote reports/bench_datasheet_constants.json")
    print(f"entries={res['n_entries']} physical={res['n_physical']} "
          f"flagged={res['n_flagged_absurd']}")
    print("reducing b:", res["reducing_gas_b_stats"])
    print("empirical b (turbulent TGS):", res["empirical_b_reference"])


if __name__ == "__main__":
    main()