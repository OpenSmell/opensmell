"""Benchmark 4 — environment robustness: temperature and humidity.

Two sources of evidence:

 PHYSICS     the humidity/temperature modules quantify the expected baseline
             swing over a plausible RH (20-70%) and T (20-45 C) range.  This
             is the *raw-feature* corruption that environment metadata must
             absorb.

 REAL UCI    the turbulent recordings log T and RH per row.  We regress each
             channel's log-baseline on (T, RH) across 180 recordings in air:
             the R2 tells how much baseline variance is environmental, and the
             coefficients give an empirical d(log R0)/dRH and d(log R0)/dT.
"""

from __future__ import annotations

import numpy as np

from _common import REPO_ROOT, record_benchmark


def _physics_sweep() -> dict:
    import physics as P

    rh = np.linspace(20, 70, 11)
    base_RH = [P.humidity_baseline_multiplier(r) for r in rh]
    supp_RH = [P.humidity_response_suppression(r) for r in rh]
    t = np.linspace(20, 45, 11) + 273.15
    k = np.array([P.arrhenius_rate(T=tt, A0=1.0, Ea=0.5) for tt in t])
    base_T = k / k[0]
    return {
        "RH_20_70": {
            "baseline_multiplier_range": [round(float(base_RH[0]), 4),
                                          round(float(base_RH[-1]), 4)],
            "baseline_swing_pct": round(100 * abs(base_RH[-1] - base_RH[0]), 2),
            "sensitivity_suppression_20_70": [round(float(supp_RH[0]), 4),
                                              round(float(supp_RH[-1]), 4)],
            "sensitivity_loss_pct": round(100 * (1 - supp_RH[-1]), 2),
        },
        "T_20C_45C_Ea05eV": {
            "conduction_rate_normalized": [round(float(base_T[0]), 4),
                                           round(float(base_T[-1]), 4)],
            "rate_swing_pct": round(100 * abs(base_T[-1] - base_T[0]), 2),
        },
        "note": ("raw R0 swings tens of % from RH/T alone; primitives that use "
                 "intra-session R0 absorb this only if the environment is "
                 "stable within the window."),
    }


def _real_uci_turbulent() -> dict:
    import glob

    files = sorted(glob.glob(str(
        REPO_ROOT /
        "e-nose-evals/data/turbulent-mixtures/dataset_twosources_raw/*")))
    rec = []
    import pandas as pd
    for f in files:
        df = pd.read_csv(f, header=None, dtype=np.float64)
        v = df.values
        if v.shape[1] < 11:
            continue
        T = float(np.median(v[:, 1]))
        RH = float(np.median(v[:, 2]))
        air = v[:, 3:11][: int(60.0 / 0.02)]          # first 60 s clean air
        R = 10.0 * (3110.0 - air) / air
        r0 = np.nanmedian(R, axis=0)
        rec.append(np.r_[T, RH, r0])
    rec = np.array(rec, dtype=float)

    X = np.column_stack([np.ones(len(rec)), rec[:, 1], rec[:, 0]])  # [1, RH, T]
    out = {"n_recordings": int(len(rec)), "T_range": [float(rec[:, 0].min()),
                                                       float(rec[:, 0].max())],
           "RH_range": [float(rec[:, 1].min()), float(rec[:, 1].max())],
           "channels": {}}
    for ch in range(rec.shape[1] - 2):
        y = np.log10(np.clip(rec[:, 2 + ch], 1e-6, None))
        beta, res_ss, tot_ss = np.linalg.lstsq(X, y, rcond=None)[0], 0.0, 0.0
        resid = y - X @ beta
        r2 = 1.0 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
        out["channels"][f"ch{ch}"] = {
            "d_logR0_dRH": round(float(beta[1]), 5),
            "d_logR0_dT": round(float(beta[2]), 5),
            "R2": round(float(r2), 4),
        }
    return out


def run() -> dict:
    return {"physics_sweep": _physics_sweep(), "real_uci_turbulent": _real_uci_turbulent()}


def main() -> None:
    res = run()
    record_benchmark("bench_environment_robustness", res)
    print("Wrote reports/bench_environment_robustness.json")
    print("physics:", res["physics_sweep"])
    real = res["real_uci_turbulent"]
    print(f"real: {real['n_recordings']} recordings, "
          f"T {real['T_range']}, RH {real['RH_range']}")
    for ch, v in list(real["channels"].items())[:4]:
        print(f"  {ch}: dlogR0/dRH={v['d_logR0_dRH']} "
              f"dlogR0/dT={v['d_logR0_dT']} R2={v['R2']}")


if __name__ == "__main__":
    main()