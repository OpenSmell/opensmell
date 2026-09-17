"""Benchmark 0 — physics self-check.

Verifies the causal chain in physics/ returns physically sane numbers end to
end, so every downstream benchmark inherits a checked foundation:
adsorption -> depletion -> transport -> power law -> kinetics -> humidity.
"""

from __future__ import annotations

import numpy as np

from _common import REPORTS, record_benchmark


def run() -> dict:
    import physics as P

    rng = np.random.default_rng(0)
    out: dict[str, dict] = {}

    # 1. adsorption -------------------------------------------------------
    theta = P.langmuir_theta(np.array([0.1, 1.0, 10.0]), K=1.0)
    out["langmuir_theta"] = {
        "values": [round(float(v), 4) for v in theta],
        "sanity": all(0.0 < v < 1.0 for v in theta),
        "expected": "theta(C=K) == 0.5",
        "mid_check": abs(float(theta[1]) - 0.5) < 1e-9,
    }

    # 2. depletion --------------------------------------------------------
    dep = P.DebyeLengthParams(N_D=1e25, eps_r=14.0, T=600.0, site_density=1e18)
    Ld_m = P.debye_length(params=dep)
    Q = P.space_charge_density(theta_O=0.1, site_density=1e18)
    eps_abs = dep.eps  # eps_r * eps_0  (F/m) -- band_bending wants absolute eps
    psi = P.band_bending(Q_sc=Q, N_D=1e25, eps=eps_abs)
    W_m = P.depletion_width(Q_sc=Q, N_D=1e25)
    out["depletion"] = {
        "debye_length_nm": round(Ld_m * 1e9, 4),
        "space_charge_Cm2": round(float(Q), 6),
        "band_bending_V": round(float(psi), 4),
        "depletion_width_nm": round(W_m * 1e9, 4),
        "sanity": Ld_m > 0 and Q < 0 and psi > 0 and 0 < W_m,
        "expected": "Ld~2nm @ 600K, ns-region C/m^2, psi~0.6V, W~10nm",
    }

    # 3. power-law exponents ----------------------------------------------
    exp = {s: P.theoretical_exponent(s) for s in ("O2-", "O-", "O2^2-")}
    out["theoretical_exponents"] = {k: round(v, 4) for k, v in exp.items()}

    # 4. synthetic power-law fit and curvature ------------------------------
    C = np.array([1, 2, 5, 10, 20, 50, 100], dtype=float)
    true_logA, true_B = 0.3, -0.5
    R = np.exp(true_logA) * C ** true_B
    # physiologic distortion: flatten at low C (noise floor), saturate at high C
    R = R * (1.0 - 0.05 * np.exp(-C / 3.0)) + 1e-3
    fit = P.fit_power_law(C, R)
    out["power_law_fit"] = {
        "A": round(fit.A, 4), "B": round(fit.B, 4),
        "r2": round(fit.r_squared, 6),
        "curvature": round(fit.curvature, 5),
        "model_class": fit.model_class,
        "B_recovered": abs(fit.B - true_B) < 0.05,
        "curvature_small": abs(fit.curvature) < 0.05,
    }

    # 5. kinetics -----------------------------------------------------------
    kp = P.KineticParams(k_ads=0.1, k_des=0.02)
    tau_r = P.first_order_response_time(C=1.0, params=kp)
    tau_d = P.first_order_recovery_time(params=kp)
    out["kinetics"] = {
        "response_tau_s": round(tau_r, 4),
        "recovery_tau_s": round(tau_d, 4),
        "sanity": tau_r > 0 and tau_d > 0 and tau_d > tau_r,
    }

    # 6. temperature (arrhenius baseline) -------------------------------------
    k25 = P.arrhenius_rate(T=25.0 + 273.15, A0=1.0, Ea=0.5)
    k40 = P.arrhenius_rate(T=40.0 + 273.15, A0=1.0, Ea=0.5)
    k_change = k40 / k25
    out["temperature"] = {
        "rate_multiplier_25C_to_40C": round(k_change, 4),
        "sanity": k_change > 1.0,  # warmer -> faster conduction -> lower R at const G0
    }

    # 7. humidity -------------------------------------------------------------
    rh30 = P.humidity_baseline_multiplier(30.0)
    rh70 = P.humidity_baseline_multiplier(70.0)
    supp = P.humidity_response_suppression(70.0)
    out["humidity"] = {
        "baseline_RH30": round(rh30, 4),
        "baseline_RH70": round(rh70, 4),
        "response_suppression_RH70": round(supp, 4),
        "note": "model convention: R0(RH)/R0_dry=(1-k*RH)^a falls with RH (class D); sensitivity is always suppressed by humidity",
        "sanity": (0.0 < rh70 < 1.0) and (0.0 < supp < 1.0),
    }

    # 8. mixture additivity oracle -------------------------------------------------
    sA, sB = 0.8, 0.6
    ideal = P.ideal_additive_response(sA, sB)
    verdict_ideal = P.masking_metric(sA, sB, sA * sB)
    verdict_mask = P.masking_metric(sA, sB, sA * sB * 0.5)
    verdict_syn = P.masking_metric(sA, sB, min(1.0, sA * sB * 1.5))
    out["mixtures"] = {
        "ideal_AB": round(ideal, 4),
        "ideal_verdict": verdict_ideal["verdict"],
        "masking_verdict": verdict_mask["verdict"],
        "synergistic_verdict": verdict_syn["verdict"],
    }

    record_benchmark("bench_physics_selfcheck", out)
    return out


def main() -> None:
    res = run()
    print(f"Wrote {REPORTS/'bench_physics_selfcheck.json'}")
    print("checks:", {k: v.get("sanity", v) for k, v in res.items()})


if __name__ == "__main__":
    main()