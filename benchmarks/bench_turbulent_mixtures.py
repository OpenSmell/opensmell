"""Benchmark 2 — turbulent mixtures: observability, mixture-additivity, power law.

Uses the 180 UCI recordings (8 TGS sensors, Ethylene ± CO/Methane at 4 levels).

 OBSERVABILITY    can the 30 concentration configs be told apart from the
                  per-channel primitives, and which pairs are ambiguous?
 MIXTURE ADDITIV. test the ideal-independent-adsorption null (s_AB = s_A s_B)
                  against competitive Langmuir masking on real Mixtures.
 POWER LAW        fit rr = a C^b on single-gas configs and compare to the
                  sensors.json datasheet constants (a,b) for the same models.
"""

from __future__ import annotations

import numpy as np

from _common import record_benchmark


def _logratio(X: np.ndarray, r0_samples: int = 15) -> np.ndarray:
    from ontology.physical_primitives import logratio_fingerprint
    return logratio_fingerprint(X, r0_samples)


def run() -> dict:
    import datasets
    from ontology.observability import distinguishability_report
    from physics import fit_power_law, masking_metric

    b = datasets.load_turbulent_mixtures(use_downsampled=True)
    X = b["X"]          # (180, 8, n_time)
    meta = b["meta"]
    chem = b["chemical"]

    # ---- per-recording primitive vectors ---------------------------------
    lr = np.stack([_logratio(x) for x in X])           # (180, 8)
    amp = []
    from ontology.physical_primitives import normalized_amplitude
    for x in X:
        row = np.array([normalized_amplitude(x[i]) for i in range(x.shape[0])])
        amp.append(row)
    amp = np.stack(amp)

    # ---- 1. observability over the 30 configs ------------------------------
    meta["cfg"] = (meta["ethylene_ppm"].astype(int).astype(str) + "_" +
                   meta["gas2"] + "_" + meta["gas2_ppm"].astype(int).astype(str))
    configs = sorted(meta["cfg"].unique())
    clouds = {}
    for cfg in configs:
        idx = meta["cfg"].values == cfg
        clouds[cfg] = amp[idx]
    rep = distinguishability_report(clouds, threshold=1.0,
                                    representation="physical_primitives")
    obs_summary = rep.summary()

    # worst ambiguous pairs (the information-destruction boundary)
    amb = sorted([p for p in rep.pairs if p["decision"] == "ambiguous"],
                 key=lambda p: p["separation"])[:8]

    # ---- 2. mixture additivity ----------------------------------------------
    # use amplitude triplets: single-gas configs (only ethylene) vs both present.
    mix = []
    for gas2 in ("CO", "Me"):
        for et_lvl in ("L", "M", "H"):
            for g2_lvl in ("L", "M", "H"):
                mask = (meta["ethylene_level"] == et_lvl) & \
                       (meta["gas2"] == gas2) & (meta["gas2_level"] == g2_lvl)
                if not mask.any():
                    continue
                whole = amp[mask]                     # (n, 8)
                # ideal = product of the separate single-gas fingerprints
                sA = amp[(meta["ethylene_level"] == et_lvl) & (meta["gas2_level"] == "n")].mean(axis=0)
                sB = amp[(meta["gas2_level"] == g2_lvl) & (meta["ethylene_level"] == "n")].mean(axis=0)
                for j in range(whole.shape[1]):
                    sAB = float(whole[:, j].mean())
                    if sA[j] > 0 and sB[j] > 0 and sAB > 0:
                        v = masking_metric(sA[j], sB[j], sAB)
                        mix.append({"gas2": gas2, "et": et_lvl, "g2": g2_lvl,
                                    "ch": j, **v})
    from collections import Counter
    verdicts = Counter(m["verdict"] for m in mix)
    ideal_errs = [m["multiplicative_error"] for m in mix]

    # additive-dose null: log(1+s_AB) == log(1+s_A) + log(1+s_B)
    # i.e. s_AB == (1+s_A)(1+s_B) - 1  (amplitudes add in dose, not ratio)
    logadd_errs, logadd_viol, linadd_viol = [], [], []
    mix_raw = []
    for gas2 in ("CO", "Me"):
        for et_lvl in ("L", "M", "H"):
            for g2_lvl in ("L", "M", "H"):
                mask = (meta["ethylene_level"] == et_lvl) & \
                       (meta["gas2"] == gas2) & (meta["gas2_level"] == g2_lvl)
                if not mask.any():
                    continue
                whole = amp[mask]
                sA = amp[(meta["ethylene_level"] == et_lvl) &
                         (meta["gas2_level"] == "n")].mean(axis=0)
                sB = amp[(meta["gas2_level"] == g2_lvl) &
                         (meta["ethylene_level"] == "n")].mean(axis=0)
                for j in range(whole.shape[1]):
                    sAB = float(whole[:, j].mean())
                    if sA[j] > 0 and sB[j] > 0 and sAB > 0:
                        mix_raw.append((float(sA[j]), float(sB[j]), sAB))
    if mix_raw:
        sa, sb, sab = np.array(mix_raw).T
        additive_null = (1 + sa) * (1 + sb) - 1
        logadd_error = np.abs(np.log((1 + sab) / (additive_null + 1e-9)))
        logadd_viol = float(np.mean(sab > additive_null * 1.05))
        linadd = float(np.mean(sab > (sa + sb) * 1.05))
    else:
        logadd_error, logadd_viol, linadd = np.array([]), float("nan"), float("nan")

    mixture = {
        "n_triplets": len(mix),
        "verdict_counts": dict(verdicts),
        "log_multiplicative_error_mean": round(float(np.mean(ideal_errs)), 4),
        "competitive_suppression_mean": round(
            float(np.mean([m["competitive_suppression"] for m in mix])), 4),
        "synergy_ratio_mean": round(
            float(np.mean([m["enhancement_ratio"] for m in mix])), 4),
        "vs_additive_dose_null": {
            "log_error_mean": round(float(np.mean(logadd_error)), 4)
                if len(logadd_error) else "n/a",
            "frac_sAB_gt_additive": round(logadd_viol, 4)
                if logadd_viol == logadd_viol else "n/a",
            "frac_sAB_gt_linear_sum": round(linadd, 4)
                if linadd == linadd else "n/a",
        },
        "note": (
            "verdict 'synergistic' means s_AB > s_A*s_B (multiplicative null "
            "fails). Compare vs additive-dose null to see if response amplitudes "
            "simply add (both reduces gases depleting the same barrier)."),
    }

    # ---- 3. power law vs datasheet -----------------------------------------
    # ethylene-only configs: levels 0,31,46,96 ppm (n / L / M / H, gas2 none)
    noct = meta[meta["gas2_level"] == "n"]
    pl = []
    consts = None
    try:
        import devices
        consts = devices.datasheet_parser.load_opensmell_sensor_constants()
    except Exception:
        pass
    tgs_models = sorted({m.split("a")[0] for m in b["channel_names"]})
    if consts:
        available = set(consts.keys())
        have = [m for m in tgs_models if any(m in (s,) for (s, _) in available)]
    else:
        have = []
    for ch, model in enumerate(b["channel_names"]):
        rows = []
        for _, r in noct.iterrows():
            c = float(r["ethylene_ppm"])
            idx = np.where(meta["cfg"].values ==
                           f"{int(c)}_{r['gas2']}_{int(r['gas2_ppm'])}")[0]
            if len(idx) == 0:
                continue
            rr = float(np.nanmean(lr[idx, ch]))   # average over the 6 repeats
            rows.append((c, rr))
        rows = [r for r in rows if r[1] == r[1] and r[1] != 0]
        if len(rows) < 3:
            continue
        C = np.array([r[0] for r in rows]); R = np.array([r[1] for r in rows])
        fit = fit_power_law(C, R)
        dat = None
        gas_aliases = ("C2H4", "ETHYLENE", "Ethylene")
        key = (model, "C2H4")
        dat = consts.get(key) if consts else None
        if dat is None and consts:
            for g in gas_aliases:
                dat = consts.get((model, g))
                if dat is not None:
                    break
        pl.append({
            "channel": model,
            "fitted_A": round(fit.A, 4), "fitted_B": round(fit.B, 4),
            "r2": round(fit.r_squared, 4), "n_points": fit.n_points,
            "datasheet_a": round(dat.a, 4) if dat else None,
            "datasheet_b": round(dat.b, 4) if dat else None,
            "expected_B": "reducing gas -> negative, ~-0.5 (O- dominated)",
        })

    return {
        "observability_configs": {
            "summary": obs_summary,
            "worst_ambiguous_pairs": [
                {"A": p["A"], "B": p["B"], "separation": round(p["separation"], 3)}
                for p in amb],
        },
        "mixture_additivity": mixture,
        "power_law_vs_datasheet": {
            "note": (
                "sensors.json holds MQ constants only; the turbulent array is "
                "Figaro TGS, so the datasheet comparison is not available here. "
                "Fitted B on ethylene-only recordings is reported instead."),
            "datasheet_entries_for_tgs": len(have),
            "fits": pl,
        },
    }


def main() -> None:
    res = run()
    record_benchmark("bench_turbulent_mixtures", res)
    print("Wrote reports/bench_turbulent_mixtures.json")
    o = res["observability_configs"]["summary"]
    print(f"cfg ambiguity: {o['ambiguous_fraction']:.3f} "
          f"(mean sep {o['mean_separation']:.2f}, min {o['min_separation']:.2f})")
    print("mixture verdicts:", res["mixture_additivity"]["verdict_counts"])
    for p in res["power_law_vs_datasheet"]["fits"][:4]:
        print(f"  {p['channel']}: B={p['fitted_B']} r2={p['r2']} "
              f"datasheet({p['datasheet_a']},{p['datasheet_b']})")


if __name__ == "__main__":
    main()