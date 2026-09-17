"""W3 — sensor identifiability audit: dose and gas identity vs noise + memory.

Public-question (digital olfaction): how long must a sensor window be before
the array can tell two gas conditions apart, given (i) the measured replicate
scatter of THIS array and (ii) the exposure memory residue of the previous
condition (W1)?

Model.  Each 8-channel recording is reduced to a dose response vector
    v_i[c] = log R0[c] - log R_tail[c]    (resistance drops on reducing gas)
with R0 the clean-air baseline (first ~12 s) and R_tail the steady tail.  The
per-channel scatter is decomposed into a *level* component (recording-to-
recording, from the repeated identical configs) and a *sample* component
(within-window temporal noise).  A k-second window mean carries variance

    var_c(k) = sigma_level_c^2 + sigma_sample_c^2 / (k * fps)

so the standardized distance between two conditions becomes

    d_k = sqrt( sum_c (v_a[c] - v_b[c])^2 / var_c(k) )

and the 2-class Bayes accuracy is  Phi(d_k / 2)  (closed form, no sampler).
Because the level noise does not shrink with the window, every pair has an
accuracy ceiling at k -> inf:  d_inf = sqrt( sum_c d_c^2 / sigma_level_c^2 ).

Memory.  The residue of a prior condition r is modelled as a fraction
    alpha(t) = 0.11 * (0.6 e^{-t/15} + 0.4 e^{-t/55})
of the prior condition's response vector, calibrated to the W1 recovery tau
medians (t_fast 15 s, t_slow 55 s) and the A1 median carryover (11%).  The
adversarially labelled observation is v_a + alpha*v_r; both the *worst-case*
residue (min over all 30 configs, mixtures included) and the *typical*
residue (median) are reported, since the worst residue is genuinely the
same-gas full dose just measured (a real field scenario).

Sweeps: window k in {0.5,1,2,5,10,30,60,120,300} s x residue gap t in
{1,30,120,600} s.  Outcome per pair: minimal window to reach acc >= 0.95, the
k -> inf accuracy ceiling, or "impossible" if the ceiling itself is below
0.95.  Separations are also stated in the observability metric (Mahalanobis-2)
for cross-reference with the framework's 1.0 threshold.
"""

from __future__ import annotations

import math

import numpy as np

from _common import record_benchmark
from harness.loaders import load_turbulent_mixtures

FPS = 10
BASELINE_S = 12.0          # clean-air reference at the start of each recording
TAIL_FRAC = 0.55           # steady-exposure tail used for the response
WIN_K = [0.5, 1, 2, 5, 10, 30, 60, 120, 300]
ACC_BAR = 0.95             # distinguishable threshold
SEP_THRESHOLD = 1.0        # observability threshold (reused)
MEM_TAUS = (15.0, 55.0)    # W1 median recovery taus (fast, slow)
MEM_AMPS = (0.6, 0.4)      # fast/slow amplitude split at t = 0
MEM_CARRY = 0.11           # A1 median carryover fraction at t = 0


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _acc_from_d(d: float) -> float:
    return _phi(d / 2.0)


def _alpha(t: float) -> float:
    return MEM_CARRY * sum(
        a * math.exp(-t / tau) for a, tau in zip(MEM_AMPS, MEM_TAUS))


def _response_vectors(b):
    """Per-recording 8-ch dose-response vectors (v = log R0/R_tail)."""
    X, m, sr = b["X"], b["meta"], b["sr"]
    n = X.shape[2]
    b0 = int(BASELINE_S * sr)
    tail = slice(-int(n * TAIL_FRAC), None)
    out = np.empty((X.shape[0], X.shape[1]))
    for i in range(X.shape[0]):
        lr = np.log(X[i])
        r0 = np.median(lr[:, :b0], axis=1)
        rt = np.median(lr[:, tail], axis=1)
        out[i] = r0 - rt
    keys = list(zip(m["ethylene_level"], m["gas2"], m["gas2_level"]))
    return out, keys


def _noise_model(v, keys, b):
    n_ch = v.shape[1]
    X = b["X"]
    n = X.shape[2]
    tail = slice(-int(n * TAIL_FRAC), None)
    samp = np.empty(0)
    for i in range(X.shape[0]):
        lr = np.log(X[i])[:, tail]
        d = lr - np.median(lr, axis=1, keepdims=True)
        samp = np.concatenate([samp, np.std(d, axis=1)])
    sigma_samp = np.sqrt(np.mean(samp ** 2)) * np.ones(n_ch)
    uni = {}
    for k, row in zip(keys, v):
        uni.setdefault(k, []).append(row)
    level = [np.std(np.stack(rows), axis=0)
             for rows in uni.values() if len(rows) >= 2]
    sigma_level = np.sqrt(np.mean(np.stack(level) ** 2, axis=0))
    return sigma_level, sigma_samp


def _config_means(v, keys):
    uni = {}
    for k, row in zip(keys, v):
        uni.setdefault(k, []).append(row)
    return {k: np.mean(np.stack(rows), axis=0) for k, rows in uni.items()}


class _Metric:
    def __init__(self, sigma_level, sigma_samp):
        self.sl = sigma_level
        self.ss = sigma_samp

    def dist(self, ma, mb, k):
        var = self.sl ** 2 + self.ss ** 2 / (k * FPS)
        return float(np.sqrt(np.sum((ma - mb) ** 2 / var)))

    def dist_inf(self, ma, mb):
        return float(np.sqrt(np.sum((ma - mb) ** 2 / self.sl ** 2)))

    def acc(self, ma, mb, k):
        return _acc_from_d(self.dist(ma, mb, k))

    def min_k(self, ma, mb):
        for k in WIN_K:
            if self.acc(ma, mb, k) >= ACC_BAR:
                return k
        return None


def run() -> dict:
    b = load_turbulent_mixtures(use_downsampled=True)
    v, keys = _response_vectors(b)
    sigma_level, sigma_samp = _noise_model(v, keys, b)
    means = _config_means(v, keys)
    universe = list(means.keys())
    M = _Metric(sigma_level, sigma_samp)

    # pure-gas configs grouped by class and dose
    def dose_label(cfg):
        et, g2, g2l = cfg
        if et == "n":
            return g2, g2l
        return "ethylene", et

    classes = {}
    for cfg in universe:
        gas, dose = dose_label(cfg)
        classes.setdefault(gas, {})[dose] = cfg
    class_doses = {g: sorted(doses) for g, doses in classes.items()}

    # -------- pair helpers --------
    def mem_acc(k, ma, mb, t, stat):
        all_acc = []
        for r in universe:
            res = _alpha(t) * means[r]
            all_acc.append(min(M.acc(ma + res, mb, k),
                               M.acc(ma, mb + res, k)))
        return float(np.min(all_acc)) if stat == "min" \
            else float(np.median(all_acc))

    def min_k_mem(ma, mb, t, stat):
        for k in WIN_K:
            if mem_acc(k, ma, mb, t, stat) >= ACC_BAR:
                return k
        return None

    def pair_report(ga, da, gb, db):
        ca, cb = classes[ga][da], classes[gb][db]
        ma, mb = means[ca], means[cb]
        d_inf = M.dist_inf(ma, mb)
        k_clean = M.min_k(ma, mb)
        k_worst = min_k_mem(ma, mb, 1.0, "min")
        k_median = min_k_mem(ma, mb, 1.0, "median")
        k_worst_600 = min_k_mem(ma, mb, 600.0, "min")
        return {
            "pair": f"{ga}-{da} vs {gb}-{db}",
            "no_memory_min_k_s": k_clean,
            "mem_worst_gap1s_min_k_s": k_worst,
            "mem_median_gap1s_min_k_s": k_median,
            "mem_worst_gap600s_min_k_s": k_worst_600,
            "acc_ceiling_k_inf": round(_acc_from_d(d_inf), 4),
            "separation_k1": round(M.dist(ma, mb, 1.0), 3),
            "observability_at_k1": ("distinguishable"
                                    if M.dist(ma, mb, 1.0) >= SEP_THRESHOLD
                                    else "ambiguous"),
        }

    # -------- 1) dose resolution (adjacent doses within a pure gas) --------
    dose_rows = []
    for gas in ["CO", "Me", "ethylene"]:
        for da, db in zip(class_doses[gas][:-1], class_doses[gas][1:]):
            dose_rows.append(pair_report(gas, da, gas, db))

    # -------- 2) gas identity at MATCHED dose --------
    matched = []
    for dose in ["L", "M", "H"]:
        for ia in range(3):
            for ib in range(ia + 1, 3):
                ga = ["CO", "Me", "ethylene"][ia]
                gb = ["CO", "Me", "ethylene"][ib]
                matched.append(pair_report(ga, dose, gb, dose))

    # -------- 3) gas identity across ALL dose pairs (hardest) --------
    all_pairs = {}
    gaslist = ["CO", "Me", "ethylene"]
    for ia in range(3):
        for ib in range(3):
            ga, gb = gaslist[ia], gaslist[ib]
            if ga == gb or ia > ib:
                continue
            reports = []
            for da in class_doses[ga]:
                for db in class_doses[gb]:
                    reports.append(pair_report(ga, da, gb, db))
            # hardest = lowest k->inf ceiling (best-guess identifiability)
            hardest = min(reports, key=lambda r: r["acc_ceiling_k_inf"])
            all_pairs[f"{ga} vs {gb}"] = {
                "n_dose_combos": len(reports),
                "hardest_pair": hardest["pair"],
                "hardest_ceiling": hardest["acc_ceiling_k_inf"],
                "hardest_no_memory_min_k_s": hardest["no_memory_min_k_s"],
                "worst_clean_min_k_over_doses": max(
                    r["no_memory_min_k_s"] for r in reports
                    if r["no_memory_min_k_s"] is not None),
            }

    # -------- aggregates --------
    clean_k = [r["no_memory_min_k_s"] for r in dose_rows + matched
               if r["no_memory_min_k_s"] is not None]
    worst_k = [r["mem_worst_gap1s_min_k_s"] for r in dose_rows + matched
               if r["mem_worst_gap1s_min_k_s"] is not None]
    worst_k_600 = [r["mem_worst_gap600s_min_k_s"] for r in dose_rows + matched
                   if r["mem_worst_gap600s_min_k_s"] is not None]
    median_k = [r["mem_median_gap1s_min_k_s"] for r in dose_rows + matched
                if r["mem_median_gap1s_min_k_s"] is not None]
    imp_clean = [r["pair"] for r in dose_rows + matched
                 if r["no_memory_min_k_s"] is None]
    imp_worst = [r["pair"] for r in dose_rows + matched
                 if r["mem_worst_gap1s_min_k_s"] is None]
    imp_600 = [r["pair"] for r in dose_rows + matched
               if r["mem_worst_gap600s_min_k_s"] is None]

    aggregate = {
        "noise_calibration": {
            "sigma_level_channels": [round(x, 5) for x in sigma_level],
            "sigma_sample_channels": [round(x, 5) for x in sigma_samp],
        },
        "memory_model": {
            "calibration": "W1 taus (15/55 s) + A1 median carryover 11%",
            "alpha_gap1s": round(_alpha(1.0), 4),
            "alpha_gap600s": round(_alpha(600.0), 6),
        },
        "dose_resolution": dose_rows,
        "gas_identity_matched_dose": matched,
        "gas_identity_overview": all_pairs,
        "global_no_memory_max_min_k_s": max(clean_k) if clean_k else None,
        "global_mem_worst_gap1s_max_min_k_s": max(worst_k) if worst_k else None,
        "global_mem_worst_gap600s_max_min_k_s": (max(worst_k_600)
                                                 if worst_k_600 else None),
        "global_mem_median_gap1s_max_min_k_s": (max(median_k)
                                                if median_k else None),
        "n_pairs_impossible_clean": len(imp_clean),
        "pairs_impossible_clean": imp_clean,
        "n_pairs_impossible_mem_worst_gap1s": len(imp_worst),
        "pairs_impossible_mem_worst_gap1s": imp_worst,
        "n_pairs_impossible_mem_worst_gap600s": len(imp_600),
        "pairs_impossible_mem_worst_gap600s": imp_600,
    }

    cr = [r["acc_ceiling_k_inf"] for r in dose_rows + matched]
    if max(cr) < ACC_BAR:
        verdict = ("verdict: no dose or identity step is resolvable to acc "
                   "0.95 on this array even with infinite windowing — the "
                   "recording-to-recording level scatter sets a hard "
                   "identifiability ceiling.")
    elif aggregate["global_mem_worst_gap1s_max_min_k_s"] is None:
        verdict = ("verdict: clean windows resolve every resolvable pair "
                   "(max min-k %s s), but under a worst-case same-gas memory "
                   "residue (A1 11%% carryover) no pair reaches acc 0.95 "
                   "within 300 s; the memory of the sensor competes with the "
                   "current dose.  Typical (median) residue is far weaker "
                   "(max min-k %s s)." % (
                       aggregate["global_no_memory_max_min_k_s"],
                       aggregate["global_mem_median_gap1s_max_min_k_s"]))
    else:
        verdict = ("verdict: gas resolution needs %s s clean, %s s under a "
                   "worst-case 1-s residue gap, %s s when that gap is 600 s; "
                   "memory shifts identifiability, it does not remove it "
                   "(median residue needs only %s s)." % (
                       aggregate["global_no_memory_max_min_k_s"],
                       aggregate["global_mem_worst_gap1s_max_min_k_s"],
                       aggregate["global_mem_worst_gap600s_max_min_k_s"],
                       aggregate["global_mem_median_gap1s_max_min_k_s"]))
    aggregate["verdict"] = verdict

    return {
        "what": (
            "Wave-4 identifiability audit: how long a window a single 8-MOX "
            "recording needs to tell two gas conditions apart (dose within a "
            "gas, and gas identity at matched dose), given the measured "
            "replicate scatter and the W1-calibrated memory residue of the "
            "previous exposure.  Closed-form Gaussian accuracy Phi(d_k/2), "
            "k->inf accuracy ceilings, Mahalanobis-2 separations "
            "cross-referenced to the observability threshold."),
        "aggregate": aggregate,
    }


def main() -> None:
    res = run()
    record_benchmark("bench_identifiability", res)
    print("Wrote reports/bench_identifiability.json")
    a = res["aggregate"]
    print("\nnoise: level", a["noise_calibration"]["sigma_level_channels"][:4],
          "sample", a["noise_calibration"]["sigma_sample_channels"][:4])
    print("memory alpha(1s)=", a["memory_model"]["alpha_gap1s"],
          "alpha(600s)=", a["memory_model"]["alpha_gap600s"])
    print("\nDOSE resolution (adjacent):")
    for r in a["dose_resolution"]:
        print(f"  {r['pair']}: clean_k={r['no_memory_min_k_s']} "
              f"worst_k={r['mem_worst_gap1s_min_k_s']} "
              f"ceiling={r['acc_ceiling_k_inf']} sep_k1={r['separation_k1']}")
    print("\nGAS identity at matched dose:")
    for r in a["gas_identity_matched_dose"]:
        print(f"  {r['pair']}: clean_k={r['no_memory_min_k_s']} "
              f"worst_k={r['mem_worst_gap1s_min_k_s']} "
              f"median_k={r['mem_median_gap1s_min_k_s']} "
              f"ceiling={r['acc_ceiling_k_inf']}")
    print("\nGAS identity overview (per class pair, hardest dose combo):")
    for k, vv in a["gas_identity_overview"].items():
        print(f"  {k}: hardest {vv['hardest_pair']} "
              f"ceiling={vv['hardest_ceiling']} "
              f"clean_k={vv['hardest_no_memory_min_k_s']} "
              f"worst_clean_k_over_doses={vv['worst_clean_min_k_over_doses']}")
    print("\nGLOBAL clean max-min-k:", a["global_no_memory_max_min_k_s"],
          "| worst-residue gap1s max-min-k:",
          a["global_mem_worst_gap1s_max_min_k_s"],
          "| worst-residue gap600s max-min-k:",
          a["global_mem_worst_gap600s_max_min_k_s"],
          "| median-residue max-min-k:",
          a["global_mem_median_gap1s_max_min_k_s"])
    print("impossible clean:", a["n_pairs_impossible_clean"],
          a["pairs_impossible_clean"])
    print("impossible worst(gap1s):", a["n_pairs_impossible_mem_worst_gap1s"],
          a["pairs_impossible_mem_worst_gap1s"])
    print("impossible worst(gap600s):",
          a["n_pairs_impossible_mem_worst_gap600s"])
    print("\nverdict:", a["verdict"])


if __name__ == "__main__":
    main()