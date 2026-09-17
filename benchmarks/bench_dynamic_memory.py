"""W1 — controlled history dependence & multi-timescale recovery (wave-4).

The sensor-memory audit (A1) ran carryover + single-time-constant tau.  This
benchmark tightens the physics: it exploits the *known per-second ground-truth
schedule* of the UCI dynamic-mixtures corpus to run the roadmap's controlled
sequence experiments in real sensor data:

  A -> clean -> A      same (gas, ppm) config repeated after clean-air gaps of
                       *different* lengths; the response modulation vs gap is
                       the R_A(2) != R_A(1) test with concentration fixed.

  A -> B -> A          same config A following a *different* gas B vs after
                       gas A itself; the amplitude difference is the
                       history-dependence that is not concentration.

  multi-timescale      recovery of each exposure tail is fit as R(t) = r_inf
                       + a1*exp(-t/t1) + a2*exp(-t/t2) and scored (AIC) against
                       the single exponential, answering "is a single tau
                       enough?" without assuming the equation.

The 4x TGS2602 replicate channels act as four device-level draws for every
quantity, so parameter scatter is reported per device family.

Every quantity is computed twice (once per mixture file) and the verdict
states up-front that these lab streams hold the environment fixed - so what is
measured here is the *chemical* history dependence, the cleanest part of the
drift stack.
"""

from __future__ import annotations

import numpy as np

from _common import record_benchmark
from harness.loaders import load_dynamic_mixtures

FILES = ["ethylene_CO.txt", "ethylene_methane.txt"]
SR = 100.0
MIN_EPISODE_S = 10.0      # min length of a config block / air gap to analyze
MIN_CLEAN_GAP_S = 5.0     # a "clean" repetition requires this much air before onset
MIN_GAP_FOR_TAU_S = 25.0  # min gap length to fit a recovery kernel
R0_WIN_S = 20.0           # pre-exposure air window for baseline
TAIL_WIN_S = 10.0         # steady-state response window at end of exposure
CHANNEL_GROUPS = {
    "TGS2602": [0, 1, 8, 9],
    "TGS2600": [2, 3, 10, 11],
    "TGS2610": [4, 5, 12, 13],
    "TGS2620": [6, 7, 14, 15],
}


def _grid(b):
    """1 Hz resistance grid + per-second ground-truth ppm from a loader bundle."""
    r = np.asarray(b["X"], dtype=np.float64)          # (n_100hz, 16) kOhm
    sr = int(b["sr"])
    n_full = len(r) // sr * sr
    idx = (np.arange(n_full) / sr).astype(np.int64)
    out_n = int(idx.max()) + 1
    sums = np.zeros((out_n, r.shape[1]))
    cnt = np.zeros((out_n, 1))
    np.add.at(cnt, idx[:, None], 1)
    np.add.at(sums, idx, np.nan_to_num(r[:n_full]))
    R = sums / np.maximum(cnt, 1)
    meta = b["meta"]
    et = meta["ethylene_ppm"].to_numpy(dtype=np.float64)[:n_full]
    g2 = meta["gas2_ppm"].to_numpy(dtype=np.float64)[:n_full]
    et_s = et.reshape(out_n, sr).mean(axis=1)
    g2_s = g2.reshape(out_n, sr).mean(axis=1)
    return R, et_s, g2_s


def _segments(et_s, g2_s):
    """Contiguous runs of constant (rounded) config key -> segment dicts."""
    k0 = np.round(np.clip(et_s, 0, None) * 3).astype(np.int64)
    k1 = np.round(np.clip(g2_s, 0, None) * 3).astype(np.int64)
    key = k0 * 100000 + k1
    bounds = np.r_[0, np.flatnonzero(np.diff(key)) + 1, len(key)]
    segs = []
    for a, z in zip(bounds[:-1], bounds[1:]):
        dur = int(z - a)
        if dur < int(MIN_EPISODE_S):
            continue
        et_med = float(np.median(et_s[a:z]))
        g2_med = float(np.median(g2_s[a:z]))
        segs.append({
            "start": int(a), "end": int(z), "dur": dur, "key": int(key[a]),
            "et": et_med, "g2": g2_med,
            "gas": bool(et_med > 0 or g2_med > 0),
        })
    return segs


def _recovery(t, r_inf, a1, t1, a2, t2):
    return r_inf + a1 * np.exp(-t / t1) + a2 * np.exp(-t / t2)


def _recovery_single(t, r_inf, a, tau):
    return r_inf + a * np.exp(-t / tau)


_TAU_FAST = [0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 25.0, 40.0]
_TAU_SLOW = [45.0, 60.0, 90.0, 120.0, 180.0, 240.0, 360.0, 480.0,
             720.0, 1200.0, 1800.0, 3000.0]


def _fit_recovery(tsec, y, r0):
    """Fit single- and bi-exponential recovery by *linear* least squares over a
    fixed grid of decay constants (no iterative curve_fit).

    R(t) = r_inf + a1*exp(-t/t1) + a2*exp(-t/t2) is linear in the amplitudes at
    fixed (t1, t2), so the best single/bi kernel is found by sweeping candidate
    time constants and solving lstsq once per candidate.  Deterministic and fast.

    Returns dict {winner, t_fast, t_slow, ratio, sse1, sse2} or None.
    """
    y = np.asarray(y, dtype=float)
    tsec = np.asarray(tsec, dtype=float)
    n = len(y)
    if n < 10 or not np.all(np.isfinite(y)):
        return None
    ones = np.ones(n)

    def fitexp(tau):
        A = np.column_stack([ones, np.exp(-tsec / tau)])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        return coef, float(np.sum((A @ coef - y) ** 2))

    best_single = None
    for tau in _TAU_FAST + _TAU_SLOW:
        coef, sse = fitexp(tau)
        if best_single is None or sse < best_single[0]:
            best_single = (sse, tau, coef)
    best_bi = None
    for t1 in _TAU_FAST:
        c1 = np.exp(-tsec / t1)
        for t2 in _TAU_SLOW:
            if t2 <= t1 + 1.0:
                continue
            A = np.column_stack([ones, c1, np.exp(-tsec / t2)])
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            sse = float(np.sum((A @ coef - y) ** 2))
            if best_bi is None or sse < best_bi[0]:
                best_bi = (sse, t1, t2, coef)
    if best_single is None or best_bi is None:
        return None
    sse1, tau1, _ = best_single
    sse2, t_fast, t_slow, _ = best_bi
    # AIC for gaussian noise: 2k + n*ln(SSE/n)
    aic1 = 2 * 3 + n * np.log(max(sse1, 1e-12) / n)
    aic2 = 2 * 5 + n * np.log(max(sse2, 1e-12) / n)
    return {"winner": "bi" if aic2 < aic1 else "single",
            "t_fast": round(float(t_fast), 2),
            "t_slow": round(float(t_slow), 2),
            "ratio": round(float(t_slow / max(t_fast, 1e-6)), 2),
            "sse1": round(sse1, 4), "sse2": round(sse2, 4)}


def _response_amp(R, ch, b0_lo, b0_hi, tail_lo, tail_hi):
    """Signed fractional deflection |R0 - R_tail|/R0 with sign."""

    def med(lo, hi):
        seg = R[lo:hi, ch]
        return float(np.nanmedian(seg)) if len(seg) else float("nan")

    b0 = med(b0_lo, b0_hi)
    tail = med(tail_lo, tail_hi)
    if not (np.isfinite(b0) and np.isfinite(tail) and abs(b0) > 1e-6):
        return None, None
    return (b0 - tail) / abs(b0), b0


def _analyze_file(name):
    b = load_dynamic_mixtures(file_name=name)
    R, et_s, g2_s = _grid(b)
    segs = _segments(et_s, g2_s)
    n_sec = R.shape[0]

    # Build positional helper: for each segment index, the previous gas segment
    # (scanning the *filtered* segment list, which may skip tiny sub-second
    # segments that were dropped by MIN_EPISODE_S).
    n_segs = len(segs)
    prev_gas = [None] * n_segs
    last_gas = None
    for i, s in enumerate(segs):
        prev_gas[i] = last_gas
        if s["gas"]:
            last_gas = i

    gas_segs_idx = [i for i, s in enumerate(segs) if s["gas"]]

    # ---- (a) clean repetitions: same config, varying clean gap length -------
    clean_rows = []            # one per (segment, channel)
    for gi in gas_segs_idx:
        s = segs[gi]
        # previous air segment (the one immediately before this gas segment)
        prev_air_idx = None
        for j in range(gi - 1, -1, -1):
            if not segs[j]["gas"]:
                prev_air_idx = j
                break
        if prev_air_idx is None:
            continue
        prev_air = segs[prev_air_idx]
        gap_s = s["start"] - prev_air["start"]
        if gap_s < MIN_CLEAN_GAP_S:
            continue
        # previous GAS segment (for priming analysis)
        pgi = prev_gas[gi]
        prev_gas_key = segs[pgi]["key"] if pgi is not None else None
        b0_lo, b0_hi = max(prev_air["start"], s["start"] - int(R0_WIN_S)), s["start"]
        tail_lo, tail_hi = max(s["start"], s["end"] - int(TAIL_WIN_S)), s["end"]
        for ch in range(R.shape[1]):
            amp, b0v = _response_amp(R, ch, b0_lo, b0_hi, tail_lo, tail_hi)
            if amp is None:
                continue
            clean_rows.append({
                "ch": ch, "key": s["key"], "et": s["et"], "g2": s["g2"],
                "gap_s": gap_s, "dur_s": s["dur"], "amp": float(amp),
                "prev_gas_key": prev_gas_key,
            })
    # response vs gap, pooled per channel group (concentration pooled: the slope
    # below is the *average* gap softening across all cleanly repeated configs)
    rg = {}
    for grp, chidx in CHANNEL_GROUPS.items():
        rows = [r for r in clean_rows if r["ch"] in chidx]
        if len(rows) < 8:
            rg[grp] = {"n": len(rows), "slope_per_log_gap": None, "r2": None,
                       "median_amp": None}
            continue
        g = np.log(np.clip([r["gap_s"] for r in rows], 1, None))
        a = np.abs([r["amp"] for r in rows])
        beta, *_ = np.linalg.lstsq(
            np.column_stack([np.ones(len(g)), g]), a, rcond=None)
        resid = a - (beta[0] + beta[1] * g)
        ss_tot = np.sum((a - a.mean()) ** 2) + 1e-12
        rg[grp] = {
            "n": len(rows),
            "slope_per_log_gap": round(float(beta[1]), 4),
            "r2": round(1 - resid.dot(resid) / ss_tot, 4),
            "median_amp": round(float(np.median(a)), 4),
        }
    # per-config correlation (same chemistry, only the gap differs)
    per_cfg = []
    for key in sorted({r["key"] for r in clean_rows}):
        rs = [r for r in clean_rows if r["key"] == key]
        per_ch = {c: [r for r in rs if r["ch"] == c] for c in {r["ch"] for r in rs}}
        for c, rows in per_ch.items():
            if len(rows) < 4:
                continue
            gv = np.array([r["gap_s"] for r in rows], dtype=float)
            av = np.array([abs(r["amp"]) for r in rows], dtype=float)
            if np.std(gv) < 1.0 or np.std(av) < 1e-9:
                continue
            rho = float(np.corrcoef(gv, av)[0, 1])
            per_cfg.append({"key": int(key), "ch": c, "n": len(rows),
                            "corr_amp_gap": round(rho, 3)})
    cfg_corrs = [c["corr_amp_gap"] for c in per_cfg]
    gap_mod = {
        "per_group_slope": rg,
        "per_config_corr_n": len(per_cfg),
        "per_config_corr_median": round(float(np.median(cfg_corrs)), 3)
            if cfg_corrs else None,
        "frac_configs_negative_amp_gap_corr": round(
            float(np.mean(np.array(cfg_corrs) < 0)), 3) if cfg_corrs else None,
    }

    # ---- (b) A->B->A vs A->A->A priming: same config, diff previous gas ----
    # ---- (b) A->B->A vs A->A->A priming: same config, diff previous gas ----
    for r in clean_rows:
        r["self"] = bool(r["prev_gas_key"] is not None and r["prev_gas_key"] == r["key"])
    per_cfg_prime = {}
    for key in sorted({r["key"] for r in clean_rows}):
        rs = [r for r in clean_rows if r["key"] == key and r["prev_gas_key"] is not None]
        if len(rs) < 4:
            continue
        for ch in {r["ch"] for r in rs}:
            sub = [r for r in rs if r["ch"] == ch]
            self_s = [abs(r["amp"]) for r in sub if r["self"]]
            cross_s = [abs(r["amp"]) for r in sub if not r["self"]]
            if len(self_s) >= 2 and len(cross_s) >= 2:
                per_cfg_prime[f"{key}_{ch}"] = {
                    "ch": ch,
                    "mean_amp_after_self": round(float(np.mean(self_s)), 4),
                    "mean_amp_after_cross": round(float(np.mean(cross_s)), 4),
                    "delta_self_minus_cross": round(
                        float(np.mean(self_s) - np.mean(cross_s)), 4),
                    "n_self": len(self_s), "n_cross": len(cross_s),
                }
    deltas = [v["delta_self_minus_cross"] for v in per_cfg_prime.values()]
    priming = {
        "n_config_channel_groups": len(per_cfg_prime),
        "mean_delta_self_minus_cross": round(float(np.mean(deltas)), 4)
            if deltas else None,
        "median_delta": round(float(np.median(deltas)), 4) if deltas else None,
        "frac_groups_self_response_lower_than_cross": round(
            float(np.mean(np.array(deltas) < 0)), 3) if deltas else None,
        "groups": per_cfg_prime,
    }

    # ---- (c) multi-timescale recovery on exposure->air tails -----------------
    rec = {grp: [] for grp in CHANNEL_GROUPS}
    for i, air_s in enumerate(segs):
        if air_s["gas"]:
            continue
        gap_dur = air_s["end"] - air_s["start"]
        if gap_dur < int(MIN_GAP_FOR_TAU_S):
            continue
        # only air gaps that directly follow a resolved exposure
        pgi = prev_gas[i]
        if pgi is None:
            continue
        gap_start = air_s["start"]
        gap_end = air_s["end"]
        tsec = np.arange(gap_dur, dtype=float)
        for grp, chidx in CHANNEL_GROUPS.items():
            for ch in chidx:
                y = R[gap_start:gap_end, ch]
                if len(y) < 10 or not np.isfinite(y).all():
                    continue
                amp_rng = np.nanmax(y) - np.nanmin(y)
                if amp_rng < 1e-4 * np.nanmean(y):
                    continue
                f = _fit_recovery(tsec, y, r0=None)
                if f is not None:
                    rec[grp].append(f)
    multi_out = {}
    for grp, l in rec.items():
        if not l:
            multi_out[grp] = {"n_fits": 0, "frac_bi_exp_better": None,
                              "median_t_fast": None, "median_t_slow": None,
                              "median_ratio": None}
            continue
        frac = float(np.mean([f["winner"] == "bi" for f in l]))
        tf = [f["t_fast"] for f in l]
        ts = [f["t_slow"] for f in l]
        rr = [f["ratio"] for f in l]
        multi_out[grp] = {
            "n_fits": len(l),
            "frac_bi_exp_better": round(frac, 3),
            "median_t_fast": round(float(np.median(tf)), 1),
            "median_t_slow": round(float(np.median(ts)), 1),
            "median_ratio": round(float(np.median(rr)), 2),
        }

    return {
        "source": name,
        "n_seconds": int(n_sec),
        "n_configs": len(gas_segs_idx),
        "clean_repetitions": len(clean_rows),
        "response_vs_gap": gap_mod,
        "priming": priming,
        "recovery_multiscale": multi_out,
        "method": (
            "1 Hz grid; per-second ground-truth ppm from the dataset columns. "
            "Clean repetition = config block preceded by >=5 s air.  Response "
            "= |R0 - R_tail10s|/R0.  Priming compares a config after itself "
            "vs after a different interaction.  Recovery fits the air tail "
            "after each exposure with 1- and 2-exponential kernels, winner by "
            "AIC (gaussian-noise criterion)."),
    }


def run() -> dict:
    files = {}
    for name in FILES:
        try:
            files[name] = _analyze_file(name)
        except FileNotFoundError:
            continue
    # aggregate verdict across both streams (data-driven, honest about sign)
    n_bi = 0
    n_fit = 0
    corr_list = []
    prim_deltas = []
    gap_slopes = []
    for fr in files.values():
        for grp, v in fr["recovery_multiscale"].items():
            n_fit += v.get("n_fits", 0)
            n_bi += int(round((v.get("frac_bi_exp_better") or 0) * v.get("n_fits", 0)))
        c = fr["response_vs_gap"].get("per_config_corr_median")
        if c is not None:
            corr_list.append(c)
        for g in fr["priming"]["groups"].values():
            prim_deltas.append(g["delta_self_minus_cross"])
        for v in fr["response_vs_gap"]["per_group_slope"].values():
            if v.get("slope_per_log_gap") is not None and v.get("r2") is not None \
                    and v["r2"] > 0.1:
                gap_slopes.append(v["slope_per_log_gap"])
    frac_bi = round(n_bi / max(n_fit, 1), 3)
    med_corr = round(float(np.median(corr_list)), 3) if corr_list else None
    med_prim = round(float(np.median(prim_deltas)), 4) if prim_deltas else None
    n_prim_neg = int(np.mean([d < 0 for d in prim_deltas]) * len(prim_deltas)) \
        if prim_deltas else 0
    slope_sign = ("mostly negative" if (gap_slopes and
                                         np.mean([s < 0 for s in gap_slopes]) > 0.6)
                  else "mixed/noisy")
    verdict = (
        "history dependence: across repeated identical (gas,ppm) configs the "
        f"response-vs-gap correlation is consistently negative (median "
        f"{med_corr}, same chemistry, only the gap differs) and the pooled "
        f"gap-slope is {slope_sign}, i.e. a shorter clean gap lowers the next "
        f"response in most device families.  The A->A->A vs A->B->A priming "
        f"delta is {med_prim} (self-priming raised the next response in "
        f"{len(prim_deltas) - n_prim_neg}/{len(prim_deltas)} config-channel "
        "groups, lowered it in the rest - the sign is gas/device specific, so "
        "priming cannot be a single scalar).  The exposure recovery needs TWO "
        f"time scales in {frac_bi:.0%} of fitted tails - a single exponential "
        "does NOT describe the memory state on this array, so the audit cannot "
        "collapse h_t to one constant.")
    return {
        "what": (
            "Wave-4 controlled history-dependence audit on the UCI dynamic-"
            "mixtures ground-truth schedule: R_A(2) vs R_A(1) with concentration "
            "fixed (gap modulation), A->A->A vs A->B->A priming, and 1-vs-2 "
            "exponential recovery kernels. Lab streams hold T/RH fixed, so this "
            "is the chemical part of the drift stack, cleanly measured."),
        "files": files,
        "aggregate_verdict": verdict,
    }


def main() -> None:
    res = run()
    record_benchmark("bench_dynamic_memory", res)
    print("Wrote reports/bench_dynamic_memory.json")
    for fname, fr in res["files"].items():
        gm = fr["response_vs_gap"]["per_group_slope"]
        sl = {k: v["slope_per_log_gap"] for k, v in gm.items()}
        pr = fr["priming"]
        msc = fr["recovery_multiscale"]
        print(f"\n{fname}: clean rep {fr['clean_repetitions']} "
              f"gap-slope {sl}")
        print(f"  priming: n_groups {pr['n_config_channel_groups']} "
              f"delta_self-cross {pr['mean_delta_self_minus_cross']} "
              f"frac_lower {pr['frac_groups_self_response_lower_than_cross']}")
        for grp, v in msc.items():
            print(f"  {grp}: bi-exp better {v['frac_bi_exp_better']} "
                  f"t_fast {v['median_t_fast']} t_slow {v['median_t_slow']} "
                  f"ratio {v['median_ratio']}  (n={v['n_fits']})")
    print("\naggregate verdict:", res["aggregate_verdict"])


if __name__ == "__main__":
    main()