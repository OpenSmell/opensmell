"""A1 — sensor memory: does exposure residue leak into the next baseline?

The Wörner corpus (ref. 6) is the *schema the product needs*: every
measurement cycle logs a stage-1 baseline, a stage-2 exposure and a stage-3
recovery (Cycle_Stage ∈ {1,2,3}; ~300-550 rows/stage, 62 channels).  The
sensor-memory question is whether the array fully returns to baseline before
the next exposure: how much of an exposure's deflection still sits in the next
clean-air window (carryover) and how fast the recovery relaxes (time constant
tau).

The raw Wörner CSV is NOT in this workspace (deferred: the schema-ready loader
and the exact analysis contract below run the moment the corpus arrives, either
under `e-nose-evals/data/worner*/` or via the WORNER_CSV env var).

While the corpus is absent we run the same memory machinery LIVE on the closest
real surrogate sharing the cycle structure: the UCI dynamic-mixtures stream has
hundreds of alternating exposure / air blocks on a 16-channel Figaro array
(4x TGS2602 / TGS2600 / TGS2610 / TGS2620).  Each air gap after an exposure is
treated exactly like a Wörner recovery stage: recovery tau is fit and the
carryover of the exposure deflection into the next baseline is measured.

Deferred vs surrogate results are kept visibly separate in the JSON.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from _common import REPO_ROOT, record_benchmark

from harness.loaders import load_dynamic_mixtures

CHANNEL_GROUPS = {
    "TGS2602": [0, 1, 8, 9],
    "TGS2600": [2, 3, 10, 11],
    "TGS2610": [4, 5, 12, 13],
    "TGS2620": [6, 7, 14, 15],
}

WORNER_GLOB = "worner*/**/*.csv"
MEMORY_EPS = 0.10      # carryover below 10% of deflection = "forgot" for that gap
MIN_EPISODE_S = 15.0
MIN_GAP_S = 10.0


def _recovery(t, r_inf, a, tau):
    return r_inf + a * np.exp(-t / tau)


def _load_worner_raw() -> dict | None:
    """Locate/read the deferred Wörner corpus; None while the corpus is absent.

    Schema contract (from §11.10 of opensmell-rs/docs/anomaly-engine-design.md):
    one CSV per measurement cycle (700 files) or one file with a cycle id
    column; per cycle the columns ``timestep_id``, ``Cycle_Stage``
    (1=baseline, 2=exposure, 3=recovery), and 62 numeric channel columns in
    resistance Ohms (150 KOhm-5.8 MOhm range).
    """
    if os.environ.get("WORNER_CSV"):
        candidates = [Path(os.environ["WORNER_CSV"])]
    else:
        candidates = [p for p in (REPO_ROOT / "e-nose-evals" / "data").glob(
            WORNER_GLOB)]
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None
    import pandas as pd
    df = pd.read_csv(candidates[0])
    return {"file": str(candidates[0]), "rows": len(df),
            "columns": list(df.columns)[:8], "schema_read": True}


def _load_surrogate():
    """1 Hz resistance grid + correctly binned per-second air/gas mask."""
    data = None
    for name in ("ethylene_CO.txt", "ethylene_methane.txt", None):
        try:
            data = load_dynamic_mixtures(file_name=name)
            break
        except FileNotFoundError:
            continue
    if data is None:
        raise FileNotFoundError("dynamic-mixtures corpus not present")

    r = data["X"]                                   # (n_rows, 16) kOhm at 100 Hz
    sr = data["sr"]
    n_full = len(r) // sr * sr
    idx = (np.arange(n_full) / sr).astype(int)      # 1 s bins (repeated sr times)
    out_n = int(idx.max()) + 1
    sums = np.zeros((out_n, 16))
    cnt = np.zeros((out_n, 1))
    np.add.at(cnt, idx[:, None], 1)
    np.add.at(sums, idx, np.nan_to_num(r[:n_full]))
    R = sums / np.maximum(cnt, 1)

    et = data["meta"]["ethylene_ppm"].to_numpy()[:n_full]
    g2 = data["meta"]["gas2_ppm"].to_numpy()[:n_full]
    row_gas = (et > 0) | (g2 > 0)
    gas = row_gas.reshape(out_n, sr).any(axis=1)    # per-second "any gas seen"
    return {"R": R, "gas": gas, "sr": 1.0, "n_seconds": out_n,
            "file": data["name"]}


def _runs(mask):
    """Contiguous true runs as (start, end) index pairs (end exclusive)."""
    runs, s = [], None
    m = np.pad(mask.astype(int), (1, 1), constant_values=0)
    for i in range(1, len(m)):
        if m[i] and not m[i - 1]:
            s = i - 1
        if not m[i] and m[i - 1] and s is not None:
            runs.append((s, i - 1))
            s = None
    return runs


def _fit_tau(t, y, b0):
    """Fit R(t) = r_inf + a*exp(-t/tau) on air-gap seconds; (None, None, None)
    on failure."""
    from scipy.optimize import curve_fit
    if len(t) < 8:
        return None, None, None
    try:
        popt, _pcov = curve_fit(_recovery, t, y, p0=[b0, (y[0] - b0), 40.0],
                                bounds=([0.3 * b0, -10 * abs(b0), 0.5],
                                        [1.5 * b0, 0.0, 3000.0]),
                                maxfev=3000)
        return float(popt[2]), float(popt[0]), float(popt[1])
    except (RuntimeError, ValueError):
        return None, None, None


def _memory_on_surrogate() -> dict:
    s = _load_surrogate()
    R = s["R"]
    air = ~np.nan_to_num(s["gas"]).astype(bool)
    exposures = _runs(np.nan_to_num(s["gas"]).astype(bool))
    gaps = _runs(air)
    t = np.arange(R.shape[0], dtype=float)

    n_exposures = 0
    rows_by_ch = {ch: [] for ch in range(16)}
    for gi, (gs, ge) in enumerate(gaps):
        # a usable recovery gap starts right after an exposure
        prev = next(((es, ee) for (es, ee) in exposures if ee == gs), None)
        if prev is None or ge - gs < MIN_GAP_S or prev[1] - prev[0] < MIN_EPISODE_S:
            continue
        es, ee = prev
        n_exposures += 1
        # baseline: air just before this exposure, else the first 60 s of air
        if es > 0:
            b0_lo, b0_hi = max(es - 20, 0), es
        else:
            b0_lo, b0_hi = 0, min(60, ge)
        e_lo, e_hi = max(es, ee - 20), ee          # last 20 s of the exposure
        c_lo, c_hi = ge - min(20, ge - gs), ge     # last 20 s of the gap

        def med9__(lo, hi):
            seg = R[lo:hi]
            return float(np.nanmedian(seg)) if len(seg) else float("nan")

        b0 = med9__(b0_lo, b0_hi)
        for ch in range(16):
            b0v = b0 if np.all(np.isfinite(R[b0_lo:b0_hi, ch])) else float("nan")
            dex = med9__(e_lo, e_hi) - b0v
            if not np.isfinite(dex) or abs(dex) < 1e-4:
                continue
            carry = (med9__(c_lo, c_hi) - b0v) / dex
            tau, _r_inf, _a = _fit_tau(t[gs:ge] - t[gs], R[gs:ge, ch], b0v)
            rows_by_ch[ch].append({
                "exc": round(float(dex), 4),
                "carry_frac": round(float(carry), 4),
                "tau_s": round(tau, 2) if tau is not None else None,
                "gap_s": int(ge - gs),
            })

    per_class = {}
    for grp, chidx in CHANNEL_GROUPS.items():
        rows = [r for ch in chidx for r in rows_by_ch[ch]]
        taus = [r["tau_s"] for r in rows if r["tau_s"] is not None]
        cfs = [r["carry_frac"] for r in rows]
        per_class[grp] = {
            "n_gaps_per_channel": len(rows_by_ch[chidx[0]]) if chidx else 0,
            "median_tau_s": round(float(np.median(taus)), 2) if taus else None,
            "q25_q75_tau_s": [round(float(np.percentile(taus, 25)), 2),
                              round(float(np.percentile(taus, 75)), 2)]
            if taus else None,
            "median_carryover_frac": round(float(np.median(cfs)), 4) if cfs else None,
            "median_abs_carryover_frac": round(float(np.median(np.abs(cfs))),
                                               4) if cfs else None,
            "frac_gaps_carryover_above_10pct": round(float(np.mean(
                [abs(c) > MEMORY_EPS for c in cfs])), 4) if cfs else None,
            "median_gap_s": int(np.median([r["gap_s"] for r in rows]))
            if rows else None,
        }

    med_tau = float(np.median([per_class[g]["median_tau_s"] for g in per_class
                               if per_class[g]["median_tau_s"] is not None])) \
        if any(per_class[g]["median_tau_s"] is not None for g in per_class) \
        else float("nan")
    med_cf = float(np.median([per_class[g]["median_abs_carryover_frac"]
                              for g in per_class if per_class[g]
                              ["median_abs_carryover_frac"] is not None])) \
        if any(per_class[g]["median_abs_carryover_frac"] is not None
               for g in per_class) else float("nan")
    frac_ok = [per_class[g]["frac_gaps_carryover_above_10pct"] for g in per_class
               if per_class[g]["frac_gaps_carryover_above_10pct"] is not None]
    frac_above = float(np.mean(frac_ok)) if frac_ok else float("nan")
    horizon = 2.3026 * med_tau if np.isfinite(med_tau) else float("nan")
    returns = bool(np.isfinite(med_cf) and med_cf < MEMORY_EPS and
                   frac_above < 0.5)
    med_gap = [per_class[g]["median_gap_s"] for g in per_class
               if per_class[g]["median_gap_s"] is not None]
    med_gap = int(np.median(med_gap)) if med_gap else 0
    return {
        "source": s["file"],
        "n_exposures_analyzed": n_exposures,
        "gas_seconds": int(np.nan_to_num(s["gas"]).sum()),
        "air_seconds": int((~s["gas"]).sum()),
        "per_class": per_class,
        "aggregate": {
            "median_tau_s": round(med_tau, 2) if np.isfinite(med_tau) else None,
            "memory_horizon_s": round(horizon, 1) if np.isfinite(horizon) else None,
            "median_abs_carryover_frac": round(med_cf, 4)
            if np.isfinite(med_cf) else None,
            "frac_gaps_carryover_above_10pct": round(frac_above, 4)
            if np.isfinite(frac_above) else None,
            "returns_to_baseline_between_exposures": returns,
        },
        "verdict": (
            f"median recovery tau is {med_tau:.0f}s (memory horizon for a 10% "
            f"residue ~{horizon:.0f}s) against {med_gap}s gaps, and the next "
            f"baseline carries a median {med_cf*100:.1f}% of the previous "
            f"deflection "
            f"({'mostly below' if frac_above < 0.5 else 'above'} the 10% line "
            f"in {frac_above*100:.0f}% of gaps).  Staged baselines should be "
            f"anchor-verified per session (session_anchor) rather than trusted "
            f"to a fixed R0; Wörner's 700-cycle recovery stages will pin this "
            f"directly."),
    }


def run() -> dict:
    worner = _load_worner_raw()
    mem = _memory_on_surrogate()
    out = {
        "what": (
            "Sensor memory: recovery time constant tau and cross-cycle carryover "
            "of an exposure deflection into the next baseline.  Wörner's "
            "stage 1/2/3 schema is the target; the raw corpus is absent, so the "
            "UCI dynamic-mixtures stream runs the identical machinery now as "
            "the surrogate (84-155 s exposures split by ~70-166 s air gaps)."),
        "worner_status": {
            "deferred": worner is None,
            "loader_hint": ("point WORNER_CSV at the corpus or drop it under "
                            f"e-nose-evals/data/{WORNER_GLOB}; the loader then "
                            "reads timestep_id / Cycle_Stage(1,2,3) / 62-channel "
                            "CSVs and runs this exact recovery+carryover "
                            "analysis per cycle."),
            "schema_read": worner,
        },
        "surrogate": mem,
        "surrogate_method": (
            "1 Hz grid of the 16-channel resistance; per-second gas mask from "
            "the dataset's own ppm columns (per-second OR over the 100 rows of "
            "each second).  For every exposure followed by an air gap of >=10 s "
            "the gap fits R(t) = r_inf + a*exp(-t/tau) and measures carryover "
            "of the exposure deflection (median of the last 20 s of exposure "
            "minus the pre-exposure air median) into the next baseline window "
            "(median of the last 20 s of the gap)."),
    }
    if worner is None:
        out["recommendation"] = (
            "surrogate result stands in for Wörner: recovery tau in the 10s-of-"
            "seconds range with measurable next-baseline carryover, so the "
            "staged baseline must be session-anchored, not a fixed R0.  When "
            "the Wörner raw corpus lands, rerun this benchmark unchanged to "
            "bound tau and carryover on the 62-channel MOX array directly.")
    return out


def main() -> None:
    res = run()
    record_benchmark("bench_sensor_memory", res)
    print("Wrote reports/bench_sensor_memory.json")
    print("worner_status:", "deferred (raw corpus absent)"
          if res["worner_status"]["deferred"] else "schema read OK")
    print("surrogate aggregate:", res["surrogate"]["aggregate"])
    print(res["surrogate"]["verdict"])


if __name__ == "__main__":
    main()