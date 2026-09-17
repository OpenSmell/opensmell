"""W2 — detector false-positive decomposition: chemistry vs memory (wave-4).

The exposure-memory audit (W1) established that a previous exposure leaves a
measurable residue in the next clean-air window.  This benchmark asks the
practical follow-up: how much of a real anomaly detector's clean-air false
alarms come from that *physical* memory residue rather than from the algorithm
itself?

It runs the production residual axes (explicit EWMA level, implicit DualKalman
innovation + latent-delta, max-|z| fusion) on the 16 channels of the UCI
dynamic-mixtures streams with the dataset's per-second ground-truth schedule:

  truth gas: scheduled ppm > 0
  clean air: scheduled ppm == 0

Every air-second alarm is then labelled by how long it is after the previous
exposure ended.  The FP rate as a function of that distance IS the memory
footprint of the detector:

  FP(bin) rising as gap shrinks   -> alarms are triggered by exposure residue
  FP(deep clean, >>60 s)          -> the algorithm's own false-alarm floor

The benchmark reports both, and the excess (short-gap FP - deep-clean FP) as
the memory-attributable share of the false-positive budget.
"""

from __future__ import annotations

import numpy as np

from _common import record_benchmark
from harness.loaders import load_dynamic_mixtures
from anomaly.axes import ewma_scores, kalman_latent_scores

FILES = ["ethylene_CO.txt", "ethylene_methane.txt"]
SR = 100.0
THRESHOLDS = [3.0, 5.0]
BIN_EDGES = [0, 10, 30, 60, 120, 300, np.inf]
DEEP_CLEAN_S = 60.0        # intermediate floor measured >=60 s after exposure end
LONG_GAP_S = 300.0         # truly-recovered floor measured >=300 s after
NEXT_SAFE_S = 30.0         # and >=30 s before the next exposure start
REF_WIN_S = 600            # robust reference window (first 600 s of stream)


def _grid(b):
    """1 Hz resistance grid + per-second gas truth."""
    r = np.asarray(b["X"], dtype=np.float64)
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
    gas = (et_s > 0) | (g2_s > 0)
    return R, gas


def _runs(mask):
    runs, s = [], None
    m = np.pad(mask.astype(int), (1, 1), constant_values=0)
    for i in range(1, len(m)):
        if m[i] and not m[i - 1]:
            s = i - 1
        if not m[i] and m[i - 1] and s is not None:
            runs.append((s, i - 1))
            s = None
    return runs


def _standardise(R, gas):
    """Robust per-channel z-score vs the initial air reference, so every
    channel feeds the axes at a comparable level and the max-|z| fuse is fair.
    """
    ref = np.where(~gas[:int(min(REF_WIN_S, len(gas)))])[0]
    if len(ref) < 50:
        ref = np.arange(min(600, R.shape[0]))
    med = np.nanmedian(R[ref], axis=0)
    mad = np.nanmedian(np.abs(R[ref] - med), axis=0)
    sig = 1.4826 * mad
    sig = np.where(sig < 1e-12, 1e-12, sig)
    return (R - med) / sig


def _distances(gas):
    """Vectorised distance-from-exposure arrays."""
    n = len(gas)
    t = np.arange(n, dtype=np.int64)
    runs = _runs(gas)
    starts = np.array([s for (s, e) in runs], dtype=np.int64)
    ends = np.array([e + 1 for (s, e) in runs], dtype=np.int64)  # exclusive
    since = np.full(n, np.inf)
    to_next = np.full(n, np.inf)
    # last exposure END <= t  (time since the memory was last refreshed)
    li = np.searchsorted(ends, t, side="right") - 1
    ok = li >= 0
    since[ok] = t[ok] - ends[li[ok]]
    # first exposure START >= t  (time until the next refresh)
    fi = np.searchsorted(starts, t, side="left")
    ok = fi < len(starts)
    to_next[ok] = starts[fi[ok]] - t[ok]
    # inside exposure: 0 distance
    since[gas] = 0.0
    to_next[gas] = 0.0
    return since, to_next


def _analyze_file(name):
    b = load_dynamic_mixtures(file_name=name)
    R, gas = _grid(b)
    Z = _standardise(R, gas)
    since, to_next = _distances(gas)
    n = len(gas)

    # ---- run axes per channel, fuse by max-|z| ----
    n_ch = Z.shape[1]
    axis_out = {k: [] for k in ("ewma", "kalman_innov", "kalman_delta")}
    for ch in range(n_ch):
        s = Z[:, ch]
        ew = ewma_scores(s, alpha=0.1, threshold_sigma=max(THRESHOLDS))
        kl = kalman_latent_scores(s, q=1e-4, r=1.0,
                                  threshold_sigma=max(THRESHOLDS))
        axis_out["ewma"].append(ew["score"])
        axis_out["kalman_innov"].append(kl["score"])
        axis_out["kalman_delta"].append(kl["delta_score"])
    scores = {k: np.max(np.abs(np.stack(v)), axis=0) for k, v in axis_out.items()}
    fused = np.max(np.stack(list(scores.values())), axis=0)

    # ---- per-axis + threshold confusion ----
    all_sc = {"level_ewma": scores["ewma"],
              "kalman_innovation": scores["kalman_innov"],
              "kalman_latent_delta": scores["kalman_delta"],
              "fused_maxz": fused}
    per_axis = {}
    for ax, sc in all_sc.items():
        for thr in THRESHOLDS:
            flag = sc > thr
            tp = float(flag[gas].mean()) if gas.any() else float("nan")
            fp = float(flag[~gas].mean()) if (~gas).any() else float("nan")
            per_axis[f"{ax}_thr{thr:g}"] = {
                "TP_in_gas": round(tp, 4),
                "FP_in_air": round(fp, 4),
            }

    # ---- memory footprint of the fused detector at thr 5 ----
    thr = 5.0
    fused_flag = fused > thr
    air = ~gas
    bins = []
    for a, z in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        m = air & (since >= a) & (since < z)
        n_air = int(m.sum())
        bins.append({
            "s_after_exposure": f"{a}-{z if np.isfinite(z) else 'inf'}",
            "n_air_seconds": n_air,
            "FP_rate": round(float(fused_flag[m].mean()), 4) if n_air else None,
        })
    deep = air & ((since >= DEEP_CLEAN_S) | ~np.isfinite(since)) & (to_next >= NEXT_SAFE_S)
    n_deep = int(deep.sum())
    fp_floor = float(fused_flag[deep].mean()) if n_deep else None
    far = air & np.isfinite(since) & (since >= LONG_GAP_S) & (to_next >= NEXT_SAFE_S)
    n_far = int(far.sum())
    fp_far = float(fused_flag[far].mean()) if n_far else None
    near = air & (since < 10)
    n_near = int(near.sum())
    fp_near = float(fused_flag[near].mean()) if n_near else None
    excess_60 = (fp_near - fp_floor) if not None in (fp_near, fp_floor) else None
    excess_300 = (fp_near - fp_far) if not None in (fp_near, fp_far) else None

    return {
        "source": name,
        "n_seconds": int(n),
        "gas_seconds": int(gas.sum()),
        "air_seconds": int(air.sum()),
        "per_axis": per_axis,
        "fused_memory_footprint": {
            "bins": bins,
            "deep_clean_fp_floor_60s": round(fp_floor, 4)
                if fp_floor is not None else None,
            "long_gap_fp_floor_300s": round(fp_far, 4)
                if fp_far is not None else None,
            "fp_0_10s_after_exposure": round(fp_near, 4)
                if fp_near is not None else None,
            "memory_excess_vs_60s": round(excess_60, 4)
                if excess_60 is not None else None,
            "memory_excess_vs_300s": round(excess_300, 4)
                if excess_300 is not None else None,
        },
        "method": (
            "per-channel robust z vs the initial air window; axes = EWMA level, "
            "DualKalman innovation + latent-delta; fuse = max|z| over channels. "
            "Alarms in air seconds are binned by time since the previous "
            "exposure ended.  The FP-vs-gap curve is its own readout: the "
            "60 s-deep-clean FP is an intermediate floor, the >=300 s-long-gap "
            "FP is the truly-recovered (no-memory) algorithm floor, and each "
            "memory-excess is FP(0-10 s after exposure) minus the reference "
            "floor — the memory-attributable share of the FP budget."),
    }


def run() -> dict:
    files = {}
    for name in FILES:
        try:
            files[name] = _analyze_file(name)
        except FileNotFoundError:
            continue
    f60s = [v["fused_memory_footprint"]["deep_clean_fp_floor_60s"]
            for v in files.values()
            if v["fused_memory_footprint"]["deep_clean_fp_floor_60s"] is not None]
    f300s = [v["fused_memory_footprint"]["long_gap_fp_floor_300s"]
             for v in files.values()
             if v["fused_memory_footprint"]["long_gap_fp_floor_300s"] is not None]
    e60 = [v["fused_memory_footprint"]["memory_excess_vs_60s"]
           for v in files.values()
           if v["fused_memory_footprint"]["memory_excess_vs_60s"] is not None]
    e300 = [v["fused_memory_footprint"]["memory_excess_vs_300s"]
            for v in files.values()
            if v["fused_memory_footprint"]["memory_excess_vs_300s"] is not None]
    return {
        "what": (
            "Wave-4 detector false-positive decomposition: how much of a real "
            "anomaly detector's clean-air false alarms are physical memory "
            "residue (exposure-dependent) vs the algorithm's own floor.  The "
            "production residual axes run on the dynamic-mixtures ground-truth "
            "schedule and every air alarm is binned by distance from the "
            "previous exposure; floors are measured at >=60 s (mid) and "
            ">=300 s (fully recovered)."),
        "files": files,
        "aggregate": {
            "median_deep_clean_fp_floor_60s": round(
                float(np.median(f60s)), 4) if f60s else None,
            "median_long_gap_fp_floor_300s": round(
                float(np.median(f300s)), 4) if f300s else None,
            "median_memory_excess_vs_60s": round(
                float(np.median(e60)), 4) if e60 else None,
            "median_memory_excess_vs_300s": round(
                float(np.median(e300)), 4) if e300 else None,
            "n_files": len(files),
        },
    }


def main() -> None:
    res = run()
    record_benchmark("bench_detector_memory_fp", res)
    print("Wrote reports/bench_detector_memory_fp.json")
    for fname, fr in res["files"].items():
        f = fr["fused_memory_footprint"]
        print(f"\n{fname}: air {fr['air_seconds']}s "
              f"FP floor60s {f['deep_clean_fp_floor_60s']} "
              f"floor300s {f['long_gap_fp_floor_300s']} "
              f"FP<10s {f['fp_0_10s_after_exposure']} "
              f"excess_vs300 {f['memory_excess_vs_300s']}")
        print("  FP-vs-gap bins:", [(b["s_after_exposure"], b["FP_rate"])
                                    for b in f["bins"]])
        for ax, v in list(fr["per_axis"].items())[-2:]:
            print(f"  {ax}: TP {v['TP_in_gas']} FP {v['FP_in_air']}")
    print("aggregate:", res["aggregate"])


if __name__ == "__main__":
    main()