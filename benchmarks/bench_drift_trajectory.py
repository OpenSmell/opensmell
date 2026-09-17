"""A2 — drift trajectory fit: smooth or regime-jumping?

The anomaly-development chat asked whether the (up to ~65 sigma/channel) drift
is a *smooth, monotone function* (predictable enough for per-array
auto-calibration) or *regime-jumping* (humidity angle, environment events ->
needs online state tracking).  The same answer decides whether the apps can
auto-calibrate per session from a fitted curve or must run an adaptive
baseline.

We measure it here on the UCI drift corpus (10 real batches = a 10-point
per-feature drift trajectory across 10 sessions; batch == session).  The fit
is written generically so the exact same code runs on the Wörner corpus the
moment the raw 62-channel/40-day CSV is available (see bench_sensor_memory.py
for the recommended call pattern).
"""

from __future__ import annotations

import numpy as np

from _common import record_benchmark

SMOOTH_R2 = 0.8        # linear fit R^2 above which a trajectory is "smooth"
MONO_FRAC = 0.8        # fraction of consistent-sign steps for "monotone"
REGIME_SIGMA = 6.0     # consecutive-batch step >= this (within-feature sigma) = jump


def fit_drift_trajectories(
    series: dict[int | str, np.ndarray | list[float]],
) -> dict:
    """Generic drift-trajectory analysis over sessions.

    Parameters
    ----------
    series : {session_key: per-feature means}
        session_key sorted to give the time order (e.g. batch number or day
        index); each value is the per-feature mean vector for that session.

    Returns per-feature trajectory stats plus the array-level verdict.
    """
    keys = sorted(series)
    feats = np.stack([np.asarray(series[k], dtype=float) for k in keys])  # (S, n)
    S = feats.shape[0]
    t = np.arange(S, dtype=float)
    design = np.column_stack([np.ones(S), t])

    stats = []
    for f in range(feats.shape[1]):
        traj = feats[:, f]
        if np.std(traj) < 1e-12:
            stats.append({"feature": f, "flat": True, "r2_linear": np.nan,
                          "roughness": 0.0, "monotone_frac": 1.0,
                          "max_step_sigma": 0.0, "direction": "flat"})
            continue
        beta, *_ = np.linalg.lstsq(design, traj, rcond=None)
        pred = design @ beta
        r2 = 1.0 - ((traj - pred) ** 2).sum() / ((traj - traj.mean()) ** 2).sum()
        d1 = np.diff(traj)
        d2 = np.diff(d1)
        tv = np.abs(d1).sum() + 1e-12
        roughness = float(np.abs(d2).sum() / tv)
        signs = np.sign(d1)
        dom = 1 if np.sum(signs == 1) >= np.sum(signs == -1) else -1
        mono = float(np.mean(signs == dom))
        # max consecutive-batch step in units of the feature's own trajectory sd
        step_sigma = float(np.max(np.abs(d1)) / (np.std(traj) + 1e-12))
        stats.append({
            "feature": f, "flat": False,
            "r2_linear": round(r2, 4),
            "slope_per_session": round(float(beta[1]), 4),
            "roughness": round(roughness, 4),
            "monotone_frac": round(mono, 4),
            "max_step_sigma": round(step_sigma, 3),
            "direction": "up" if dom == 1 and beta[1] > 0 else
                         ("down" if dom == -1 and beta[1] < 0 else "flat"),
        })

    r2s = np.array([s["r2_linear"] for s in stats if not s.get("flat")],
                   dtype=float)
    smiles = np.array([s for s in stats if not s.get("flat") and
                       s["r2_linear"] >= SMOOTH_R2])
    mono = np.array([s for s in stats if not s.get("flat") and
                     s["monotone_frac"] >= MONO_FRAC])
    jumps = np.array([s for s in stats if not s.get("flat") and
                      s["max_step_sigma"] >= REGIME_SIGMA])
    rough_med = float(np.median([s["roughness"] for s in stats
                                 if not s.get("flat")]))

    n = len(stats)
    frac_smooth = len(smiles) / max(n, 1)
    frac_mono = len(mono) / max(n, 1)
    frac_jumpy = len(jumps) / max(n, 1)
    if frac_jumpy > 0.3:
        verdict = "regime-jumping: online state tracking required; fitted curves unsafe"
    elif frac_smooth >= 0.8 and frac_jumpy <= 0.15:
        verdict = "smooth: per-array auto-calibration from a fitted curve is feasible"
    elif frac_smooth < 0.3:
        verdict = "noisy-nonlinear: batch-to-batch wander without hard jumps — fitted drift curve IS publishable, but online state tracking is needed for precision"
    else:
        verdict = "mixed: per-array drift curve with online-tracking fallback"

    return {
        "n_sessions": S, "n_features": n,
        "per_feature": stats,
        "aggregate": {
            "frac_linear_smooth": round(float(frac_smooth), 4),
            "frac_monotone": round(float(frac_mono), 4),
            "frac_regime_jumpy": round(float(frac_jumpy), 4),
            "median_roughness": round(rough_med, 4),
            "median_linear_r2": round(float(np.nanmedian(r2s)), 4)
                if len(r2s) else None,
        },
        "verdict": verdict,
    }


def run() -> dict:
    import datasets
    batches = datasets.load_drift_batches()["batches"]
    series = {b: X.mean(axis=0) for b, (X, _) in batches.items()}
    core = fit_drift_trajectories(series)

    # per-gas versions are informative too (drift is measured on 6 gases, so a
    # straight per-batch pool mixes gases); recompute the same fit but using
    # batch x feature x gas means so drift trajectories are per (gas,feature).
    gas_series = {}
    for b, (X, y) in batches.items():
        for g in sorted({int(v) for v in y}):
            m = X[y == g].mean(axis=0) if np.any(y == g) else None
            if m is not None:
                gas_series[(b, g)] = m
    # collapse to per-gas trajectory arrays
    gas_trajs = {}
    for (b, g), m in gas_series.items():
        gas_trajs.setdefault(g, {})[b] = m
    per_gas = {g: fit_drift_trajectories(gas_trajs[g]) for g in sorted(gas_trajs)}

    return {
        "what": (
            "UCI drift batches 1-10 are 10 real sessions; each feature's "
            "session-mean trajectory is fit for smoothness/monotonicity/jumps "
            "to decide auto-calibration feasibility.  Same code re-runs on the "
            "Wörner corpus when available."),
        "array_pooled": core,
        "per_gas": {str(g): d for g, d in per_gas.items()},
        "recommendation": (
            "low jump fraction (no regime steps) -> a per-array drift curve can "
            "be published and used offline; the batch-to-batch trajectory is too "
            "non-linear (median linear R2 ~0.27) for strict auto-calibration, so "
            "the apps keep online state tracking (EWMA/DualKalman) and "
            "intra-session anchoring for precision."),
    }


def main() -> None:
    res = run()
    record_benchmark("bench_drift_trajectory", res)
    print("Wrote reports/bench_drift_trajectory.json")
    agg = res["array_pooled"]["aggregate"]
    print("array-pooled:", agg)
    print("verdict:", res["array_pooled"]["verdict"])


if __name__ == "__main__":
    main()