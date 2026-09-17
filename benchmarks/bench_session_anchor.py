"""WS1 — session-anchor + drift deconfounding.

The goal is to prove (or disprove) that per-session anchoring plus
environment residualization collapses the drift corruption measured in
bench_uci_drift (the *same* gas across batch1->batch10 sat 4.8-29.6
Mahalanobis units apart — larger than the inter-gas distances).

Two legs, both on already-held data:

 DRIFT ANCHOR (UCI drift, no T/RH recorded)
     Each batch is one session.  The session anchor is *self-normalization*:
     every recording is expressed relative to its own session statistics
     (per-feature mean/std), which is exactly the intra-session R0 convention
     applied at feature level.  We then remeasure batch1->batch10 same-gas
     separation and the batch1->batch10 transfer accuracy, and compare to the
     unanchored baselines (0.379 raw / 0.366 stable-40).

 ENV RESIDUALIZATION (turbulent, T/RH recorded per row)
     Each channel's logR0 is regressed on (RH, T); logR0_resid = logR0 - fit.
     We measure how much of the inter-repeat (same config, 6 repeats) cloud
     variance is environmental, and whether residualization regenerates
     separation between dose levels that the raw baseline blurred.
"""

from __future__ import annotations

import numpy as np

from _common import REPO_ROOT, record_benchmark

GAS_IDS = list(range(1, 7))


def _fit_z(batches):
    """Per-batch per-feature mean/std (the session anchor transform)."""
    fit = {}
    for b, (X, _) in batches.items():
        fit[b] = (X.mean(axis=0), X.std(axis=0) + 1e-9)
    return fit


def _apply(zfit, X: np.ndarray, b: int) -> np.ndarray:
    mu, sd = zfit[b]
    return (X - mu) / sd


def _transfer(batches, transform, n_stable: int | None = None) -> dict:
    """Logistic batch1 -> batchT accuracy; transform maps a batch's X."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    b1 = min(batches)
    bt = max(batches)
    X1, y1 = transform(batches[b1][0]), batches[b1][1]
    XT, yT = transform(batches[bt][0]), batches[bt][1]

    feat = list(range(X1.shape[1]))
    if n_stable is not None:
        means = {b: batches[b][0].mean(axis=0) for b in batches}
        arr = np.stack([means[b] for b in sorted(means)])
        centered = arr - arr.mean(axis=0)
        scale = np.abs(arr.mean(axis=0)) + 1e-9
        stability = scale / (np.sqrt(centered.var(axis=0)) + 1e-9)
        feat = list(np.argsort(-stability)[:n_stable])

    sc = StandardScaler()
    model = LogisticRegression(max_iter=2000, C=1.0)
    model.fit(sc.fit_transform(X1[:, feat]), y1)
    return round(float(model.score(sc.transform(XT[:, feat]), yT)), 4)


def _batch_of(X: np.ndarray, batches) -> int:
    """Recover a batch id for transform application (X is a single batch)."""
    for b, (Xb, _) in batches.items():
        if Xb.shape[0] == X.shape[0] and np.allclose(Xb, X):
            return b
    raise ValueError("could not identify batch for transform")


def _turbulent_leg() -> dict:
    """Regress logR0 on (RH, T) per channel; quantify the residual gain."""
    import glob
    import pandas as pd

    files = sorted(glob.glob(str(
        REPO_ROOT / "e-nose-evals/data/turbulent-mixtures/"
                  "dataset_twosources_raw/*")))
    records = []          # (T, RH, log10 R0 per channel, config key, repeat idx)
    for f in files:
        v = pd.read_csv(f, header=None, dtype=np.float64).values
        if v.shape[1] < 11:
            continue
        stem = f.split("/")[-1]
        parts = stem.replace(".txt", "").split("_")   # 000_Et_L_CO_L
        T = float(np.median(v[:, 1]))
        RH = float(np.median(v[:, 2]))
        air = v[:, 3:11][: int(60.0 / 0.02)]
        R = 10.0 * (3110.0 - air) / air
        r0 = np.nanmedian(R, axis=0)
        if parts[0] and len(parts) >= 5:
            cfg = f"{parts[1]}_{parts[2]}_{parts[3]}_{parts[4]}"
        else:
            cfg = stem
        records.append((T, RH, np.log10(np.clip(r0, 1e-6, None)), cfg))
    records.sort(key=lambda r: r[3])

    # per-channel OLS on the full 180-recording pool
    from collections import defaultdict
    grouped = defaultdict(list)
    for T, RH, logr0, cfg in records:
        grouped[cfg].append((T, RH, logr0))

    X = np.array([(T, RH) for T, RH, _, _ in records])
    Y = np.array([logr0 for _, _, logr0, _ in records])   # (n, 8)

    design = np.column_stack([np.ones(len(X)), X])          # [1, T, RH]
    beta_list, r2_list, resid_list = [], [], []
    for ch in range(Y.shape[1]):
        y = Y[:, ch]
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ beta
        r2 = 1.0 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
        beta_list.append(beta)
        r2_list.append(float(r2))
        resid_list.append(resid)
    resid_mat = np.stack(resid_list, axis=1)

    cfg_keys = sorted(grouped)
    per_cfg_raw = {c: np.array([r[2] for r in v]) for c, v in grouped.items()}
    resid_by_cfg = defaultdict(list)
    for (T, RH, logr0, cfg), resid in zip(records, resid_mat):
        resid_by_cfg[cfg].append(resid)
    per_cfg_resid = {c: np.stack(v) for c, v in resid_by_cfg.items()}

    def repeat_scatter(cloud):
        # mean within-config std of log10 R0 (residuals are centered ~0, so CV is
        # meaningless there; report the variance directly)
        stds = [float(np.std(c, axis=0).mean()) for c in cloud.values() if len(c) > 1]
        return float(np.mean(stds)) if stds else float("nan")

    raw_scatter = repeat_scatter(per_cfg_raw)
    resid_scatter = repeat_scatter(per_cfg_resid)

    # dose-separation: ethylene-only configs across 4 levels, per channel
    from ontology.observability import response_separation
    sep_raw, sep_resid = {}, {}
    et_levels = [(0.0, "n"), (31.0, "L"), (46.0, "M"), (96.0, "H")]
    for i, (cA, lvA) in enumerate(et_levels):
        for (cB, lvB) in et_levels[i + 1:]:
            key = f"{lvA}_vs_{lvB}"
            cloudA = np.stack([per_cfg_raw[c] for c in cfg_keys if f"Et_{lvA}" in c])
            cloudB = np.stack([per_cfg_raw[c] for c in cfg_keys if f"Et_{lvB}" in c])
            resA = np.stack([per_cfg_resid[c] for c in cfg_keys if f"Et_{lvA}" in c])
            resB = np.stack([per_cfg_resid[c] for c in cfg_keys if f"Et_{lvB}" in c])
            if len(cloudA) and len(cloudB):
                sep_raw[key] = round(response_separation(np.vstack(cloudA).reshape(-1, 8),
                                                         np.vstack(cloudB).reshape(-1, 8)), 3)
                sep_resid[key] = round(response_separation(np.vstack(resA).reshape(-1, 8),
                                                           np.vstack(resB).reshape(-1, 8)), 3)

    return {
        "n_recordings": len(records),
        "per_channel_logR0_ols": {
            f"ch{i}": {"d_logR0_dRH": round(float(beta[2]), 5),
                       "d_logR0_dT": round(float(beta[1]), 5),
                       "R2": round(r2_list[i], 4)}
            for i, beta in enumerate(beta_list)},
        "mean_repeat_scatter_log10": {"raw_logR0": round(raw_scatter, 5),
                                      "residualized": round(resid_scatter, 5),
                                      "scatter_reduction_pct":
                                          round(100 * (1 - resid_scatter / raw_scatter), 1)
                                          if raw_scatter else None},
        "ethylene_dose_separation": {
            "raw_logR0": sep_raw, "residualized": sep_resid,
            "gain": {k: round(sep_resid[k] - sep_raw[k], 3) for k in sep_raw}},
    }


def run() -> dict:
    import datasets
    batches = datasets.load_drift_batches()["batches"]

    b1, bt = min(batches), max(batches)
    zfit = _fit_z(batches)
    from ontology.observability import response_separation
    raw_sep = {}
    for g in GAS_IDS:
        A = batches[b1][0][batches[b1][1] == g]
        B = batches[bt][0][batches[bt][1] == g]
        if len(A) and len(B):
            raw_sep[f"gas{g}"] = round(response_separation(A, B), 3)
    anch_sep = {}
    for g in GAS_IDS:
        A = _apply(zfit, batches[b1][0], b1)[batches[b1][1] == g]
        B = _apply(zfit, batches[bt][0], bt)[batches[bt][1] == g]
        if len(A) and len(B):
            anch_sep[f"gas{g}"] = round(response_separation(A, B), 3)

    trans = lambda X: _apply(zfit, X, _batch_of(X, batches))  # noqa: E731
    transfer = {
        "raw_all128": _transfer(batches, lambda X: X),
        "anchored_all128": _transfer(batches, trans),
        "raw_stable40": _transfer(batches, lambda X: X, n_stable=40),
        "anchored_stable40": _transfer(batches, trans, n_stable=40),
    }

    return {
        "drift_anchor": {
            "batch_pair": f"{b1}_vs_{bt}",
            "raw_separation": raw_sep,
            "anchored_separation": anch_sep,
            "separation_change": {g: round(anch_sep[g] - raw_sep[g], 3)
                                  for g in raw_sep},
            "transfer": transfer,
            "n_batches": len(batches),
        },
        "environment_residualization": _turbulent_leg(),
        "recommendation": (
            "expression rules: record a session_anchor (60 s clean air + T/RH/P) "
            "and normalize every recording to its own session statistics before "
            "sharing; residualize calibrated-absolute features on the recorded "
            "environment before treating R0 as comparable across sessions."),
    }


def main() -> None:
    res = run()
    record_benchmark("bench_session_anchor", res)
    print("Wrote reports/bench_session_anchor.json")
    d = res["drift_anchor"]
    print("drift anchor separation batch1->10:",
          {k: v for k, v in d["separation_change"].items()})
    tr = d["transfer"]
    print(f"transfer raw={tr['raw_all128']} anchored={tr['anchored_all128']} "
          f"(stable40 raw={tr['raw_stable40']} anchored={tr['anchored_stable40']})")
    e = res["environment_residualization"]
    rpt = e["mean_repeat_scatter_log10"]
    print(f"turbulent repeat-scatter raw={rpt['raw_logR0']} "
          f"resid={rpt['residualized']} "
          f"({rpt['scatter_reduction_pct']}% reduction)")


if __name__ == "__main__":
    main()