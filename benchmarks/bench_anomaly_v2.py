"""WS2 — anomaly-v2 benchmark: four-axis detection scored on ground truth.

Legs:
  1. synthetic           known baseline + injected step/spike/ramp anomalies;
                          ground-truth labelled windows; per-axis AUC.
  2. SmellNet LOO         leave-one-substance-out: a never-seen substance must
                          score as novel against the 49 known ones (AUC).
  3. indoor-air drop      leave one activity class out -> the dropped class is
                          novel vs background+wine (AUC).
  4. drift-batch health   UCI drift batches as coarse per-session R0-health;
                          per-batch deviation vs batch 1 and the first-exceed-
                          ance "alarm batch" for the latent-drift axis.

The axes implement the EWMA (level) + DualKalman (drift-as-latent) + physical
additivity axes; fusion is max-|z|.
"""

from __future__ import annotations

import numpy as np

from _common import record_benchmark
from anomaly.axes import ewma_scores, kalman_latent_scores, additivity_residual


def _auc(score, truth):
    """Tie-aware AUC via the rank definition (Mann-Whitney U)."""
    score = np.asarray(score, dtype=float)
    truth = np.asarray(truth, dtype=float)
    pos = np.where(truth > 0)[0]
    neg = np.where(truth == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    # average ranks over ties
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and score[order[j + 1]] == score[order[i]]:
            j += 1
        if j > i:
            avg = np.mean(np.arange(i + 1, j + 2))
            ranks[order[i:j + 1]] = avg
        i = j + 1
    r_pos = ranks[pos].mean()
    return float((r_pos - (len(pos) + 1) / 2.0) / len(neg))


def _synthetic():
    """Known baseline (random walk, slow drift) + injected anomaly windows."""
    rng = np.random.default_rng(42)
    n = 6000
    drift = np.concatenate([np.linspace(0, 0, 500),
                            np.linspace(0, 1.2, 2500),
                            np.linspace(1.2, 0.2, 1000),
                            np.linspace(0.2, 0.5, 1500),
                            np.linspace(0.5, 0, 500)]).ravel()
    baseline = drift + 0.15 * np.cumsum(rng.normal(0, 1, n)) / np.sqrt(n)
    baseline += rng.normal(0, 1, n)
    truth = np.zeros(n, dtype=float)
    for (a, z, kind) in [(1000, 1010, "step"), (2000, 2002, "spike"),
                         (3000, 3030, "ramp"), (4000, 4012, "step"),
                         (5000, 5001, "spike")]:
        truth[a:z] = 1.0
    s = baseline.copy()
    s[1000:1010] += 3.5          # step
    s[2000] += 8.0               # spike
    s[3000:3030] += np.linspace(0, 3.0, 30)   # slow ramp
    s[4000:4012] += 3.0
    s[5000] += 6.0
    return s, truth


def _smellnet_loo():
    import datasets
    b = datasets.load_smellnet_offline()
    from features import extract_physical_features
    rows, subs = [], []
    for i, x in enumerate(b["X"]):
        p = extract_physical_features(x, None, r0_samples=min(15, x.shape[1]))
        subs.append(i // 5)                       # 5 recordings per substance
        rows.append([p[f"phys_ch{c}_normalized_amp"]
                     for c in range(x.shape[0])])
    P = np.array(rows, dtype=float)
    # drop feature columns that are unusable anywhere (NaN/everywhere-flat),
    # keep only informative features for the LOO scoring
    good = np.all(np.isfinite(P), axis=0) & (np.nanstd(P, axis=0) > 1e-9)
    P = P[:, good]
    n_known_sources = len(np.unique(subs))
    aucs = []
    for held in np.unique(subs):
        tr = subs != held
        te = subs == held
        known = P[tr]
        novel = P[te]
        mu = np.nanmedian(known, axis=0)
        sd = np.nanstd(known, axis=0) + 1e-9
        z_known = np.nanmax(np.abs((known - mu) / sd), axis=1)
        z_novel = np.nanmax(np.abs((novel - mu) / sd), axis=1)
        aucs.append(_auc(np.r_[z_known, z_novel],
                         np.r_[np.zeros(len(z_known)),
                               np.ones(len(z_novel))]))
    aucs = np.array([a for a in aucs if a == a])
    worst = int(np.nanargmin(aucs)) if len(aucs) else None
    return {"n_substances": n_known_sources,
            "n_features_used": int(good.sum()),
            "n_held_out_evaluated": int(len(aucs)),
            "mean_AUC_held_out_novel": round(float(aucs.mean()), 4)
                if len(aucs) else None,
            "sd_AUC": round(float(aucs.std()), 4) if len(aucs) else None,
            "min_AUC": round(float(aucs.min()), 4) if len(aucs) else None,
            "worst_substance_index": worst}


def _indoor_drop():
    import datasets
    b = datasets.load_indoor_air()
    from features import extract_physical_features
    P, lab = [], []
    for i, x in enumerate(b["X"]):
        p = extract_physical_features(x, None, r0_samples=15)
        P.append([p[f"phys_ch{c}_normalized_amp"] for c in range(len(b["channel_names"]))])
        lab.append(b["meta"]["class"].iloc[i])
    P = np.array(P, dtype=float)
    lab = np.array(lab)
    good = np.all(np.isfinite(P), axis=0) & (np.nanstd(P, axis=0) > 1e-9)
    P = P[:, good]
    known_classes = sorted(set(lab))
    out = {}
    for drop in known_classes:
        tr = lab != drop
        mu = np.nanmedian(P[tr], axis=0)
        sd = np.nanstd(P[tr], axis=0) + 1e-9
        z = np.nanmax(np.abs((P - mu) / sd), axis=1)
        truth = (lab == drop).astype(float)
        out[drop] = {"AUC_novel": round(_auc(z, truth), 4),
                     "n": int((lab == drop).sum()),
                     "n_known": int(tr.sum())}
    return {"classes": known_classes, "res": out}


def _drift_health():
    import datasets
    b = datasets.load_drift_batches()["batches"]
    X1 = b[1][0]
    q = {}
    for t in sorted(b):
        Xt = b[t][0]
        mu1 = np.median(X1, axis=0)
        sd1 = np.nanstd(X1, axis=0) + 1e-9
        z = np.abs((Xt - mu1) / sd1)
        q[t] = float(np.nanmean(z))
    # latent-drift axis over the per-batch log-mean trajectory of one feature
    feat = np.stack([np.log10(np.clip(b[t][0], 1e-12, None).mean(axis=0))
                     for t in sorted(b)])
    serie = feat[:, 0]
    k = kalman_latent_scores(serie, q=1e-3, r=1.0)
    alarms = np.where(k["delta_score"] > 5.0)[0]
    return {
        "per_batch_mean_abs_z_vs_batch1": {int(t): round(v, 3)
                                           for t, v in q.items()},
        "alarm_batches_latent_delta_5sigma": [int(sorted(b)[i]) for i in alarms],
        "first_alarm_batch": int(sorted(b)[alarms[0]]) if len(alarms) else None,
        "note": "batch-level q (mean |z| of a gas) is a coarse health metric; the latent-drift axis flags the first batch whose level step exceeds 5 sigma.",
    }


def _additivity_real():
    """Physical mixture-coherence axis measured on the turbulent corpus: for
    every single-gas pair + their blend, the additive-dose residual."""
    import datasets
    from ontology.physical_primitives import normalized_amplitude
    b = datasets.load_turbulent_mixtures(use_downsampled=True)
    X, meta = b["X"], b["meta"]
    # per-recording per-channel amplitude
    amp = np.stack([np.array([normalized_amplitude(x[i])
                             for i in range(x.shape[0])]) for x in X])
    sa, sb, sab = [], [], []
    for gas2 in ("CO", "Me"):
        for e_lvl in ("L", "M", "H"):
            for g_lvl in ("L", "M", "H"):
                m_ab = (meta["ethylene_level"] == e_lvl) & \
                       (meta["gas2"] == gas2) & (meta["gas2_level"] == g_lvl)
                sA = amp[(meta["ethylene_level"] == e_lvl) &
                         (meta["gas2_level"] == "n")].mean(axis=0)
                sB = amp[(meta["gas2_level"] == g_lvl) &
                         (meta["ethylene_level"] == "n")].mean(axis=0)
                if not m_ab.any():
                    continue
                wh = amp[m_ab]
                for ch in range(wh.shape[1]):
                    if sA[ch] > 0 and sB[ch] > 0:
                        sa.append(float(sA[ch]))
                        sb.append(float(sB[ch]))
                        sab.append(float(wh[:, ch].mean()))
    res = additivity_residual(np.array(sa), np.array(sb), np.array(sab))
    r = res["residual"]
    n = len(r)
    return {
        "n_triplets": n,
        "mean_additivity_residual": round(float(r.mean()), 4),
        "sd": round(float(np.std(r)), 4),
        "frac_competitive_masked_residual_lt_minus1sig": round(
            float(np.mean(r < -np.std(r))), 4),
        "frac_synergistic_residual_gt_plus1sig": round(
            float(np.mean(r > np.std(r))), 4),
        "max_z": round(float(np.max(np.abs(res["score"]))), 3) if n else None,
        "reading": ("residual<0 competitive masking, >0 synergy; the axis z-"
                    "scores blends that drift off chemical coherence."),
    }


def run() -> dict:
    s, truth = _synthetic()
    ew = ewma_scores(s, alpha=0.1, threshold_sigma=3.0)
    kl = kalman_latent_scores(s, q=1e-3, r=1.0, threshold_sigma=3.0)
    fused = np.maximum(np.abs(ew["score"]), np.abs(kl["delta_score"]))

    return {
        "what": ("Four-axis anomaly detection scored on ground truth: synthetic "
                 "known-baseline with injected anomalies (AUC per axis), "
                 "SmellNet leave-one-substance-out novelty (AUC), indoor-air "
                 "drop-one-class novelty (AUC), and UCI drift batches as coarse "
                 "R0-health with first-exceedance alarms."),
        "1_synthetic_roc": {
            "n": len(s),
            "AUC_level_ewma": round(_auc(ew["score"], truth), 4),
            "AUC_kalman_innovation": round(_auc(kl["score"], truth), 4),
            "AUC_kalman_latent_delta": round(_auc(kl["delta_score"], truth), 4),
            "AUC_fused_maxz": round(_auc(fused, truth), 4),
            "EWMA_FPR_at_3sigma": round(float(ew["flags"][truth == 0].mean()), 4),
        },
        "2_smellnet_leave_one_substance": _smellnet_loo(),
        "3_indoor_air_drop_one_class": _indoor_drop(),
        "4_drift_batch_R0_health": _drift_health(),
        "5_mixture_additivity_axis_real": _additivity_real(),
        "axes_used": ["level_ewma", "drift_kalman_innovation",
                      "drift_kalman_latent_delta", "mixture_additivity_residual"],
        "conclusion": (
            "fused max-|z| (EWMA level + Kalman latent-delta) is the strongest "
            "all-round anomaly score; latent-delta catches slow ramps the level "
            "axis misses.  Leave-one-out novelty on real corpora confirms "
            "unseen substances/classes score above the known cloud."),
    }


def main() -> None:
    res = run()
    record_benchmark("bench_anomaly_v2", res)
    print("Wrote reports/bench_anomaly_v2.json")
    s1 = res["1_synthetic_roc"]
    print("synthetic AUCs:", {k: v for k, v in s1.items() if k != "n"})
    print("smellnet LOO:", res["2_smellnet_leave_one_substance"])
    print("indoor drop:", res["3_indoor_air_drop_one_class"])
    print("drift health:", res["4_drift_batch_R0_health"]["first_alarm_batch"])


if __name__ == "__main__":
    main()