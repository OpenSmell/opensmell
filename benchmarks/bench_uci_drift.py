"""Benchmark 1 — UCI drift: observability + constant selection.

The UCI drift dataset is 10 batches of the same 16-sensor array measured across
months; batch index == drift time.  Two questions:

 OBSERVABILITY   can the 6 gases be told apart within a rig, and how much does
                 their separation degrade across batches (drift corruption)?
 CONSTANTS       which of the 128 features are *device-stable constants* across
                 batches, and does restricting to them help (or hurt) the
                 cross-batch classification transfer batch1 -> batch10?

This is the data-side complement of the physics self-check and the datasheet
priors: the empirical answer to "what transfers".
"""

from __future__ import annotations

import numpy as np

from _common import record_benchmark


def _load():
    import datasets
    return datasets.load_drift_batches()


def _clouds(X: np.ndarray, y: np.ndarray, gas_ids: list[int]) -> dict[str, np.ndarray]:
    return {f"gas{g}": X[y == g] for g in gas_ids if np.any(y == g)}


def _report_per_batch(batches) -> dict:
    from ontology.observability import distinguishability_report

    gas_ids = sorted({int(y) for _, (_, ys) in batches.items() for y in ys})
    per = {}
    for b, (X, y) in batches.items():
        clouds = _clouds(X, y, gas_ids)
        rep = distinguishability_report(clouds, threshold=1.0, representation="raw128")
        per[b] = rep.summary()
    return per


def _cross_batch_same_gas_separation(batches) -> dict:
    from ontology.observability import response_separation

    gas_ids = sorted({int(y) for _, (_, ys) in batches.items() for y in ys})
    b1 = batches[1][0]
    b10 = batches[max(batches)][0]
    out = {}
    for g in gas_ids:
        A = b1[batches[1][1] == g]
        B = b10[batches[max(batches)][1] == g]
        if len(A) and len(B):
            out[f"gas{g}"] = round(response_separation(A, B), 3)
    return {"batch_pairs": "1" + "_" + str(max(batches)), "separations": out}


def _constant_selection(batches, n_stable: int = 40) -> dict:
    """Rank the 128 features by cross-batch stability (small |batch variation|).

    'Constant' here = feature whose per-batch mean is nearly batch-independent,
    i.e. carries the gas signal without carrying the drift.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    gas_ids = sorted({int(y) for _, (_, ys) in batches.items() for y in ys})
    means = {b: X.mean(axis=0) for b, (X, _) in batches.items()}
    all_means = np.stack([means[b] for b in sorted(means)])
    # between-batch spread relative to pooled mean magnitude
    centered = all_means - all_means.mean(axis=0)
    batch_var = centered.var(axis=0)
    pooled_mean = abs(all_means.mean(axis=0))
    scale = pooled_mean + 1e-9
    stability = scale / (np.sqrt(batch_var) + 1e-9)
    stability_idx = np.argsort(-stability)  # most stable first

    X1, y1 = batches[1]
    bT = max(batches)
    XT, yT = batches[bT]

    def acc(feat_idx):
        model = LogisticRegression(max_iter=2000, C=1.0)
        sc = StandardScaler()
        Xtr = sc.fit_transform(X1[:, feat_idx])
        model.fit(Xtr, y1)
        Xte = sc.transform(XT[:, feat_idx])
        return float(model.score(Xte, yT))

    all_idx = list(range(X1.shape[1]))
    res = {
        "n_samples_batch1": int(len(y1)),
        "n_samples_batchT": int(len(yT)),
        "acc_all128": round(acc(all_idx), 4),
        "acc_stable_k": round(acc(list(stability_idx[:n_stable])), 4),
        "acc_least_stable_k": round(acc(list(stability_idx[-n_stable:])), 4),
        "n_stable": n_stable,
    }
    # sanity: is stability ranking actually predicting transfer? report best k
    best_k, best_acc, chosen = 1, 0.0, []
    for k in [10, 20, 40, 60, 80]:
        a = acc(list(stability_idx[:k]))
        if a > best_acc:
            best_acc, best_k = a, k
    res["acc_best_stable_prefix"] = round(best_acc, 4)
    res["best_k"] = best_k
    return res


def run() -> dict:
    batches = _load()["batches"]
    return {
        "within_batch_observability": _report_per_batch(batches),
        "cross_batch_same_gas_separation": _cross_batch_same_gas_separation(batches),
        "constant_selection": _constant_selection(batches),
    }


def main() -> None:
    res = run()
    record_benchmark("bench_uci_drift", res)
    print("Wrote reports/bench_uci_drift.json")
    ob = res["within_batch_observability"]
    print("within-batch ambiguity fractions by batch:",
          {k: round(v["ambiguous_fraction"], 3) for k, v in ob.items()})
    cs = res["constant_selection"]
    print(f"transfer batch1->{max(_load()['batches'])}: all={cs['acc_all128']} "
          f"stable-top{cs['n_stable']}={cs['acc_stable_k']} "
          f"least-stable={cs['acc_least_stable_k']}")
    print("cross-batch same-gas separation:", res["cross_batch_same_gas_separation"])


if __name__ == "__main__":
    main()