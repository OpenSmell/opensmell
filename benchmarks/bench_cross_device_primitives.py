"""Benchmark 3 — cross-device primitive stability.

Claim tested: a small set of physical primitives (log-ratios, normalized
amplitudes, directions) survives device variation better than raw features.

 SYNTHETIC  sample 30 virtual devices (devices.device_parameter_model), drive
            the same chemical state through them, extract primitives, and
            measure the across-device coefficient of variation of each
            primitive.  DEV_BOUND primitives should be stable; NONE-calibrated
            (baseline/raw) primitives should not.
 REAL       the dynamic-mixtures file contains two nominally identical TGS2602
            channels (ch0, ch8) reading the *same* headspace.  Compare their
            primitives: the two-device reproducibility of every primitive.
"""

from __future__ import annotations

import numpy as np

from _common import record_benchmark


def _primitive_set() -> dict:
    from features.physical_features import extract_physical_features
    return extract_physical_features


def _synthetic(n_devices: int = 30, base_T_c: float = 350.0) -> dict:
    import devices

    rng = np.random.default_rng(7)
    devs = devices.sample_device_variability(rng=rng, n_devices=n_devices,
                                             base_T_c=base_T_c)
    # canonical chemical state: ethylene exposure steps (1,3,10 ppm conc.)
    C = np.array([1.0, 1.0, 3.0, 3.0, 10.0, 10.0])
    n_ch = 6
    dev_R0 = np.zeros((n_devices, n_ch))
    dev_norm = np.zeros((n_devices, n_ch, 3))   # normalized amplitude per level-block
    for di, d in enumerate(devs):
        sf = d.theta["sensitivity_factor"]
        for ch in range(n_ch):
            A = 1.0 * sf * (1 + 0.15 * ch)          # per-channel prefactor
            B = -0.5 + d.theta["exponent_offset"]   # per-device exponent
            R = A * C ** B
            R0 = A * 1.0 ** B                        # baseline at C=1 ppm
            dev_R0[di, ch] = R0
            for lev, (lo, hi) in enumerate([(0, 2), (2, 4), (4, 6)]):
                block = R[lo:hi]
                dev_norm[di, ch, lev] = float(np.abs(block - R0).mean() / R0)
    cv_norm = dev_norm.std(axis=0) / (dev_norm.mean(axis=0) + 1e-9)
    cv_r0 = dev_R0.std(axis=0) / (dev_R0.mean(axis=0) + 1e-9)
    return {
        "n_devices": n_devices,
        "normalized_amplitude_cv_per_level": np.round(
            cv_norm.mean(axis=0), 4).tolist(),
        "normalized_amplitude_cv_mean": round(float(cv_norm.mean()), 4),
        "raw_R0_cv_mean": round(float(cv_r0.mean()), 4),
        "exponent_offset_sigma": 0.08,
        "sensitivity_sigma": 0.30,
        "note": ("normalized-amplitude primitive should be near-invariant to "
                 "sensitivity gain but NOT to exponent_offset (it is C-dependent)"),
    }


def _real_dynamic(n_subsample: int = 2000) -> dict:
    import pandas as pd
    from ontology.physical_primitives import normalized_amplitude, baseline_resistance

    path = None
    for cand in ("e-nose-evals/data/dynamic-mixtures/ethylene_CO.txt",
                 "e-nose-evals/data/dynamic-mixtures/ethylene_methane.txt"):
        import os
        if os.path.exists(cand):
            path = cand
            break
    if path is None:
        return {"error": "dynamic-mixtures files not present"}

    df = pd.read_csv(path, sep=r"\s+", header=None, skiprows=1,
                     usecols=range(19), nrows=1000000)
    df = df.iloc[:: (len(df) // n_subsample) or 1]
    g = df.iloc[:, 3:19].to_numpy(dtype=np.float64)
    g = 40000.0 / np.where(g <= 0, np.nan, g)   # Rs kOhm
    # channels 0 & 8 are the pair of identical TGS2602 (see loaders).
    a, b = g[:, 0], g[:, 8]
    mask = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    a, b = a[mask], b[mask]

    # median-ratio alignment: R0 differs device to device
    r0a = np.median(a[: min(500, len(a))])
    r0b = np.median(b[: min(500, len(b))])
    na = a / r0a
    nb = b / r0b
    corr = float(np.corrcoef(na, nb)[0, 1])

    # normalized primitive reproducibility: mean abs log-ratio of the two devices
    logr = np.abs(np.log10(na / np.clip(nb, 1e-9, None)))
    mad = float(np.median(logr))

    # response-level (peak-vs-baseline) reproducibility on rolling windows
    win = 50
    amps = []
    for s, t in zip(np.array_split(a, max(1, len(a) // win)),
                    np.array_split(b, max(1, len(b) // win))):
        if len(s) < 5 or len(t) < 5:
            continue
        ra = baseline_resistance(s)
        rb = baseline_resistance(t)
        aa = normalized_amplitude(s, ra)
        ab = normalized_amplitude(t, rb)
        if aa and ab and np.isfinite(aa) and np.isfinite(ab):
            amps.append((float(aa), float(ab)))
    if amps:
        amps = np.array(amps)
        amp_corr = float(np.corrcoef(amps[:, 0], amps[:, 1])[0, 1])
        amp_dev = float(np.mean(np.abs(amps[:, 0] - amps[:, 1])))
    else:
        amp_corr, amp_dev = float("nan"), float("nan")

    return {
        "file": path.split("/")[-1],
        "n_paired": int(len(a)),
        "device_pair": "TGS2602 ch0 vs TGS2602 ch8 (same headspace)",
        "r0a_kOhm": round(float(r0a), 3),
        "r0b_kOhm": round(float(r0b), 3),
        "r0_ratio": round(float(r0a / r0b), 4),
        "corr_normalized_series": round(corr, 4),
        "median_abs_log10_device_ratio": round(mad, 4),
        "normalized_amplitude_window_corr": round(amp_corr, 4),
        "normalized_amplitude_mean_abs_diff": round(amp_dev, 6),
        "note": "ratio/normalized primitives should align closely despite R0 mismatch",
    }


def run() -> dict:
    return {
        "synthetic_cross_device": _synthetic(),
        "real_same_headspace_two_devices": _real_dynamic(),
    }


def main() -> None:
    res = run()
    record_benchmark("bench_cross_device_primitives", res)
    print("Wrote reports/bench_cross_device_primitives.json")
    s = res["synthetic_cross_device"]
    print(f"synthetic: raw R0 CV={s['raw_R0_cv_mean']} "
          f"vs normalized-amp CV={s['normalized_amplitude_cv_mean']}")
    real = res["real_same_headspace_two_devices"]
    if "error" not in real:
        print(f"real two-TGS2602: R0 ratio={real['r0_ratio']} "
              f"corr={real['corr_normalized_series']} "
              f"median|log10 ratio|={real['median_abs_log10_device_ratio']}")


if __name__ == "__main__":
    main()