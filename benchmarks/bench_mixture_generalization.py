"""WS3 — additive-dose mixture generalisation: hold-out R² on dynamic streams.

The additive-dose law (OpenSmell's physical null for n-type MOX arrays) says:

    s_AB  =  (1 + s_A)(1 + s_B) - 1

where s_X is the single-gas normalised amplitude |R - R0| / R0 for gas X at
its *own* concentration.  The law follows from independent Langmuir adsorption
on separate surface sites (two gases adding linearly in dose).

We test it here on the two UCI dynamic-mixture continuous streams
(ethylene+Methane and ethylene+CO, 16 Figaro TGS channels) by:

  1. estimating R0 from the clean-air block at the start of each stream;
  2. measuring each config's steady-state amplitude from the tail (last 30%)
     of that config's blocks (skipping onset transients);
  3. training only on single-gas amplitudes and predicting the *held-out
     mixture* amplitude via the additive-dose law;
  4. reporting hold-out R² and bias in log10-amplitude space.

A "stacked-identical-TGS bootstrap" uses the 4 replicate TGS2602 channels
(ch 0, 1, 8, 9) as four independent device-level draws to bound the
law-deviation variance that is internal to a sensor family.
"""

from __future__ import annotations

import numpy as np

from _common import REPO_ROOT, record_benchmark

SR = 100   # dynamic mixtures sample rate (Hz)
TAIL_FRAC = 0.30      # use the *end* of each config's block (steady state)
AIR_SEC_FRAC = 0.50   # first 50% of the air block as R0 training window
KEY_MOD = 100000      # separator for the (et,g2) -> int sort key


def _load_raw():
    """Load both dynamic files, return dict keyed by filename."""
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location(
        "hl", str(REPO_ROOT / "e-nose-evals/harness/loaders.py"))
    hl = module_from_spec(spec)
    spec.loader.exec_module(hl)
    out = {}
    for f in sorted(hl.DYNAMIC_DIR.glob("*.txt")):
        p = hl.load_dynamic_mixtures(file_name=f.name)
        out[f.name] = {
            "X": p["X"],
            "et": p["meta"]["ethylene_ppm"].to_numpy(dtype=float),
            "g2": p["meta"]["gas2_ppm"].to_numpy(dtype=float),
            "sr": SR,
            "channels": list(p["channel_names"]),
        }
    return out


def _run_one_file(name, d):
    """Run the additive-dose hold-out test on one dynamic file."""
    X, et, g2, chs = d["X"], d["et"], d["g2"], d["channels"]
    n_ch = len(chs)

    # --- R0 training window: first 50% of the clean-air block -------------
    air_rows = np.where((et == 0) & (g2 == 0))[0]
    air_half = air_rows[:int(len(air_rows) * AIR_SEC_FRAC)]
    R0 = np.nanmedian(X[air_half], axis=0)                 # (16,)

    # --- group rows by (et, g2) config with a fast sort key -----------------
    k0 = np.round(np.clip(et, 0, None) * 3).astype(np.int64)
    k1 = np.round(np.clip(g2, 0, None) * 3).astype(np.int64)
    key = k0 * KEY_MOD + k1
    order = np.argsort(key, kind="stable")
    skey = key[order]
    bounds = np.r_[0, np.flatnonzero(np.diff(skey)) + 1, len(key)]

    resp = {}   # (et, g2) -> (16,) log10(s) with s = |R0 - R| / R0
    for a, z in zip(bounds[:-1], bounds[1:]):
        rows = order[a:z]
        et_t, g_t = et[rows[0]], g2[rows[0]]
        run = rows[int(len(rows) * (1 - TAIL_FRAC)):]
        if len(run) < 2:
            continue
        R = np.nanmedian(X[run], axis=0)
        s = np.abs(R0 - R) / np.clip(R0, 1e-6, None)
        s = np.clip(s, 1e-9, 1e3)
        resp[(float(et_t), float(g_t))] = np.log10(s)

    # --- single-gas lookups and mixtures -----------------------------------
    singles_A = {k: v for k, v in resp.items() if k[0] > 0 and k[1] == 0}
    singles_B = {k: v for k, v in resp.items() if k[0] == 0 and k[1] > 0}
    mixtures = {k: v for k, v in resp.items() if k[0] > 0 and k[1] > 0}

    if not singles_A or not singles_B or not mixtures:
        return {"error": "insufficient config coverage"}

    # --- per-channel hold-out predictions ----------------------------------
    out_channels = {}
    for ch in range(n_ch):
        preds, actuals = [], []
        for (e, g) in sorted(mixtures):
            if (e, 0) not in singles_A or (0, g) not in singles_B:
                continue
            s_A = singles_A[(e, 0)][ch]      # log10(s)
            s_B = singles_B[(0, g)][ch]
            s_pred_lin = (1 + 10 ** s_A) * (1 + 10 ** s_B) - 1
            preds.append(float(np.log10(np.clip(s_pred_lin, 1e-9, None))))
            actuals.append(float(mixtures[(e, g)][ch]))
        if len(preds) < 3:
            out_channels[chs[ch]] = {"n_configs": len(preds), "status": "insufficient"}
            continue
        preds, actuals = np.array(preds), np.array(actuals)
        bias = float(preds.mean() - actuals.mean())
        # raw R² includes the (constant) competitive-masking bias; reporting it
        # alone is misleading -> also give the bias-removed "shape" R² and the
        # suppression factor observed/predicted (10^-bias).
        ss_res = ((preds - actuals) ** 2).sum()
        ss_tot = ((actuals - actuals.mean()) ** 2).sum()
        p_c = preds - preds.mean()
        a_c = actuals - actuals.mean()
        ss_res_c = ((p_c - a_c) ** 2).sum()
        ss_tot_c = ((a_c) ** 2).sum()
        out_channels[chs[ch]] = {
            "hold_out_R2_raw": round(float(1 - ss_res / max(ss_tot, 1e-12)), 4),
            "hold_out_R2_shape": round(
                float(1 - ss_res_c / max(ss_tot_c, 1e-12)), 4),
            "bias": round(bias, 4),
            "suppression_ratio_obs_over_pred": round(float(10 ** (-bias)), 4),
            "MAE": round(float(np.abs(preds - actuals).mean()), 4),
            "n_configs": len(preds),
        }

    # --- stacked-identical-TGS bootstrap (ch 0,1,8,9 = TGS2602) -----------
    idxs = [i for i, c in enumerate(chs) if c == "TGS2602"]
    r2vals = [out_channels.get(chs[i], {}).get("hold_out_R2_shape")
              for i in idxs]
    r2vals = [v for v in r2vals if v is not None]
    supp = [out_channels.get(chs[i], {}).get("suppression_ratio_obs_over_pred")
            for i in idxs]
    supp = [v for v in supp if v is not None]
    device_boot = {
        "n_devices": len(r2vals),
        "device_shape_R2": [round(float(v), 4) for v in r2vals],
        "mean_shape_R2": round(float(np.mean(r2vals)), 4) if r2vals else None,
        "sd_shape_R2": round(float(np.std(r2vals)), 4) if r2vals else None,
        "device_suppression": [round(float(v), 4) for v in supp],
        "mean_suppression": round(float(np.mean(supp)), 4) if supp else None,
    }

    return {
        "n_configs_total": len(resp),
        "n_single_A": len(singles_A),
        "n_single_B": len(singles_B),
        "n_mixtures": len(mixtures),
        "channels": out_channels,
        "TGS2602_device_bootstrap": device_boot,
        "note": (
            "positive bias = additive-dose prediction sits ABOVE observed "
            "mixture amplitude (competitive Langmuir masking dominates); "
            "R2 in log10((R0-R)/R0) space."),
    }


def run() -> dict:
    data = _load_raw()
    return {
        "what": (
            "Hold-out R² of the additive-dose mixture law s_AB = "
            "(1+sA)(1+sB)-1 on the two UCI dynamic-mixture streams. "
            "Single-gas amplitudes from the same stream predict held-out "
            "mixture amplitudes. TGS2602 replicate channels (4 devices) "
            "bound device-identity scatter."),
        "files": {name: _run_one_file(name, d) for name, d in data.items()},
    }


def main() -> None:
    res = run()
    record_benchmark("bench_mixture_generalization", res)
    print("Wrote reports/bench_mixture_generalization.json")
    for fname, fr in res["files"].items():
        tgs = fr.get("TGS2602_device_bootstrap", {})
        print(f"\n{fname}: TGS2602 shapeR2={tgs.get('mean_shape_R2')} "
              f"(sd {tgs.get('sd_shape_R2')}) suppr={tgs.get('mean_suppression')} "
              f"n_mix={fr.get('n_mixtures')}")
        for ch, v in list(fr.get("channels", {}).items()):
            if isinstance(v, dict) and "hold_out_R2_raw" in v:
                print(f"  {ch}: R2raw={v['hold_out_R2_raw']} "
                      f"R2shape={v['hold_out_R2_shape']} "
                      f"suppr={v['suppression_ratio_obs_over_pred']} "
                      f"n={v['n_configs']}")


if __name__ == "__main__":
    main()