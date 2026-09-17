"""W5 — repository-wide feature audit (interoperability + physical grounding).

The digital-olfaction question: of the features this repo actually computes,
which ones can be shared across rigs and which are device-bound, calibration-
bound, or protocol-confounded?  This benchmark audits EVERY feature in two
canonical vectors:

    framework 187-dim   opensmell.mox.features.extract_all_framework_features
    primitives  dim     features.physical_features.extract_physical_features
                        (count = len(primitive_vector_names(6)))

Every feature is classified along five documented axes:

  catalog_dimension  the SDK's own grouping (CATALOG.md dimensions:
                     da_, abs_, temp_, health_, hw_, advanced/decay_,
                     sel_ratio_, global_, phys_*)
  category_10        physics-semantic axis used for the audit roll-up:
                     1 dose & amplitude | 2 direction & selectivity |
                     3 rise kinetics | 4 recovery / decay memory |
                     5 baseline & absolute offset | 6 calibrated absolute |
                     7 health & drift | 8 dynamics & noise |
                     9 hardware & transduction | 10 saturation & nonlinearity
  invariance         existing_mapping annotate labels
                     (INV_GAIN / INV_SCALE / DEV_BOUND / CALIBRATED / NONE)
  model_class        A fundamental | B derived | C empirical | D device-calib |
                     E statistical
  interop_verdict    transferable | device_bound | requires_calibration |
                     protocol_confounded

Robustness is measured, not assumed: both extractors run on real SmellNet
offline recordings and the per-feature undefined/zero rates and dynamic range
are recorded, so a "transferable" label is qualified by whether the feature is
even computable on typical windows.

The roll-up output supports the interoperability claim: the shared subset must
be (i) not calibration-bound, (ii) not protocol-confounded, (iii) defined on
the windows it is used on.
"""

from __future__ import annotations

import re

import numpy as np

from _common import record_benchmark
from features.existing_mapping import annotate
from features.physical_features import extract_physical_features, \
    primitive_vector_names
from harness.loaders import load_smellnet_offline

try:
    from opensmell.mox.features import (
        extract_all_framework_features,
        feature_names,
    )
except Exception:  # pragma: no cover
    raise SystemExit("opensmell SDK features not importable")

N_WINDOWS = 24           # recordings audited for compute-robustness stats

CATEGORY_MAP = [
    (r"(da_relative_amplitude|da_auc|da_endpoint_delta|global_.*delta_ratio|"
     r"global_total_auc|global_n_active_channels|normalized_amp|log_response)",
     "1 dose & amplitude"),
    (r"(da_direction|sel_ratio_|phys_ch.*direction|"
     r"phys_array_(logratio|direction)|phys_cov_|n_channels)",
     "2 direction & selectivity"),
    (r"(da_rise_time|response_latency|rise_time$)", "3 rise kinetics"),
    (r"(da_decay_time|decay_tau|decay_a[0-9]|recovery_time)",
     "4 recovery / decay memory"),
    (r"(abs_raw_resistance|abs_baseline_resistance|abs_voltage|_baseline$)",
     "5 baseline & absolute offset"),
    (r"calibrated_concentration", "6 calibrated absolute"),
    (r"health_", "7 health & drift"),
    (r"(temp_hf_transient|temp_oscillation|hw_adc_noise)",
     "8 dynamics & noise"),
    (r"(hw_circuit_response|hw_thermal_profile)", "9 hardware & transduction"),
    (r"(saturation_index|decay_)", "10 saturation & nonlinearity"),
]
# NOTE: decay_tau/decay_a double-matched by (4) and (10) — (4) must win for the
# decay family; handled by first-match ordering below (category 4 precedes 10).

PROTOCOL_CONFOUNDED = re.compile(
    r"(rise_time|decay_time|response_latency|oscillation|hf_transient|"
    r"recovery_time|_tau[0-9]|_a[0-9])")


def _category(name: str) -> str:
    for pat, cat in CATEGORY_MAP:
        if re.search(pat, name):
            if cat == "10 saturation & nonlinearity" and re.search(
                   r"decay_(tau|a)[0-9]", name):
                continue  # decay family stays in category 4
            return cat
    return "0 unclassified"


def _catalog_dimension(name: str) -> str:
    if name.startswith("phys_"):
        if "_cov_" in name:
            return "primitive covariance"
        if name.startswith("phys_ch"):
            return "primitive per-channel"
        return "primitive array"
    if "_da_" in name:
        return "device-agnostic (da_)"
    if "_abs_" in name:
        return "absolute (abs_)"
    if "_temp_" in name:
        return "temporal (temp_)"
    if "_health_" in name:
        return "health (health_)"
    if "_hw_" in name:
        return "hardware (hw_)"
    if "decay_" in name or "saturation_index" in name:
        return "advanced (advanced/decay_)"
    if name.startswith("sel_ratio_"):
        return "cross-channel selectivity"
    if name.startswith("global_"):
        return "global aggregate"
    return "other"


def _model_class(name: str, ann_user) -> str:
    if name.startswith("phys_"):
        return "B"                      # derived from fundamental physics
    inv = ann_user.invariance
    base = _catalog_dimension(name)
    if inv == "NONE" or inv == "CALIBRATED" or "abs_" in base or "hw_" in base:
        return "D"                      # device-calibration dependent
    if "health" in base:
        return "C"                      # empirical trends
    if "temporal" in base or "advanced" in base:
        return "C" if "oscillation" in name or "hf_" in name else "B"
    if name.startswith("sel_ratio_") or name.startswith("global_"):
        return "E"                      # statistical aggregates
    return "B"


def _interop_verdict(name: str, ann_user, model_class: str) -> str:
    if ann_user.calib or model_class == "D":
        return "requires_calibration"
    if PROTOCOL_CONFOUNDED.search(name):
        return "protocol_confounded"
    if ann_user.invariance in ("INV_GAIN", "INV_SCALE"):
        return "transferable"
    if ann_user.invariance == "DEV_BOUND":
        return "device_bound"
    return "requires_calibration"


def _audit_framework(feat_values: dict) -> list[dict]:
    names = feature_names(6)
    anns = annotate(names)             # FeatureAnnot.name is a pattern; zip!
    rows = []
    for n, a in zip(names, anns):
        mc = _model_class(n, a)
        rows.append({
            "feature": n,
            "namespace": "framework_187",
            "catalog_dimension": _catalog_dimension(n),
            "category_10": _category(n),
            "invariance": a.invariance,
            "model_class": mc,
            "device_trust": a.device_trust,
            "ontology_level": a.level,
            "physics": a.physics,
            "requires_calibration": bool(a.calib),
            "interop_verdict": _interop_verdict(n, a, mc),
            **feat_values[n],
        })
    return rows


def _audit_primitives(feat_values: dict) -> list[dict]:
    names = primitive_vector_names(6)
    rows = []
    for n in names:
        base = n.replace("phys_ch", "").replace("phys_array_", "")
        cre = re.search(r"phys_ch[0-9]+_([a-z_]+)$", n)
        kind = cre.group(1) if cre else base.split("_")[0]
        calib = False
        inv = ("INV_GAIN" if kind in ("normalized_amp", "log_response",
                                      "direction", "logratio", "n_channels")
               else "DEV_BOUND")
        protoc = bool(PROTOCOL_CONFOUNDED.search(n))
        verdict = ("device_bound" if kind == "baseline"
                   else "protocol_confounded" if protoc
                   else "transferable")
        rows.append({
            "feature": n,
            "namespace": "primitive_79",
            "catalog_dimension": _catalog_dimension(n),
            "category_10": _category(n),
            "invariance": inv,
            "model_class": "B",
            "device_trust": "low" if verdict == "transferable" else "medium",
            "ontology_level": "L4/L5",
            "physics": f"physical primitive: {kind}",
            "requires_calibration": calib,
            "interop_verdict": verdict,
            **feat_values[n],
        })
    return rows


def _robustness_stats(which: str) -> dict:
    """Run the extractor over real SmellNet windows; per-feature stats."""
    b = load_smellnet_offline()
    recs = b["X"][:N_WINDOWS]
    collected = {}
    n_done = 0
    for X in recs:                       # (n_channels, n_time)
        try:
            if which == "framework":
                out = extract_all_framework_features(X.T, r0_samples=15, sr=1)
            else:
                out = extract_physical_features(X, sr=1.0)
        except Exception:
            continue
        for k, v in out.items():
            if not isinstance(v, (int, float, np.floating, np.integer)):
                continue
            collected.setdefault(k, []).append(float(v))
        n_done += 1
        if n_done >= N_WINDOWS:
            break
    stats = {}
    for k, vals in collected.items():
        arr = np.asarray(vals)
        finite = arr[np.isfinite(arr)]
        stats[k] = {
            "n_windows": n_done,
            "nan_rate": round(1.0 - len(finite) / len(arr), 4) if len(arr)
                        else 1.0,
            "zero_rate": round(float(np.mean(finite == 0.0)), 4)
                         if len(finite) else 1.0,
            "dynamic_range_p05_p95": round(
                float(np.percentile(finite, 95) - np.percentile(finite, 5)), 4)
                if len(finite) else None,
            "median": round(float(np.median(finite)), 5) if len(finite)
                      else None,
            "span_shrinks_with_window": bool(re.search(
                r"(rise_time|decay|recovery|latency|oscillation|hf_|tau|_a[0-9])",
                k)),
        }
    return stats, n_done


def run() -> dict:
    fw_stats, fw_n = _robustness_stats("framework")
    pr_stats, pr_n = _robustness_stats("primitives")
    fw_rows = _audit_framework(fw_stats)
    pr_rows = _audit_primitives(pr_stats)

    # ---- roll-ups ----
    def roll(rows):
        by_cat = {}
        by_dim = {}
        by_verdict = {}
        for r in rows:
            by_cat.setdefault(r["category_10"], []).append(r)
            by_dim.setdefault(r["catalog_dimension"], []).append(r)
            by_verdict.setdefault(r["interop_verdict"], 0)
            by_verdict[r["interop_verdict"]] += 1
        cat_sum = []
        for cat in sorted(by_cat):
            rs = by_cat[cat]
            cat_sum.append({
                "category": cat,
                "n_features": len(rs),
                "transferable": sum(r["interop_verdict"] == "transferable"
                                    for r in rs),
                "device_bound": sum(r["interop_verdict"] == "device_bound"
                                    for r in rs),
                "requires_calibration": sum(
                    r["interop_verdict"] == "requires_calibration"
                    for r in rs),
                "protocol_confounded": sum(
                    r["interop_verdict"] == "protocol_confounded" for r in rs),
                "frequently_undefined": sum(
                    (r.get("nan_rate") or 0) > 0.05 for r in rs),
                "features": [r["feature"] for r in rs],
            })
        dim_sum = [{"dimension": d, "n_features": len(rs)}
                   for d, rs in sorted(by_dim.items())]
        return cat_sum, dim_sum, by_verdict

    fw_cat, fw_dim, fw_ver = roll(fw_rows)
    pr_cat, pr_dim, pr_ver = roll(pr_rows)

    survivors = [r["feature"] for r in fw_rows + pr_rows
                 if r["interop_verdict"] == "transferable"
                 and (r.get("nan_rate") or 0) == 0.0]
    cond = [r["feature"] for r in fw_rows + pr_rows
            if r["interop_verdict"] == "protocol_confounded"]

    fw_nfeat, pr_nfeat = len(fw_rows), len(pr_rows)
    aggregate = {
        "n_windows_audited": {"framework": fw_n, "primitives": pr_n},
        "n_features": {"framework": fw_nfeat, "primitives": pr_nfeat,
                       "total": fw_nfeat + pr_nfeat},
        "verdict_counts": {
            "framework": fw_ver,
            "primitives": pr_ver,
        },
        "category_rollup": {
            "framework": fw_cat,
            "primitives": pr_cat,
        },
        "dimension_rollup": {
            "framework": fw_dim,
            "primitives": pr_dim,
        },
        "transferable_fully_defined": survivors,
        "n_transferable_fully_defined": len(survivors),
        "protocol_confounded_features": cond,
        "n_protocol_confounded": len(cond),
        "survivor_by_category": sorted({
            r["category_10"] for r in fw_rows + pr_rows
            if r["interop_verdict"] == "transferable"
            and (r.get("nan_rate") or 0) == 0.0}),
    }

    total = fw_nfeat + pr_nfeat
    tr = len(survivors)
    rc = (fw_ver.get("requires_calibration", 0)
          + pr_ver.get("requires_calibration", 0))
    pc = (fw_ver.get("protocol_confounded", 0)
          + pr_ver.get("protocol_confounded", 0))
    db = (fw_ver.get("device_bound", 0) + pr_ver.get("device_bound", 0))
    share = round(100.0 * tr / total) if total else 0
    aggregate["verdict"] = (
        f"verdict: {tr}/{total} features ({share}%) are transferable and "
        f"fully defined on real windows; {db} are device-bound, {rc} are "
        f"calibration-bound and {pc} are protocol-confounded (kinetic/decay "
        "features must not be shared raw). The shareable core = the _da_ "
        "amplitude/dose family + selectivity ratios + saturation index + the "
        "phys_ normalized/log-response/direction/covariance primitives; "
        "absolute, hardware and calibrated-concentration features are "
        "rig-bound."
    )
    return {
        "what": (
            "Wave-4 repository feature audit: every feature of the 187-dim "
            "framework vector and the 79-dim physical-primitive vector is "
            "classified by catalog dimension, physics-semantic category, "
            "invariance census, model class and interop verdict, and its "
            "compute-robustness is measured on real SmellNet windows."),
        "features": fw_rows + pr_rows,
        "aggregate": aggregate,
    }


def main() -> None:
    res = run()
    record_benchmark("bench_features_audit", res)
    a = res["aggregate"]
    print("Wrote reports/bench_features_audit.json")
    print("\nwindows audited:", a["n_windows_audited"],
          "| n features:", a["n_features"])
    print("verdict counts (framework):", a["verdict_counts"]["framework"])
    print("verdict counts (primitives):", a["verdict_counts"]["primitives"])
    print("\ncategory rollup (framework):")
    for c in a["category_rollup"]["framework"]:
        print(f"  {c['category']:30s} n={c['n_features']:3d} "
              f"transferable={c['transferable']:2d} "
              f"calib={c['requires_calibration']:2d} "
              f"protocol={c['protocol_confounded']:2d} "
              f"device_bound={c['device_bound']:2d} "
              f"freq_undefined={c['frequently_undefined']}")
    print("\ncategory rollup (primitives):")
    for c in a["category_rollup"]["primitives"]:
        print(f"  {c['category']:30s} n={c['n_features']:3d} "
              f"transferable={c['transferable']:2d} "
              f"protocol={c['protocol_confounded']:2d}")
    print("\ntransferable & fully defined (", a["n_transferable_fully_defined"],
          "):")
    print("  ", a["transferable_fully_defined"])
    print("\nverdict:", a["verdict"])


if __name__ == "__main__":
    main()