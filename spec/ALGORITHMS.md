# ALGORITHMS

## Multi-Exponential Decay Model

The recovery phase of each MOX channel is fit to a sum of exponential decays:

```
Single: y(t) = a1·exp(-t/tau1) + c
Double: y(t) = a1·exp(-t/tau1) + a2·exp(-t/tau2) + c
Triple: y(t) = a1·exp(-t/tau1) + a2·exp(-t/tau2) + a3·exp(-t/tau3) + c
```

The default model is **bi-exponential** (`compute_multi_exp_decay(..., n_components=2)`):
a fast component (tau1 ~1-3 s) for weakly adsorbed surface species and a slow
component (tau2 ~10-30 s) for strongly bound species. The ratio tau2/tau1 is a
measure of surface heterogeneity.

A tri-exponential model is over-parameterized for MOX recovery data: the three
components are not uniquely identifiable, so the fit can alternate between
near-equal-cost local minima. It is therefore opt-in only via
`n_components=3` and is documented as advanced and possibly nondeterministic.

### Reproducibility note

The decay fit uses scipy's MINPACK optimizer (`scipy.optimize.curve_fit`).
On data where several parameter sets produce essentially the same residual
cost, MINPACK may converge to different local minima from one run to the next.
Consequences:

- **Decay tau values may vary slightly across runs** (measured up to ~6%
  relative difference), and occasional tau/amplitude values may flip to the
  `-1.0` "fit failed" sentinel when the optimizer reports non-convergence.
- **Fit quality (residual cost) remains consistent** across runs — the
  differing solutions are equal-cost minima, not worse fits.

This is a known limitation of the optimizer on equal-cost solutions, not a flaw
in the model. All returned solutions are equally valid fits to the recovery
curve.

Guarantees enforced by `tests/test_legacy_api.py::test_determinism`:

1. Every non-decay feature is byte-identical between runs.
2. Decay tau values agree within 10% relative difference whenever both runs
   produced a valid fit.
3. At most 20 features may differ at all (observed range 0-20 across 91
   pairwise runs; decay amplitudes may flip between equal-cost solutions).

For publication-grade work, verify fits manually or fix the optimizer seed.
