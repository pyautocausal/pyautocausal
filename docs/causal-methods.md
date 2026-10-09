# Methods and limitations

PyAutoCausal selects execution paths from observable data structure. That routing does not determine whether a causal interpretation is justified. Outcome definition, covariate timing, treatment assignment, overlap, anticipation and interference remain research-design decisions.

## Cross-sectional and panel regression

Cross-sectional specifications preserve treatment coding: zero is control and one is treated. Other labels require an explicit `treatment_value`. `control_cols=None` infers eligible controls; `control_cols=[]` requests no controls. Do not include outcomes measured after treatment as adjustment covariates without a defensible identification argument.

OLS coefficients require an appropriate functional form and identifying assumptions for causal interpretation. Panel fixed effects do not automatically eliminate time-varying confounding. Conventional two-way fixed effects can be misleading under heterogeneous staggered treatment effects; inspect the selected estimator and comparison group.

## Callaway–Sant’Anna

Both the never-treated and not-yet-treated wrappers must use the requested comparison group and covariate formula. Identification requires the corresponding parallel-trends and no-anticipation assumptions. Report the actual controls and aggregation target. A no-control model is not equivalent to a covariate-adjusted design.

The wrappers use `random_state=42` by default for reproducible multiplier-bootstrap inference and record the seed and iteration count in exported provenance. Supply another integer seed to assess simulation variation, or `None` for a nondeterministic run. Fitting does not alter global NumPy random state.

The bundled implementation is adapted from the Python C&S implementation. The scientific tests compare wrapper behavior to direct calls with the same settings and exercise known-effect fixtures. They do not establish validity for arbitrary datasets.

## Synthetic DiD

`T0` is the **number of pretreatment columns**, not the index of the final untreated column. Input requires a complete unit/time grid, untreated donor units and a supported simultaneous-adoption pattern. Treatment reversals, missing cells and insufficient information for the chosen uncertainty method must be rejected explicitly.

The point estimate and uncertainty methods have different sample-size requirements. A point estimate may exist when a particular bootstrap, jackknife or placebo variance is unavailable. Report unavailable inference explicitly; do not fabricate error bars or heterogeneous effect curves. See the [reference implementation](https://github.com/synth-inference/synthdid) for the estimator and inference algorithms.

## Experimental uplift APIs

S-, T- and X-learners fit binary-outcome conditional effect models. Class balancing an outcome classifier changes its probability scale; raw class-balanced probabilities cannot be subtracted as calibrated outcome means. Homogeneous predicted effects are legitimate and must not be perturbed to look heterogeneous.

S/T/X methods do not expose validated confidence intervals for their average effects in this release. Returned point estimates and CATE predictions are experimental. The separately implemented binary double-robust method must distinguish its average effect from individual conditional effects; an influence-function score is not an individual treatment effect. Consult result metadata for the uncertainty method and availability.

## Optional DoubleML partially linear model

Install `pyautocausal[doubleml]` and call `pyautocausal.causal_methods.double_ml.double_ml_estimation`. This wrapper uses the library’s `DoubleMLData`, cross-fitting and `confint()` APIs. It supports numeric finite data, varying treatment and at least one covariate. Its estimand is a partially linear treatment coefficient, not a CATE. Lasso scaling is fitted inside training folds. Seeds control learners and sample splits without changing global NumPy state.

See the [official DoubleML API](https://docs.doubleml.org/stable/api/generated/doubleml.plm.DoubleMLPLR.html) for model assumptions and inference details.

## Validation scope

Fast tests check exact examples, invariance, forwarding and invalid-input behavior. Longer scientific benchmarks use predetermined seeds, designs and tolerances. Passing these checks is evidence for the tested designs, not a general proof of estimator validity. A corrected version can change previous results; review the [release notes](../CHANGELOG.md) and rerun affected analyses.
