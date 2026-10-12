# Methods and limitations

PyAutoCausal selects execution paths from observable data structure. That routing does not determine whether a causal interpretation is justified. Outcome definition, covariate timing, treatment assignment, overlap, anticipation and interference remain research-design decisions.

## Cross-sectional and panel regression

Cross-sectional specifications preserve treatment coding: zero is control and one is treated. Other labels require an explicit `treatment_value`. `control_cols=None` infers numeric controls without changing their dtypes; `control_cols=[]` requests no controls. Identifier and period columns, plus conventional `post`, `relative_time` and `treatment_time` metadata, are excluded from automatic inference. Use `unit_col` and `time_col` to identify custom structural columns in cross-sectional specifications. Unsupported categorical candidates require deliberate encoding or explicit control selection; see [data requirements](data-requirements.md#control-selection). Do not include outcomes measured after treatment as adjustment covariates without a defensible identification argument.

OLS coefficients require an appropriate functional form and identifying assumptions for causal interpretation. Panel fixed effects do not automatically eliminate time-varying confounding. Conventional two-way fixed effects can be misleading under heterogeneous staggered treatment effects; inspect the selected estimator and comparison group.

## Callaway–Sant’Anna

The never-treated and not-yet-treated wrappers use their respective comparison group and the selected covariate formula. C&S controls must be finite numeric columns with identifier names; encode categorical variables before calling these wrappers. Selected controls are preserved even when their names match generated cohort or event columns. Identification requires the corresponding parallel-trends and no-anticipation assumptions. Report the actual controls and aggregation target. A no-control model is not equivalent to a covariate-adjusted design.

The wrappers use `random_state=42` by default for reproducible multiplier-bootstrap inference and record the seed and iteration count in exported provenance. Supply another integer seed to assess simulation variation, or `None` for a nondeterministic run. Fitting does not alter global NumPy random state.

Degenerate or non-finite influence-function/bootstrap dimensions retain their point estimates but have unavailable uncertainty. Their standard errors and interval endpoints are `NaN`, not zero-width confidence intervals. The fitted ATT object and aggregation results expose an `inference` status (`asymptotic`, `partial` or `unavailable`), with a reason when uncertainty is missing. Printed summaries identify unavailable inference and do not attach significance markers to those estimates. Inspect both the group-time and requested aggregation results; availability can differ across them.

The bundled implementation is adapted from the Python C&S implementation. The scientific tests compare wrapper behavior to direct calls with the same settings and exercise known-effect fixtures. They do not establish validity for arbitrary datasets.

## Synthetic DiD

`T0` is the **number of pretreatment columns**, not the index of the final untreated column. Input requires a complete unit/time grid, untreated donor units and a supported simultaneous-adoption pattern. Treatment reversals, missing cells and insufficient information for the chosen uncertainty method must be rejected explicitly.

The point estimate and uncertainty methods have different sample-size requirements. A point estimate may exist when a particular bootstrap, jackknife or placebo variance is unavailable. Report unavailable inference explicitly; do not fabricate error bars or heterogeneous effect curves. See the [reference implementation](https://github.com/synth-inference/synthdid) for the estimator and inference algorithms.


## Hainmueller synthetic control

This branch requires exactly one treated unit and the complete panel supplied by the synthetic-DiD specification. The wrapper passes only the selected controls and required design columns to the backend; pretreatment outcomes are also used as predictors. Extra columns left in the original frame do not implicitly enter this model.

The wrapper sorts the backend input by unit and time before reshaping outcomes and donor data. Shuffling input rows therefore does not change which outcomes belong to each unit. Direct numeric period values, including fractional periods, retain their treatment boundary instead of being truncated to integers. Factory pipelines first map observed periods to chronological indices and retain the original mapping in cleaning metadata.

Placebo diagnostics fit independent copies of the model. `fit_hainmueller_placebo_test` defaults to `random_state=42` and selects an observed pretreatment period for the in-time placebo, recording the seed and chosen period in `hainmueller_placebo_metadata`. It requires at least two observed pretreatment periods. Placebo comparisons are diagnostics under the chosen design; they do not establish parallel trends or guarantee a causal interpretation.

## Experimental uplift APIs

S-, T- and X-learners fit binary-outcome conditional effect models. Class balancing an outcome classifier changes its probability scale; raw class-balanced probabilities cannot be subtracted as calibrated outcome means. Homogeneous predicted effects are legitimate and must not be perturbed to look heterogeneous.

S/T/X methods do not expose validated confidence intervals for their average effects in this release. Returned point estimates and CATE predictions are experimental. The separately implemented binary double-robust method must distinguish its average effect from individual conditional effects; an influence-function score is not an individual treatment effect. Consult result metadata for the uncertainty method and availability.

## Optional DoubleML partially linear model

Install `pyautocausal[doubleml]` and call `pyautocausal.causal_methods.double_ml.double_ml_estimation`. This wrapper uses the library’s `DoubleMLData`, cross-fitting and `confint()` APIs. It supports numeric finite data, varying treatment and at least one covariate. Its estimand is a partially linear treatment coefficient, not a CATE. Lasso scaling is fitted inside training folds. Seeds control learners and sample splits without changing global NumPy state.

See the [official DoubleML API](https://docs.doubleml.org/stable/api/generated/doubleml.plm.DoubleMLPLR.html) for model assumptions and inference details.

## Validation scope

Fast tests check exact examples, invariance, forwarding and invalid-input behavior. Longer scientific benchmarks use predetermined seeds, designs and tolerances. Passing these checks is evidence for the tested designs, not a general proof of estimator validity. A corrected version can change previous results; review the [release notes](../CHANGELOG.md) and rerun affected analyses.
