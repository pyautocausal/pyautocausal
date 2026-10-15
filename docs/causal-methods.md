# Methods and limitations

PyAutoCausal selects execution paths from observable data structure. That routing does not determine whether a causal interpretation is justified. Outcome definition, covariate timing, treatment assignment, overlap, anticipation and interference remain research-design decisions.

## Available entry points

| Analysis | Specification or entry point | Fitting function | Main returned result |
| --- | --- | --- | --- |
| Cross-sectional regression | `create_cross_sectional_specification` | `fit_ols` | Specification with Statsmodels results in `.model` |
| DiD / event study | `create_did_specification` / `create_event_study_specification` | `fit_did_panel` / `fit_panel_ols` | Specification with fitted `.model` |
| Staggered DiD | `create_staggered_did_specification` | `fit_callaway_santanna_estimator` or `fit_callaway_santanna_nyt_estimator` | Specification with group-time and aggregate effects in `.model` |
| Synthetic DiD | `create_synthdid_specification` | `fit_synthdid_estimator` | Specification with estimate and inference in `.model` |
| Hainmueller synthetic control | `create_synthdid_specification` | `fit_hainmueller_synth_estimator` | Specification with fitted `.model` |
| Experimental binary uplift | `create_uplift_specification` | `fit_s_learner`, `fit_t_learner`, `fit_x_learner`, `fit_double_ml_binary` | Specification with effect estimates and inference status |
| Optional partially linear DoubleML | DataFrame and explicit `covariates` | `double_ml_estimation` | Text summary |

Specification constructors are in `pyautocausal.pipelines.library.specifications`; the fitting functions are in `pyautocausal.pipelines.library.estimators`, except `double_ml_estimation`, which is in `pyautocausal.causal_methods.double_ml`. The fitted `.model` has different types across estimators: for example, `fit_ols(spec).model.params` holds coefficients, while the C&S wrappers store their outputs in a `.model` dictionary.

The panel factory routes a single treated unit to synthetic methods; multiple treated units with fewer than two observed post-adoption relative periods go to DiD. With at least two post-adoption relative periods, simultaneous adoption goes to an event study and staggered adoption goes to C&S. The C&S route selects never-treated comparisons when at least 10% of units are never treated, and not-yet-treated comparisons otherwise. Inspect the graph's completed nodes and recorded comparison group. This routing threshold is a software default, not a statistical sufficiency guarantee. Uplift and optional DoubleML are direct APIs, not automatic branches of the cross-sectional factory.

## Cross-sectional and panel regression

Cross-sectional specifications preserve treatment coding: zero is control and one is treated. Other labels require an explicit `treatment_value`. `control_cols=None` infers numeric controls without changing their dtypes; `control_cols=[]` requests no controls. Identifier and period columns, plus conventional `post`, `relative_time` and `treatment_time` metadata, are excluded from automatic inference. Use `unit_col` and `time_col` to identify custom structural columns in cross-sectional specifications. Unsupported categorical candidates require deliberate encoding or explicit control selection; see [data requirements](data-requirements.md#control-selection). Do not include outcomes measured after treatment as adjustment covariates without a defensible identification argument.

OLS coefficients require an appropriate functional form and identifying assumptions for causal interpretation. Panel fixed effects do not automatically eliminate time-varying confounding. Conventional two-way fixed effects can be misleading under heterogeneous staggered treatment effects; inspect the selected estimator and comparison group.

## Callaway–Sant’Anna

The never-treated and not-yet-treated wrappers use their respective comparison group and the selected covariate formula. C&S controls must be finite numeric columns with identifier names; encode categorical variables before calling these wrappers. Selected controls are preserved even when their names match generated cohort or event columns. Identification requires the corresponding parallel-trends and no-anticipation assumptions. Report the actual controls and aggregation target. A no-control model is not equivalent to a covariate-adjusted design.

Both wrappers fit panel group-time effects with a doubly robust estimator, zero anticipation and a varying base period. The wrappers fix the bootstrap iteration count at 1,000 and the confidence level at 95%. They return `att_gt_object`, `overall_effect`, `dynamic_effects` and `group_effects` in `.model`, together with `control_group`, `random_state` and `bootstrap_iterations`. The wrappers permit unbalanced panels; comparison support must still exist for the contrasts being estimated.

The wrappers use `random_state=42` by default for reproducible multiplier-bootstrap inference and record the seed and iteration count in exported provenance. Supply another integer seed to assess simulation variation, or `None` for a nondeterministic run. Fitting does not alter global NumPy random state.

Degenerate or non-finite influence-function/bootstrap dimensions retain their point estimates but have unavailable uncertainty. Their standard errors and interval endpoints are `NaN`, not zero-width confidence intervals. The fitted ATT object and aggregation results expose an `inference` status (`asymptotic`, `partial` or `unavailable`), with a reason when uncertainty is missing. Printed summaries identify unavailable inference and do not attach significance markers to those estimates. Inspect both the group-time and requested aggregation results; availability can differ across them.

The bundled implementation is adapted from the Python C&S implementation. Regression tests compare wrapper behavior to direct calls with the same settings and exercise known-effect fixtures. They do not establish validity for arbitrary datasets.

## Synthetic DiD

`T0` is the **number of pretreatment columns**, not the index of the final untreated column. Input requires a complete unit/time grid, untreated donor units and a supported simultaneous-adoption pattern. Treatment reversals, missing cells and insufficient information for the chosen uncertainty method must be rejected explicitly.

The point estimate and uncertainty methods have different sample-size requirements. A point estimate may exist when a particular bootstrap, jackknife or placebo variance is unavailable. Report unavailable inference explicitly; do not fabricate error bars or heterogeneous effect curves. See the [reference implementation](https://github.com/synth-inference/synthdid) for the estimator and inference algorithms.

`fit_synthdid_estimator` stores the point estimate in `.model` and placebo uncertainty in `.model.inference`, using seed 42. Its placebo calculation requires more donor units than treated units. Otherwise the estimate is retained with `status='unavailable'`, `se=None`, `ci=None` and a reason. The lower-level variance functions in `pyautocausal.pipelines.library.synthdid.vcov` expose bootstrap, jackknife and placebo methods separately. A direct synthetic-DiD specification can support multiple simultaneously treated units even though the automatic factory chooses its synthetic branch for one treated unit.

## Hainmueller synthetic control

This branch requires exactly one treated unit and the complete panel supplied by the synthetic-DiD specification. The wrapper passes only the selected controls and required design columns to the backend; pretreatment outcomes are also used as predictors. Extra columns left in the original frame do not implicitly enter this model.

The wrapper sorts the backend input by unit and time before reshaping outcomes and donor data. Shuffling input rows therefore does not change which outcomes belong to each unit. Direct numeric period values, including fractional periods, retain their treatment boundary instead of being truncated to integers. Factory pipelines first map observed periods to chronological indices and retain the original mapping in cleaning metadata.

Placebo diagnostics fit independent copies of the model. `fit_hainmueller_placebo_test` defaults to `random_state=42` and selects an observed pretreatment period for the in-time placebo, recording the seed and chosen period in `hainmueller_placebo_metadata`. It requires at least two observed pretreatment periods. Placebo comparisons are diagnostics under the chosen design; they do not establish parallel trends or guarantee a causal interpretation.

## Experimental uplift APIs

S-, T- and X-learners fit binary-outcome conditional effect models. Class balancing an outcome classifier changes its probability scale; raw class-balanced probabilities cannot be subtracted as calibrated outcome means. Homogeneous predicted effects are legitimate and must not be perturbed to look heterogeneous.

S/T/X methods return `cate_estimates` and their sample average in `ate_estimate`. They do not expose validated confidence intervals for their average effects in this release: `ate_ci` is `None` and `inference['status']` is `unavailable`. Returned point estimates and CATE predictions are experimental. The fixed learner seed is 42; these wrapper signatures do not expose a seed parameter. Features must be finite numeric values, and treatment and outcome must be coded 0/1. An empty control list is supported as an unadjusted model.

Despite its name, `fit_double_ml_binary` is the package's experimental cross-fitted AIPW implementation and does not require the optional `doubleml` dependency. It estimates the ATE and returns a normal 95% interval under its stated assumptions. It uses five stratified folds, requires at least five observations per treatment arm, rejects repeated unit IDs, and rejects fitted propensities outside the open interval `(0.01, 0.99)`. Its inference assumes independent units, unconfoundedness, overlap and sufficiently accurate nuisance fits. Clustered/panel uncertainty is unsupported. `cate_estimates` is `None`: an influence-function score is not an individual treatment effect.

## Optional DoubleML partially linear model

Install the `doubleml` extra using the [current installation instructions](../README.md#installation), then call `pyautocausal.causal_methods.double_ml.double_ml_estimation`. This wrapper uses the library's `DoubleMLData`, cross-fitting and `confint()` APIs. It supports numeric finite data, varying treatment and at least one covariate. Its estimand is a partially linear treatment coefficient, not a CATE. Lasso scaling is fitted inside training folds. Seeds control learners and sample splits without changing global NumPy state.

Pass `covariates` explicitly. Unlike the specification constructors, this wrapper's default includes every column other than the named outcome and treatment, including numeric identifiers and time columns. It accepts `ml_learner='random_forest'` (default) or `'lasso'`, `n_folds` (default 5), `n_rep` (default 1) and `random_state` (default 42). It returns a formatted summary string containing the coefficient, standard error, confidence interval and fitting settings; it does not return the DoubleML model object.

See the [official DoubleML API](https://docs.doubleml.org/stable/api/generated/doubleml.plm.DoubleMLPLR.html) for model assumptions and inference details.

## Validation scope

Fast tests check exact examples, invariance, forwarding and invalid-input behavior. Longer scientific benchmarks use predetermined seeds, designs and tolerances. Passing these checks is evidence for the tested designs, not a general proof of estimator validity. A corrected version can change previous results; review the [release notes](../CHANGELOG.md) and rerun affected analyses.

The [scientific benchmark suite](../tests/scientific/test_estimator_benchmarks.py) specifies the statistical support currently checked:

| Benchmark | Design and target | Prespecified acceptance |
| --- | --- | --- |
| Binary AIPW | 40 randomized samples of 1,600 independent units; true ATE 0.3 | Mean estimate within 0.025; nominal 95% interval coverage at least 80% |
| S/T/X point estimates | Balanced binary cells; conditional effects 0.1 and 0.5, average 0.3 | Each target recovered within 0.04; no interval claimed |
| Synthetic-DiD inference | 30 panels with 40 donors, 20 treated units, 10 pre and 5 post periods, serially correlated errors and effect 2 | Mean estimate within 0.1; coverage at least 80% for each bootstrap, jackknife and placebo interval |

The coverage tolerance is a coarse check for severe undercoverage; passing it does not establish exact 95% calibration. C&S, OLS, Hainmueller and optional DoubleML have functional or reference-behavior checks, but are not covered by these repeated-simulation interval benchmarks. Preserve the input snapshot, package version, selected controls, comparison group, sample audit and inference settings when reporting or reproducing an analysis.
