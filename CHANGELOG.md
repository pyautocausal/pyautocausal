# Changelog

## 0.1.2 — Unreleased correctness release

Re-run affected analyses produced with 0.1.0 or 0.1.1. Both versions were already published before these corrections; key estimator files in the published 0.1.1 source distribution match the reviewed baseline. Existing test success did not rule out the numerical and export errors corrected here. Saved reports are not automatically rewritten.

### Results that can change

- Cross-sectional treatment coding preserves 0/1 meaning instead of reversing it based on the first row. Other encodings require an explicit treated value.
- Synthetic DiD consistently uses the count of pretreatment periods. Panel validation, weighting, uncertainty calculations and plotted effect paths have been reviewed against the reference implementation. Artificial random plot variation is removed.
- Synthetic-control predictors are scaled across units, and the fitting loss compares matching predictor dimensions. Outcome-only specifications now work; balance diagnostics explicitly report when no adjustment covariates are available.
- Callaway–Sant’Anna receives requested covariates and retains the requested comparison-group mode. Explicit empty control lists are respected. Multiplier-bootstrap inference uses a recorded local seed, so repeated runs and notebook exports reproduce the same uncertainty estimates without changing global random state.
- Never-treated units no longer receive a spurious event-period-zero treatment dummy.
- Uplift models no longer balance away arm-specific probability differences or manufacture heterogeneity. Unsupported meta-learner intervals are explicitly unavailable. Average-effect inference is distinguished from CATE predictions.
- Time cleaning preserves intraday precision and records the original mapping. Cleaning failures propagate; changed samples are audited and revalidated.

### Execution and output

- Graph merges preserve branch classifications and argument bindings; joins made ready by skipped paths execute deterministically. Conditions are evaluated once. Graph reuse requires an explicit reset.
- Alternative predecessor arguments and branch prerequisites are validated.
- Figure persistence saves the requested figure. Scalar/list/NumPy JSON values are handled consistently, with explicit non-finite-value behavior.
- Notebook exports include execution diagnostics and provenance and use the intended interpreter for execution. Experimental or unavailable inference remains visible.

### Installation and maintenance

- Notebook execution and DoubleML have documented installation extras. The DoubleML wrapper uses proper data objects, fold-local preprocessing, reproducible cross-fitting and the library’s confidence-interval API.
- Dependency bounds exclude scikit-learn versions incompatible with the selected DoubleML API; fresh installed wheels exercise an actual fit.
- The advertised PENN dataset loads correctly.
- CI covers Python 3.10–3.12, base/all-extra installed wheels and separate scientific benchmarks.
- Quickstarts, example data paths, data policies and issue acceptance criteria are documented. Examples no longer fill missing covariates with an arbitrary sentinel.

This release remains alpha software. Passing the included benchmarks supports the tested designs; it does not establish identification or statistical validity for every application. See [methods and limitations](docs/causal-methods.md).
