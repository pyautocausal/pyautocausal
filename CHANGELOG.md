# Changelog

## 0.1.2 — 2026-10-12 correctness release

Correction notes, the wheel and source archive are available in the [GitHub release](https://github.com/pyautocausal/pyautocausal/releases/tag/v0.1.2). GitHub and PyPI publication are separate; use the installation instructions in the [README](README.md#installation) to obtain this corrected version.

Re-run affected analyses produced with 0.1.0 or 0.1.1. Both versions were already published before these corrections; key estimator files in the published 0.1.1 source distribution match the reviewed baseline. Existing test success did not rule out the numerical and export errors corrected here. Saved reports are not automatically rewritten.

### Results that can change

- Cross-sectional treatment coding preserves 0/1 meaning instead of reversing it based on the first row. Other encodings require an explicit treated value.
- Default cleaning retains numeric controls, including binary covariates. IDs, time and conventional design metadata are excluded from automatic control selection. Unsupported automatic categorical-control inference fails explicitly, and generated design columns cannot overwrite selected controls.
- Synthetic DiD consistently uses the count of pretreatment periods. Panel validation, weighting, uncertainty calculations and plotted effect paths have been reviewed against the reference implementation. Artificial random plot variation is removed.
- Synthetic-control predictors are scaled across units, and the fitting loss compares matching predictor dimensions. Outcome-only specifications now work; balance diagnostics explicitly report when no adjustment covariates are available.
- Synthetic control sorts the panel before constructing matrices, honors the selected predictor list and retains fractional treatment times. Placebos retain the best feasible fit and raise an explicit error if every optimization attempt fails.
- Callaway–Sant’Anna receives requested covariates and retains the requested comparison-group mode. Explicit empty control lists are respected. Multiplier-bootstrap inference uses a recorded local seed, so repeated runs and notebook exports reproduce the same uncertainty estimates without changing global random state.
- Callaway–Sant’Anna retains point estimates for zero-variance effects and labels unavailable uncertainty in summaries, plots and provenance. It does not report zero-width bands as valid inference.
- Never-treated units no longer receive a spurious event-period-zero treatment dummy.
- Event-study plots distinguish the actual reference period from estimated zero effects, retaining confidence intervals for the latter.
- Uplift models no longer balance away arm-specific probability differences or manufacture heterogeneity. Unsupported meta-learner intervals are explicitly unavailable. Average-effect inference is distinguished from CATE predictions.
- Time cleaning preserves intraday precision and records the original mapping. Cleaning failures propagate; changed samples are audited and revalidated.
- Missingness thresholds use each column's original sample denominator consistently. Direct specification validation detects Boolean and unsigned treatment reversals.

### Execution and output

- Graph merges preserve branch classifications and argument bindings; joins made ready by skipped paths execute deterministically. Conditions are evaluated once. Graph reuse requires an explicit reset.
- Alternative predecessor arguments and branch prerequisites are validated.
- Explicit optional-argument bindings cannot overwrite required run-context inputs. Notebook replay preserves supported scientific data types and rejects ambiguous single-file loading for multiple external inputs.
- Figure persistence saves the requested figure. Scalar/list/NumPy JSON values are handled consistently, with explicit non-finite-value behavior.
- Notebook exports include execution diagnostics and provenance and use the intended interpreter for execution. Experimental or unavailable inference remains visible.
- Compact HTML hides tagged implementation inputs while executing the code and retaining its results. Export failures propagate and record their stage separately from estimation status.
- DataFrame notebook exports use a verified typed snapshot, preventing CSV inference from collapsing distinct identifiers or changing timestamps/categories. Aware datetime replay preserves the original instant and timezone offset.

### Installation and maintenance

- Notebook execution and DoubleML have documented installation extras. The DoubleML wrapper uses proper data objects, fold-local preprocessing, reproducible cross-fitting and the library’s confidence-interval API.
- Dependency bounds exclude scikit-learn versions incompatible with the selected DoubleML API; fresh installed wheels exercise an actual fit.
- The advertised PENN dataset loads correctly.
- CI covers Python 3.10–3.12, base/all-extra installed wheels and separate scientific benchmarks.
- Quickstarts, example data paths, data policies and issue acceptance criteria are documented. Examples no longer fill missing covariates with an arbitrary sentinel.

This release remains alpha software. Passing the included benchmarks supports the tested designs; it does not establish identification or statistical validity for every application. See [methods and limitations](docs/causal-methods.md).
