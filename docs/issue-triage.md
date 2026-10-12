# Issue triage for the correctness release

Tracking owner: **ytchokni**. These are implementation and review responsibilities, not a claim that changes have already reached main. Close issues only after their acceptance criteria are verified on the merged version.

| Issue | Disposition | Acceptance criteria |
| --- | --- | --- |
| [#43](https://github.com/pyautocausal/pyautocausal/issues/43) | Correctness release | Alternative branch outputs can bind one parameter; ambiguous active inputs fail clearly. |
| [#62](https://github.com/pyautocausal/pyautocausal/issues/62) | Correctness release | Integer and string unit-ID panels yield equivalent estimates through the public pipeline. |
| [#64](https://github.com/pyautocausal/pyautocausal/issues/64) | Correctness release | Missingness threshold and drop/reject policy are configurable, audited and validated after cleaning. |
| [#58](https://github.com/pyautocausal/pyautocausal/issues/58) | Correctness release | Missing branch prerequisites fail before partially constructing the branch; valid composition is tested. |
| [#42](https://github.com/pyautocausal/pyautocausal/issues/42) | Correctness release | Export diagnostics preserve failed and skipped nodes without presenting failure as success. |
| [#38](https://github.com/pyautocausal/pyautocausal/issues/38) | Retriaged: implementation existed; installation/kernel repairs needed | Installed notebook extra executes a generated notebook to HTML using the intended interpreter. |
| [#66](https://github.com/pyautocausal/pyautocausal/issues/66) | Correctness release | README local links resolve and advertised quickstarts execute. |
| [#44](https://github.com/pyautocausal/pyautocausal/issues/44) | Descriptions and provenance | Node descriptions survive composition/export; explicit descriptions can supplement callable docstrings. |
| [#33](https://github.com/pyautocausal/pyautocausal/issues/33) | Concise presentation | An optional compact notebook view hides implementation clutter without removing execution or diagnostic information. |
| [#41](https://github.com/pyautocausal/pyautocausal/issues/41) | Common report context | Method, sample, transformations and inference availability remain visible across result types. |
| [#30](https://github.com/pyautocausal/pyautocausal/issues/30) | Converted to release gates | Known-effect tests, documented inputs, explicit unsupported cases, reproducible output and clean installation pass. |
| [#19](https://github.com/pyautocausal/pyautocausal/issues/19) | Deferred product feature | Interactive CLI nodes remain out of this correctness release; design separately after the release gates pass. |

The thirteen original findings and subsequent corrections from independent reviews are summarized in [the changelog](../CHANGELOG.md). A local repair or draft PR does not by itself resolve the remote issue.

## Verification evidence

The implementation is in [PR #71](https://github.com/pyautocausal/pyautocausal/pull/71). These automated checks exercise the acceptance criteria; final issue closure still requires the changes to reach main.

| Criteria | Regression evidence |
| --- | --- |
| #43: alternative bindings and deterministic graph execution | `tests/orchestration/test_execution_correctness.py`: active alternatives, ambiguous providers, late skips, nested merges, named optional inputs and reset after failure. |
| #62: identifiers remain labels | `tests/integration/test_public_cleaning_policies.py` compares numeric/string panel IDs; `test_input_control_contract.py` verifies equivalent cross-sectional estimates and exclusion of ID/time columns. |
| #64: audited missing-data policy | `tests/data_cleaning/test_cleaning_correctness.py` and public cleaning tests check threshold boundaries, original denominators, rejection, dropped rows and post-cleaning design validation. |
| #58: branch composition | `tests/pipelines/test_branch_prerequisites.py` checks missing prerequisites and valid construction; orchestration regressions check composed results. |
| #42: faithful failure diagnostics | `tests/persistence/test_report_export_contract.py` checks propagated notebook/HTML errors and recorded export stages; execution tests preserve failed/skipped node states. |
| #38: installed notebook execution | `scripts/smoke_installed.py` runs outside the checkout; notebook regressions verify the current interpreter, replayed values and numerical estimates. |
| #66: README accuracy | `tests/test_documentation.py` executes the causal quickstart and checks local links; both example scripts regenerate executed notebook/HTML reports. |
| #44: descriptions | `tests/persistence/test_notebook_correctness.py` verifies explicit/default descriptions and YAML persistence. |
| #33: compact presentation | `tests/persistence/test_report_export_contract.py` executes compact HTML and checks that implementation inputs disappear while displayed results and diagnostics remain. |
| #41: consistent result context | Pipeline tests check provenance; `tests/estimators/test_csdid_degenerate.py` checks known effects and unavailable inference through strict JSON serialization. |
| #30: release gates | Known-effect estimator tests, five prespecified scientific benchmarks, supported Python CI, fresh base/all-extra wheels, and regenerated bundled reports. |
| #19: interactive CLI | Deferred as stated in the improvement plan; no claim of implementation. |
