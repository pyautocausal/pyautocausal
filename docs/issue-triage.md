# Issue triage for the correctness release

Status at the **12 October 2026** release. Tracking owner: **ytchokni**. The correctness implementation and release follow-up merged in [PR #71](https://github.com/pyautocausal/pyautocausal/pull/71) and [PR #72](https://github.com/pyautocausal/pyautocausal/pull/72). [Version 0.1.2](https://github.com/pyautocausal/pyautocausal/releases/tag/v0.1.2) is published on GitHub. Acceptance checks were repeated against the merged source and nine completed issues were closed. Original enhancement requests that exceed this release remain open with the narrower follow-up scope below.

| Issue | Disposition | Acceptance criteria |
| --- | --- | --- |
| [#43](https://github.com/pyautocausal/pyautocausal/issues/43) | Verified in 0.1.2 | Alternative branch outputs can bind one parameter; ambiguous active inputs fail clearly. |
| [#62](https://github.com/pyautocausal/pyautocausal/issues/62) | Verified in 0.1.2 | Integer and string unit-ID panels yield equivalent estimates through the public pipeline. |
| [#64](https://github.com/pyautocausal/pyautocausal/issues/64) | Verified in 0.1.2 | Missingness threshold and drop/reject policy are configurable, audited and validated after cleaning. |
| [#58](https://github.com/pyautocausal/pyautocausal/issues/58) | Verified in 0.1.2 | Missing branch prerequisites fail before partially constructing the branch; valid composition is tested. |
| [#42](https://github.com/pyautocausal/pyautocausal/issues/42) | Verified in 0.1.2 | Export diagnostics preserve failed and skipped nodes without presenting failure as success. |
| [#38](https://github.com/pyautocausal/pyautocausal/issues/38) | Verified in 0.1.2 | Installed notebook extra executes a generated notebook to HTML using the intended interpreter. |
| [#66](https://github.com/pyautocausal/pyautocausal/issues/66) | Verified in 0.1.2 | README local links resolve and advertised quickstarts execute. |
| [#44](https://github.com/pyautocausal/pyautocausal/issues/44) | Verified in 0.1.2 | Node descriptions survive composition/export; explicit descriptions can supplement callable docstrings. |
| [#33](https://github.com/pyautocausal/pyautocausal/issues/33) | Verified in 0.1.2 | An optional compact notebook view hides implementation clutter without removing execution or diagnostic information. |
| [#41](https://github.com/pyautocausal/pyautocausal/issues/41) | Open presentation follow-up | Shared method/sample/transformation/inference provenance is implemented. Unified coefficient-table and plot styling across estimators remains separate work. |
| [#30](https://github.com/pyautocausal/pyautocausal/issues/30) | Open notebook portability follow-up | Correctness and installation release gates are implemented. The original goal of notebooks running without installing PyAutoCausal is not implemented; package helper imports remain necessary. |
| [#19](https://github.com/pyautocausal/pyautocausal/issues/19) | Deferred product feature | Interactive CLI nodes remain separate product work and were not implemented in this correctness release. |

The thirteen original findings and subsequent corrections from independent reviews are summarized in [the changelog](../CHANGELOG.md). The three remaining enhancements (#19, #30 and #41) are not represented as completed features.

## Verification evidence

The [final-main CI](https://github.com/pyautocausal/pyautocausal/actions/runs/38237367997) passed all 10 jobs, including 484 tests in each of six locked or freshly resolved environments across Python 3.10–3.12. All five [scientific benchmarks](https://github.com/pyautocausal/pyautocausal/actions/runs/38236428662) passed. The release includes correction notes, source and wheel checksums, and the measured validation evidence. These counts describe that release, not a fixed requirement on future test-suite size.

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
