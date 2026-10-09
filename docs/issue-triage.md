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

The thirteen reproduced findings and subsequent related corrections are summarized in [the changelog](../CHANGELOG.md). A local repair or draft PR does not by itself resolve the remote issue.
