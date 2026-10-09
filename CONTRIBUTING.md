# Contributing

Use Python 3.10–3.12. Create a feature branch and install the locked environment:

```bash
poetry install --all-extras
MPLBACKEND=Agg poetry run pytest tests/ -m 'not scientific'
poetry check --lock
poetry build
```

Run the longer, prespecified statistical benchmarks before a correctness release:

```bash
MPLBACKEND=Agg poetry run pytest tests/ -m scientific
```

Tests must verify the behavior that failed: known treatment effects, row-order invariance, covariate forwarding, complete graph execution, faithful saved output and installed-package behavior. File existence and successful execution alone do not establish correctness. Numerical tolerances and simulation designs should be fixed before examining results.

## Installed-wheel validation

CI tests base and all-extra wheel installations outside the source checkout for each supported Python version. All-extra wheels also run the fast suite against freshly resolved dependencies, separately from the locked development environment. To reproduce locally, create a new virtual environment, install the wheel (with `[all]` for extras), change into a temporary directory, and run the absolute path to `scripts/smoke_installed.py` with `python -I` and `--mode base` or `--mode all`.

The aggregate `Required checks` CI status fails if any supported-version test or wheel job fails. Preserve it as a merge requirement. The scientific workflow can be run manually for a candidate release; its release trigger provides a repeat check, not a substitute for validation before release.

## Statistical changes

Document the estimand, identification assumptions, uncertainty method, sample requirements and random-state behavior. Cite and compare with a primary reference implementation where possible. If inference has not been validated, report it as unavailable rather than supplying plausible-looking intervals. Do not add artificial variation to predictions or plots.

## Release checklist

1. Fast tests, supported-version matrix, installed wheels and scientific benchmarks pass.
2. Run the documented examples into a fresh output directory; inspect reports and diagnostics.
3. Check the [published PyPI versions](https://pypi.org/project/pyautocausal/#history) and choose an unused version; GitHub tags and repository metadata may lag package-index releases. Update version metadata and the changelog. Explain which earlier results need to be rerun.
4. Review the complete diff and issue acceptance criteria; merge only after required checks and review pass.
5. Build the release artifacts from the reviewed commit, verify installation from the source archive, and record SHA-256 checksums. Tag and publish the reviewed version. Keep repository and package-index publication separate; a successful local build is not a published release.

Do not overwrite local analysis outputs or include user datasets in a release.
