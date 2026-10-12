# Contributing

Use Python 3.10–3.12 and Poetry **2.1.1**, matching CI. Clone the repository, create a feature branch, and select the Python interpreter you intend to test:

```bash
git clone https://github.com/pyautocausal/pyautocausal.git
cd pyautocausal
git switch -c your-change
python -m pip install poetry==2.1.1
poetry env use python3.12
poetry install --all-extras
```

Use another supported interpreter in place of `python3.12` when testing that version. Run commands with `poetry run` so a system interpreter or a stale Jupyter kernel does not determine the environment.

## Tests and documentation

From the repository root:

```bash
MPLBACKEND=Agg poetry run pytest tests/ -m 'not scientific'
poetry check --lock
poetry build
```

For a documentation change, execute the checked quickstarts with:

```bash
MPLBACKEND=Agg poetry run pytest tests/test_documentation.py
```

Also execute new or changed snippets and check relative links in the guides you edit. The [documentation index](docs/README.md) is the navigation entry point. Keep examples aligned with the implemented API and clearly mark extra dependencies, shared state between snippets, and experimental behavior.

Run the longer, prespecified statistical benchmarks before a correctness release. They repeatedly refit models and may take tens of minutes on a hosted runner:

```bash
MPLBACKEND=Agg poetry run pytest tests/ -m scientific
```

Tests must verify the behavior that failed: known treatment effects, row-order invariance, covariate forwarding, complete graph execution, faithful saved output and installed-package behavior. File existence and successful execution alone do not establish correctness. Numerical tolerances and simulation designs should be fixed before examining results.

## Installed-wheel validation

CI tests base and all-extra wheel installations outside the source checkout for each supported Python version. All-extra wheels also run the fast suite against freshly resolved dependencies, separately from the locked development environment. To reproduce locally, create a new virtual environment, install the wheel (with `[all]` for extras), change into a temporary directory, and run the absolute path to `scripts/smoke_installed.py` with `python -I` and `--mode base` or `--mode all`.

For the current 0.1.2 wheel, after `poetry build`:

```bash
repo_dir="$PWD"
wheel_check_dir="$(mktemp -d)"
python3.12 -m venv "$wheel_check_dir/venv"
"$wheel_check_dir/venv/bin/python" -m pip install "$repo_dir/dist/pyautocausal-0.1.2-py3-none-any.whl[all]"
cd "$wheel_check_dir"
MPLBACKEND=Agg "$wheel_check_dir/venv/bin/python" -I "$repo_dir/scripts/smoke_installed.py" --mode all
cd "$repo_dir"
```

Adjust the artifact name when testing a new version. Test the base installation in a **separate fresh environment**, omit `[all]`, and use `--mode base`; the base smoke test verifies that notebook and DoubleML packages are absent. These checks exercise dataset loading, estimation and persistence, plus an actual DoubleML fit and notebook-to-HTML execution for the all-extra installation.

The aggregate `Required checks` CI status fails if any supported-version test or wheel job fails. Preserve it as a merge requirement. The scientific workflow can be run manually for a candidate release; its release trigger provides a repeat check, not a substitute for validation before release.

See the [fast/installed-wheel workflow](.github/workflows/pytest_validation.yml) and [scientific workflow](.github/workflows/scientific_validation.yml) for the exact matrix and uploaded test reports. Do not use development dependencies to fill a missing runtime or optional dependency in a wheel.

## Statistical changes

Document the estimand, identification assumptions, uncertainty method, sample requirements and random-state behavior. Cite and compare with a primary reference implementation where possible. If inference has not been validated, report it as unavailable rather than supplying plausible-looking intervals. Do not add artificial variation to predictions or plots.

## Release checklist

1. Fast tests, supported-version matrix, installed wheels and scientific benchmarks pass.
2. Run the documented examples into a fresh output directory; inspect reports and diagnostics.
3. Check the [published PyPI versions](https://pypi.org/project/pyautocausal/#history) and choose an unused version; GitHub tags and repository metadata may lag package-index releases. Update version metadata and the changelog. Explain which earlier results need to be rerun.
4. Review the complete diff and issue acceptance criteria; merge only after required checks and review pass.
5. Build the release artifacts from the reviewed commit, verify installation from the source archive, and record SHA-256 checksums. Tag and publish the reviewed version. Keep repository and package-index publication separate; a successful local build is not a published release.
6. Verify installation from the public download URL, including the documented examples. Publish validation evidence with the source commit and artifact hashes. Update installation links, correction notes and issue status only to reflect what has actually shipped.

Do not overwrite local analysis outputs or include user datasets in a release.
