# PyPI Readiness Audit

This page records the current PyPI readiness status for `matgpr`. It is not an
upload instruction by itself. Treat every TestPyPI or PyPI upload as an
explicit release action.

## Current Status

Status as of 2026-09-07:

- Package name: `matgpr`.
- Import package: `matgpr`.
- Target release version: `0.2.0`.
- License metadata: `Apache-2.0`.
- Python support: Python 3.10, 3.11, and 3.12.
- Build backend: `setuptools.build_meta`.
- PyPI name check: `python -m pip index versions matgpr` returned no matching
  distribution on 2026-09-07.
- Local package metadata, version files, release docs, and publish workflow are
  prepared for `0.2.0`.
- Local build: source distribution and wheel built successfully on 2026-09-07.
- README rendering check: exact-version `twine check` passed on 2026-09-07.
- Recommended publishing path: GitHub Actions Trusted Publishing with separate
  TestPyPI and PyPI environments.
- Current recommendation: upload to TestPyPI first, verify a clean install,
  then run the live PyPI job only from tag `v0.2.0`.

The package name should be checked again immediately before the first upload
because PyPI availability can change.

## Audit Results

| Area | Current result | Status |
| --- | --- | --- |
| Project name | `matgpr` is valid and currently appears unused on PyPI | Ready, recheck before upload |
| Version | `pyproject.toml`, `CITATION.cff`, and `matgpr.__version__` use `0.2.0` | Ready |
| License | SPDX license expression and license file are included | Ready |
| Author metadata | Author and maintainer metadata are present in `pyproject.toml` | Ready |
| README | Markdown long description passes exact-version `twine check` | Ready |
| Wheel contents | Wheel includes the importable package and license metadata | Ready |
| Source distribution contents | Source distribution includes package source, tests, README, license, `pyproject.toml`, `CITATION.cff`, and `CHANGELOG.md` | Ready |
| Public examples | Examples are not installed by the wheel; they remain repository examples | Intentional |
| Trusted Publishing workflow | `.github/workflows/publish-pypi.yml` builds artifacts and publishes through OIDC | Ready after CI review |
| TestPyPI trusted publisher | Must be configured in TestPyPI before running the workflow | Blocking before TestPyPI |
| Clean install from TestPyPI | Not yet performed | Blocking before live PyPI |
| Documentation URL | `pyproject.toml` points to the configured GitHub Pages custom domain | Verify after each Pages deployment |
| PyPI trusted publisher | Must be configured in PyPI before live upload | Blocking before live PyPI |

## Metadata Notes

`matgpr` uses modern project metadata in `pyproject.toml`, including:

- `[build-system]` with `setuptools.build_meta`,
- `[project]` name, version, description, README, Python requirement, license,
  authors, maintainers, classifiers, dependencies, and project URLs,
- `[project.optional-dependencies]` extras for examples, documentation,
  Bayesian optimization, and heavier fingerprinting backends.

Use Trusted Publishing/OIDC for uploads where possible. This avoids storing
long-lived PyPI API tokens in GitHub Secrets.

## Package Contents

The wheel should contain:

- `matgpr/*.py`,
- `matgpr-0.2.0.dist-info/*`,
- `LICENSE`.

The wheel should not contain:

- public example notebooks,
- example datasets,
- documentation source,
- tests,
- local agent instructions,
- BO draft notebooks.

This is a clean wheel for library installation. Public examples remain in the
GitHub repository, where Colab notebooks can fetch `dataset.pkl` files by raw
URL.

The source distribution should contain:

- package source,
- tests,
- `README.md`,
- `LICENSE`,
- `pyproject.toml`,
- `CITATION.cff`,
- `CHANGELOG.md`.

Including tests in the source distribution is acceptable and useful for
downstream verification. Full documentation sources and public examples remain
repository/Zenodo assets so notebooks and datasets are not silently installed
through the Python package index.

## Required Trusted Publisher Setup

Before running the publish workflow, configure Trusted Publishers on TestPyPI
and PyPI.

TestPyPI:

- owner: `harikrishna-chem`
- repository: `matgpr`
- workflow filename: `publish-pypi.yml`
- environment: `testpypi`

PyPI:

- owner: `harikrishna-chem`
- repository: `matgpr`
- workflow filename: `publish-pypi.yml`
- environment: `pypi`

Use GitHub environments named `testpypi` and `pypi` so publish jobs can be
review-gated in GitHub before an upload occurs.

## Remaining Blockers Before Live PyPI

Do not upload to live PyPI until all of these are resolved:

- Confirm `https://harikrishnasahu.com/matgpr/` opens after the GitHub Pages
  deploy workflow runs.
- Confirm the `matgpr` project name still appears available.
- Register or verify the PyPI owner account and project ownership plan.
- Register or verify the TestPyPI account.
- Configure TestPyPI Trusted Publishing for `.github/workflows/publish-pypi.yml`
  with environment `testpypi`.
- Run the manual TestPyPI workflow.
- Install from TestPyPI in a clean environment.
- Run `pip check` after the TestPyPI install.
- Verify `import matgpr` and `matgpr.__version__`.
- Verify extras installation strategy, especially `examples`, `docs`, and
  `bo`.
- Create the final `v0.2.0` GitHub tag and release after CI/docs are green.
- Configure PyPI Trusted Publishing for `.github/workflows/publish-pypi.yml`
  with environment `pypi`.
- Run the manual live PyPI workflow from tag `v0.2.0`.

## Local Artifact Check

Build and check artifacts before triggering any upload workflow:

```bash
rm -rf dist build matgpr.egg-info
python -m build
VERSION=0.2.0
python -m twine check dist/matgpr-${VERSION}*
```

## TestPyPI Flow

Run the manual workflow from `main` after the TestPyPI trusted publisher is
configured:

```bash
gh workflow run publish-pypi.yml \
  --ref main \
  -f target=testpypi \
  -f version=0.2.0
```

Install from TestPyPI with live PyPI as the dependency source:

```bash
python -m venv /tmp/matgpr-testpypi
/tmp/matgpr-testpypi/bin/python -m pip install --upgrade pip
/tmp/matgpr-testpypi/bin/python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  "matgpr[examples,bo]==0.2.0"
/tmp/matgpr-testpypi/bin/python -m pip check
/tmp/matgpr-testpypi/bin/python -c "import matgpr; print(matgpr.__version__)"
```

## Live PyPI Upload Gate

Only upload to live PyPI after:

- TestPyPI upload and install are successful,
- CI and docs workflows are green on the release commit,
- release tag `v0.2.0` is final,
- README, metadata, license, and citation are reviewed,
- documentation URL decision is resolved,
- `CHANGELOG.md` has a dated release entry.

Run the manual workflow from the final tag:

```bash
gh workflow run publish-pypi.yml \
  --ref v0.2.0 \
  -f target=pypi \
  -f version=0.2.0
```

After upload, immediately verify:

```bash
python -m venv /tmp/matgpr-pypi
/tmp/matgpr-pypi/bin/python -m pip install --upgrade pip
/tmp/matgpr-pypi/bin/python -m pip install "matgpr[examples,bo]==0.2.0"
/tmp/matgpr-pypi/bin/python -m pip check
/tmp/matgpr-pypi/bin/python -c "import matgpr; print(matgpr.__version__)"
```

## DOI Follow-Up

`CITATION.cff` is prepared for `0.2.0` but does not include a new version DOI
until Zenodo archives the `v0.2.0` GitHub release. After Zenodo creates the new
record, update `README.md`, `CITATION.cff`, `.zenodo.json` if needed, and the
documentation with the exact `v0.2.0` DOI.

## Final Recommendation

`matgpr` is ready for a controlled first PyPI release once the local gate, CI,
docs workflow, TestPyPI upload, and TestPyPI clean-install checks pass.
