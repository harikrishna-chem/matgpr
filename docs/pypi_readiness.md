# PyPI Readiness Audit

This page records the current PyPI readiness status for `matgpr`. It is not an
upload instruction by itself. Treat every TestPyPI or PyPI upload as an
explicit release action.

## Current Status

Status as of 2026-09-08:

- Package name: `matgpr`.
- Import package: `matgpr`.
- Current PyPI release version: `0.2.0`.
- License metadata: `Apache-2.0`.
- Python support: Python 3.10, 3.11, and 3.12.
- Build backend: `setuptools.build_meta`.
- PyPI project: <https://pypi.org/project/matgpr/0.2.0/>.
- TestPyPI project: <https://test.pypi.org/project/matgpr/0.2.0/>.
- Zenodo version DOI: <https://doi.org/10.5281/zenodo.22653726>.
- Zenodo concept DOI: <https://doi.org/10.5281/zenodo.21210386>.
- Local package metadata, version files, release docs, and publish workflow were
  prepared for `0.2.0`.
- Local build: source distribution and wheel built successfully on 2026-09-07.
- README rendering check: exact-version `twine check` passed on 2026-09-07.
- Publishing path: GitHub Actions Trusted Publishing with separate TestPyPI and
  PyPI environments.
- TestPyPI publication and clean install passed on 2026-09-08.
- Live PyPI publication and clean install passed on 2026-09-08.

## Audit Results

| Area | Current result | Status |
| --- | --- | --- |
| Project name | `matgpr` is published on PyPI | Complete |
| Version | `pyproject.toml`, `CITATION.cff`, and `matgpr.__version__` use `0.2.0` | Ready |
| License | SPDX license expression and license file are included | Ready |
| Author metadata | Author and maintainer metadata are present in `pyproject.toml` | Ready |
| README | Markdown long description passes exact-version `twine check` | Ready |
| Wheel contents | Wheel includes the importable package and license metadata | Ready |
| Source distribution contents | Source distribution includes package source, tests, README, license, `pyproject.toml`, `CITATION.cff`, and `CHANGELOG.md` | Ready |
| Public examples | Examples are not installed by the wheel; they remain repository examples | Intentional |
| Trusted Publishing workflow | `.github/workflows/publish-pypi.yml` builds artifacts and publishes through OIDC | Ready after CI review |
| TestPyPI trusted publisher | Configured for `publish-pypi.yml` and environment `testpypi` | Complete |
| Clean install from TestPyPI | Passed for `matgpr[examples,bo]==0.2.0` | Complete |
| Documentation URL | `pyproject.toml` points to the configured GitHub Pages custom domain | Verify after each Pages deployment |
| PyPI trusted publisher | Configured for `publish-pypi.yml` and environment `pypi` | Complete |
| Clean install from PyPI | Passed for `matgpr==0.2.0` | Complete |

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

## Completed v0.2.0 Release Checks

The first live PyPI release has passed these checks:

- `https://harikrishnasahu.com/matgpr/` returned `HTTP/2 200`.
- `https://pypi.org/project/matgpr/` returned `HTTP/2 200`.
- `python -m pip index versions matgpr` reported `0.2.0` as latest.
- TestPyPI Trusted Publishing was configured and the TestPyPI workflow passed.
- TestPyPI clean install passed for `matgpr[examples,bo]==0.2.0`.
- PyPI Trusted Publishing was configured and the live PyPI workflow passed.
- Live PyPI clean install passed for `matgpr==0.2.0`.
- `pip check` passed after both TestPyPI and live PyPI clean installs.
- `matgpr.__version__` reported `0.2.0` after clean installation.
- Zenodo archived `v0.2.0` with DOI
  <https://doi.org/10.5281/zenodo.22653726>.

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

Zenodo archived the `v0.2.0` GitHub release as
<https://doi.org/10.5281/zenodo.22653726>. The all-versions concept DOI is
<https://doi.org/10.5281/zenodo.21210386>.

## Final Recommendation

For future releases, keep using the same controlled path: local gate, CI/docs,
TestPyPI upload and clean install, live PyPI upload from the final tag, GitHub
Release, Zenodo archive check, then DOI documentation follow-up.
