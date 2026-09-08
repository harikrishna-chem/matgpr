# PyPI And Release Readiness

This page records the current package-publication state for `matgpr` and the
reusable gate for future `0.2.x` releases. Treat every TestPyPI or PyPI upload
as an explicit release action.

## Current v0.2.x State

Status as of 2026-09-08:

- Current package version: `0.2.0`.
- Current PyPI release: <https://pypi.org/project/matgpr/0.2.0/>.
- Current TestPyPI release: <https://test.pypi.org/project/matgpr/0.2.0/>.
- Current GitHub release: <https://github.com/harikrishna-chem/matgpr/releases/tag/v0.2.0>.
- Current Zenodo version DOI: <https://doi.org/10.5281/zenodo.22653726>.
- All-versions Zenodo concept DOI: <https://doi.org/10.5281/zenodo.21210386>.
- License metadata: `Apache-2.0`.
- Python support: Python 3.10, 3.11, and 3.12.
- Build backend: `setuptools.build_meta`.
- Publishing path: GitHub Actions Trusted Publishing with separate TestPyPI and
  PyPI environments.

The `v0.2.0` tag and PyPI files are immutable release artifacts. Do not retag,
force-push, delete, or overwrite them. Future fixes should be released as a new
patch version, such as `0.2.1`.

## Completed v0.2.0 Snapshot

The first live PyPI release passed these checks:

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

## Metadata Checklist

Before each future release, confirm:

- `pyproject.toml`, `matgpr/_version.py`, `CITATION.cff`, `README.md`, and
  `CHANGELOG.md` agree on the intended version.
- `CITATION.cff` does not contain a stale version DOI before Zenodo archives
  the new release.
- `README.md` installation, example, citation, and feature claims match the
  package being released.
- `CHANGELOG.md` has a dated release entry and a clean `Unreleased` section.
- The documentation site builds strictly and contains no obsolete release
  blocker language.
- Public examples and `dataset.pkl` links are still valid.
- Local draft notebooks, paper PDFs, private notes, caches, and generated
  artifacts are not staged.

## Package Contents

The wheel should contain:

- `matgpr/*.py`,
- `matgpr-<version>.dist-info/*`,
- `LICENSE`.

The wheel should not contain:

- public example notebooks,
- example datasets,
- documentation source,
- tests,
- local agent instructions,
- draft BO notebooks.

The source distribution should contain:

- package source,
- tests,
- `README.md`,
- `LICENSE`,
- `pyproject.toml`,
- `CITATION.cff`,
- `CHANGELOG.md`.

Examples and documentation remain repository assets. They should not be
silently installed through the Python package index.

## Trusted Publishing Setup

Trusted Publishing is already configured for the `matgpr` project with these
GitHub environments:

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

Keep using these environments so uploads can be review-gated in GitHub before
publication.

## Local Release Gate

Example for a future patch release:

```bash
VERSION=0.2.1
python -m ruff check matgpr tests scripts
python -m pytest
python -m mkdocs build --strict
rm -rf dist build matgpr.egg-info
python -m build
python -m twine check dist/matgpr-${VERSION}*
```

For public examples, also run:

```bash
python scripts/smoke_notebooks.py
```

## TestPyPI Gate

Run TestPyPI first from `main` after the release-preparation commit is pushed
and CI/docs are green:

```bash
VERSION=0.2.1
gh workflow run publish-pypi.yml \
  --ref main \
  -f target=testpypi \
  -f version=${VERSION}
```

Then verify a clean install:

```bash
VERSION=0.2.1
python -m venv /tmp/matgpr-testpypi
/tmp/matgpr-testpypi/bin/python -m pip install --upgrade pip
/tmp/matgpr-testpypi/bin/python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  "matgpr[examples,bo]==${VERSION}"
/tmp/matgpr-testpypi/bin/python -m pip check
/tmp/matgpr-testpypi/bin/python -c "import matgpr; print(matgpr.__version__)"
```

## Live PyPI Gate

Only publish to live PyPI after the TestPyPI gate passes, the final tag is
pushed, and the release has explicit approval.

```bash
VERSION=0.2.1
git tag -a v${VERSION} -m "matgpr v${VERSION}"
git push origin v${VERSION}
gh workflow run publish-pypi.yml \
  --ref v${VERSION} \
  -f target=pypi \
  -f version=${VERSION}
```

After upload, verify:

```bash
VERSION=0.2.1
python -m venv /tmp/matgpr-pypi
/tmp/matgpr-pypi/bin/python -m pip install --upgrade pip
/tmp/matgpr-pypi/bin/python -m pip install "matgpr[examples,bo]==${VERSION}"
/tmp/matgpr-pypi/bin/python -m pip check
/tmp/matgpr-pypi/bin/python -c "import matgpr; print(matgpr.__version__)"
```

## Post-Release Follow-Up

After each live release:

- create or update the GitHub Release notes,
- confirm PyPI and documentation pages open,
- wait for Zenodo to archive the release,
- record the version DOI and concept DOI,
- update `README.md`, `CITATION.cff`, documentation, and release notes with
  the new DOI,
- push the DOI documentation follow-up,
- verify CI, Docs, and the deployed MkDocs site.

## v0.2.x Development Direction

The `v0.2.x` line should focus on polish, interoperability, and adoption before
large API expansion:

- keep public APIs stable where practical,
- add compatibility aliases only when they are low maintenance,
- improve examples and documentation before adding many new model families,
- keep BO draft notebooks uncommitted until reviewed,
- consider a local MCP server after a design review confirms the tool boundary,
  security model, and maintenance cost.
