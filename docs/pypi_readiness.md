# PyPI Readiness Audit

This page records the current PyPI readiness status for `matgpr`. It is not an
upload instruction by itself. Treat every PyPI or TestPyPI upload as an
explicit release action.

## Current Status

Status as of 2026-07-05:

- Package name: `matgpr`.
- Import package: `matgpr`.
- Current version: `0.1.1`.
- License metadata: `Apache-2.0`.
- Python support: Python 3.10, 3.11, and 3.12.
- Build backend: `setuptools.build_meta`.
- PyPI name check: `python -m pip index versions matgpr` returned no matching
  distribution on 2026-07-02.
- Local build: source distribution and wheel build successfully.
- README rendering check: `twine check dist/*` passes.
- Current recommendation: do not upload to live PyPI until the remaining
  blockers below are resolved.

The package name should be checked again immediately before the first upload
because PyPI availability can change.

## Audit Results

| Area | Current result | Status |
| --- | --- | --- |
| Project name | `matgpr` is valid and currently appears unused on PyPI | Ready, recheck before upload |
| Version | `pyproject.toml`, `CITATION.cff`, and `matgpr.__version__` use `0.1.1` | Ready |
| License | SPDX license expression and license file are included | Ready |
| Author metadata | Author and maintainer metadata are present in `pyproject.toml` | Ready |
| README | Markdown long description passes `twine check` | Ready |
| Wheel contents | Wheel includes only the importable package and license metadata | Ready |
| Source distribution contents | Source distribution includes package source, tests, README, license, and `pyproject.toml` | Acceptable |
| Public examples | Examples are not installed by the wheel; they remain repository examples | Intentional |
| TestPyPI upload | Not yet performed | Blocking before live PyPI |
| Clean install from TestPyPI | Not yet performed | Blocking before live PyPI |
| Documentation URL | `pyproject.toml` points to the configured GitHub Pages custom domain | Verify after each Pages deployment |
| Live PyPI account/token | Not verified in this audit | Blocking before live PyPI |

## Metadata Notes

`matgpr` uses modern project metadata in `pyproject.toml`, including:

- `[build-system]` with `setuptools.build_meta`,
- `[project]` name, version, description, README, Python requirement, license,
  authors, maintainers, classifiers, dependencies, and project URLs,
- `[project.optional-dependencies]` extras for examples, documentation,
  Bayesian optimization, and heavier fingerprinting backends.

The Python Packaging User Guide recommends declaring the build backend and
project metadata in `pyproject.toml`, including project name, version,
description, README, dependencies, license expression, license files, and
project URLs:

- <https://packaging.python.org/en/latest/guides/writing-pyproject-toml/>

## Package Contents

The wheel currently contains:

- `matgpr/*.py`,
- `matgpr-0.1.1.dist-info/*`,
- `LICENSE`.

The wheel does not contain:

- public example notebooks,
- example datasets,
- documentation source,
- tests,
- local agent instructions,
- BO draft notebooks.

This is a clean wheel for library installation. Public examples remain in the
GitHub repository, where Colab notebooks can fetch `dataset.pkl` files by raw
URL.

The source distribution currently contains:

- package source,
- tests,
- `README.md`,
- `LICENSE`,
- `pyproject.toml`.

Including tests in the source distribution is acceptable and useful for
downstream verification. If a smaller source distribution is preferred later,
add an explicit `MANIFEST.in` policy and recheck the artifact contents.

## Remaining Blockers Before Live PyPI

Do not upload to live PyPI until all of these are resolved:

- Confirm `https://harikrishnasahu.com/matgpr/` opens after the GitHub Pages
  deploy workflow runs.
- Register or verify the PyPI owner account and project ownership plan.
- Register or verify the TestPyPI account.
- Create scoped API tokens for TestPyPI and PyPI.
- Upload to TestPyPI first.
- Install from TestPyPI in a clean environment.
- Run `pip check` after the TestPyPI install.
- Verify `import matgpr` and `matgpr.__version__`.
- Verify extras installation strategy, especially `examples`, `docs`, and
  `bo`.
- Recheck package-name availability immediately before live upload.

## Recommended TestPyPI Flow

TestPyPI is a separate package index for testing the publishing flow. It has a
separate user database and can be used without affecting live PyPI:

- <https://packaging.python.org/en/latest/guides/using-testpypi/>

Build and check artifacts:

```bash
rm -rf dist build matgpr.egg-info
python -m build
python -m twine check dist/*
```

Upload to TestPyPI:

```bash
python -m twine upload --repository testpypi dist/*
```

Install from TestPyPI with live PyPI as the dependency source:

```bash
python -m venv /tmp/matgpr-testpypi
/tmp/matgpr-testpypi/bin/python -m pip install --upgrade pip
/tmp/matgpr-testpypi/bin/python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  "matgpr[examples,bo]==0.1.1"
/tmp/matgpr-testpypi/bin/python -m pip check
/tmp/matgpr-testpypi/bin/python -c "import matgpr; print(matgpr.__version__)"
```

The Python Packaging User Guide notes that TestPyPI is separate from live PyPI
and that `--extra-index-url` can be useful when dependencies come from live
PyPI.

## Live PyPI Upload Gate

Only upload to live PyPI after:

- TestPyPI upload and install are successful,
- CI and docs workflows are green on the release commit,
- release tag is final,
- README, metadata, license, and citation are reviewed,
- documentation URL decision is resolved,
- `CHANGELOG.md` has a dated release entry.

Live upload command:

```bash
python -m twine upload dist/*
```

After upload, immediately verify:

```bash
python -m venv /tmp/matgpr-pypi
/tmp/matgpr-pypi/bin/python -m pip install --upgrade pip
/tmp/matgpr-pypi/bin/python -m pip install "matgpr[examples,bo]==0.1.1"
/tmp/matgpr-pypi/bin/python -m pip check
/tmp/matgpr-pypi/bin/python -c "import matgpr; print(matgpr.__version__)"
```

## Final Recommendation

`matgpr` is close to PyPI-ready, but the first live upload should wait until:

- the documentation URL is deployed and verified,
- TestPyPI upload and clean install are completed,
- PyPI account ownership and token handling are confirmed.

Until then, the best public installation path remains a pinned GitHub release
or exact commit.
