# PyPI Release v0.2.0 Plan

Date: 2026-09-07

## Scope

Prepare `matgpr` for its first PyPI release as `v0.2.0`.

This task covers package metadata, release automation, and documentation. It
does not upload to TestPyPI or PyPI, create a GitHub tag, create a GitHub
release, or modify Zenodo records.

## Ownership Boundary

- `matgpr` owns public package metadata, PyPI release workflow, release docs,
  versioning docs, and install instructions.
- GitHub Actions owns build and publishing automation after the workflow is
  triggered manually.
- PyPI/TestPyPI accounts, trusted-publisher setup, project ownership, and live
  upload approval remain explicit human release actions.
- Private `genmatics-matgpr` and the GenMatics app are not part of this public
  release-prep task.

## Ordered Steps

1. Confirm the working tree and leave unrelated untracked files untouched.
2. Recheck whether the public PyPI project name `matgpr` appears available.
3. Bump public package metadata from `0.1.1` to `0.2.0`.
4. Add a manual GitHub Actions workflow for TestPyPI and PyPI Trusted
   Publishing.
5. Document required trusted-publisher settings for TestPyPI and PyPI.
6. Refresh install, versioning, release-checklist, and PyPI-readiness docs.
7. Build and check release artifacts locally.
8. Run package tests and documentation checks.

## Acceptance Checks

- `pyproject.toml`, `matgpr.__version__`, `CITATION.cff`, and docs agree on
  `0.2.0`.
- The publish workflow has separate TestPyPI and live PyPI jobs.
- The live PyPI job requires the workflow to run from tag `v0.2.0`.
- Publishing uses PyPI Trusted Publishing/OIDC, not long-lived API tokens.
- Release docs clearly state that TestPyPI and PyPI uploads require explicit
  release approval.
- Package artifacts pass `twine check`.
- Tests, lint, and MkDocs strict build pass.

## Verification Commands

Run from `/Users/hari/projects/genmatics/matgpr`:

```bash
python -m ruff check matgpr tests scripts
python -m pytest
python -m mkdocs build --strict
python -m build
python -m twine check dist/matgpr-0.2.0*
```

## Handoff Notes

After review and push, configure Trusted Publishers on TestPyPI and PyPI using:

- Repository owner: `harikrishna-chem`
- Repository name: `matgpr`
- Workflow filename: `publish-pypi.yml`
- TestPyPI environment: `testpypi`
- PyPI environment: `pypi`

Then run the workflow manually for TestPyPI first. Only run the live PyPI job
from the final `v0.2.0` tag after TestPyPI install checks pass.
