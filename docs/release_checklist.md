# Release Checklist

This checklist is the release gate for `matgpr` `v0.1.0` and can be reused for
later `0.x` releases. The goal is to make each release reproducible,
auditable, citable, and easy for materials-informatics users to install.

## Release Scope

Before tagging a release, write down the intended scope:

- release version, for example `v0.1.0`,
- release date,
- release owner,
- short release theme,
- user-facing features included,
- user-facing features intentionally deferred,
- known limitations.

For `v0.1.0`, the intended theme is:

```text
First public release of matgpr: Gaussian Process Regression, uncertainty,
materials featurization, physics-informed mean functions, validation helpers,
Bayesian-optimization utilities, and two published-paper examples.
```

## Do Not Release If

Do not tag the release if any of these are true:

- CI is failing on `main`.
- Documentation build is failing.
- Package build is failing.
- Public notebooks cannot access their `dataset.pkl` files.
- Example datasets, notebooks, or reports include files that should not be
  redistributed.
- The version in `pyproject.toml` and `CITATION.cff` do not match.
- The changelog does not explain the release.
- The README points users to functionality that is not available.
- License text, citation guidance, or package metadata are inconsistent.

## Metadata Checks

Confirm these files are correct before release:

- `pyproject.toml`
  - `version`
  - package description
  - Python requirement
  - license metadata
  - project URLs
  - dependency extras
- `CITATION.cff`
  - title
  - author list
  - version
  - repository URL
  - license
- `CHANGELOG.md`
  - move relevant `Unreleased` notes under the release heading,
  - include the release date,
  - keep user-facing changes clear and concise.
- `README.md`
  - installation instructions,
  - documentation links,
  - examples,
  - versioning guidance,
  - citation and license notes.
- `LICENSE`
  - must match package metadata.

## Repository Hygiene

Run these checks before committing release preparation changes:

```bash
git status -sb --untracked-files=all
git diff --check
```

Confirm:

- no paper PDFs are staged,
- no local notebooks, temporary outputs, caches, or private notes are staged,
- untracked BO example drafts remain uncommitted until intentionally added,
- `AGENTS.md` and other local agent instructions remain ignored,
- generated directories such as `dist/`, `build/`, `site/`, and
  `matgpr.egg-info/` are not committed.

## Local Validation

Use the project development environment and run:

```bash
python -m ruff check matgpr tests scripts
python -m pytest
python -m mkdocs build --strict
python -m build
python -m twine check dist/*
```

For the public examples, run at least the reduced notebook smoke test:

```bash
python scripts/smoke_notebooks.py
```

For `v0.1.0`, also run a fresh-clone smoke test before tagging:

```bash
python -m venv /tmp/matgpr-release-smoke
/tmp/matgpr-release-smoke/bin/python -m pip install --upgrade pip
/tmp/matgpr-release-smoke/bin/python -m pip install "matgpr[examples,bo] @ git+https://github.com/harikrishna-chem/matgpr.git@main"
/tmp/matgpr-release-smoke/bin/python -m pip check
```

Then verify:

- `import matgpr` works,
- `matgpr.__version__` matches the release version,
- OPV and solvent diffusivity raw `dataset.pkl` URLs are reachable,
- notebook smoke execution works from the fresh clone,
- package build artifacts pass `twine check`.

## CI And GitHub Checks

Before tagging:

- push release-preparation changes to `main`,
- wait for CI to pass on GitHub Actions,
- wait for the docs workflow to pass,
- confirm the latest commit on `main` is the intended release commit,
- record the release commit hash.

The release tag should point to a green commit on `main`.

## Tagging

After local checks and CI are green, create an annotated tag:

```bash
git tag -a v0.1.0 -m "matgpr v0.1.0"
git push origin v0.1.0
```

Use an annotated tag so the release has explicit release metadata. Do not
retag or force-push a public release tag unless the release is broken and the
correction has been clearly documented.

## GitHub Release

Create a GitHub release from the tag. The release notes should include:

- one-sentence summary,
- major features,
- public examples,
- validation status,
- installation command,
- citation guidance,
- known limitations,
- link to the changelog.

Suggested install command for a GitHub-tagged release:

```bash
python -m pip install "matgpr[examples] @ git+https://github.com/harikrishna-chem/matgpr.git@v0.1.0"
```

If Bayesian optimization examples or APIs are needed:

```bash
python -m pip install "matgpr[examples,bo] @ git+https://github.com/harikrishna-chem/matgpr.git@v0.1.0"
```

## Zenodo DOI

Before the first DOI-backed release:

- enable or confirm GitHub-Zenodo archiving for the repository,
- keep `.zenodo.json` metadata current before tagging future releases,
- confirm `CITATION.cff` is correct,
- create the GitHub release from the final tag,
- let Zenodo archive the release,
- edit the Zenodo record metadata if needed,
- add the Zenodo DOI to `README.md`, `CITATION.cff`, and the documentation in
  a follow-up commit.

Once a DOI exists, tell users to cite the DOI for the exact release they used
and the original papers for the example datasets.

For GitHub-connected Zenodo releases:

1. Log in to Zenodo with the GitHub account that can access
   `harikrishna-chem/matgpr`.
2. Open the Zenodo GitHub settings page, sync repositories if needed, and enable
   archiving for `harikrishna-chem/matgpr`.
3. Confirm that GitHub Release `v0.1.0` is archived or trigger the available
   Zenodo sync/archive action for the release.
4. If Zenodo cannot archive an already-published GitHub release, keep the
   GitHub Release as the software release of record and create the DOI on the
   next patch release after Zenodo is enabled.
5. Record both the version DOI and concept DOI. Use the version DOI for exact
   reproducibility and the concept DOI for the latest package line.

## PyPI Readiness

Do not upload to PyPI until the package name, metadata, and release artifacts
are reviewed. Before any PyPI release:

- review `docs/pypi_readiness.md`,
- confirm the `matgpr` package name and account access,
- upload to TestPyPI first,
- install from TestPyPI in a clean environment,
- confirm extras install as expected,
- confirm `pip check` passes,
- confirm the README renders correctly on the package index,
- only then upload to PyPI.

Recommended TestPyPI flow:

```bash
python -m build
python -m twine check dist/*
python -m twine upload --repository testpypi dist/*
```

PyPI upload should be treated as a separate explicit release decision.

## Documentation Deployment

Before enabling public hosted docs:

- confirm the MkDocs site builds with `python -m mkdocs build --strict`,
- confirm the docs nav includes the release checklist, quickstart, user guide,
  example cards, physics-informed GPR guide, fingerprinting guide, versioning
  policy, API reference, and license page,
- confirm GitHub Pages settings use GitHub Actions as the intended source,
- confirm the MkDocs docs workflow uploads the `site/` artifact and deploys it
  with GitHub Pages actions,
- wait for the deploy workflow to pass.

After deployment, confirm the public docs URL opens:

```text
https://harikrishnasahu.com/matgpr/
```

## Post-Release Checks

After the release is published:

- install from the release tag in a clean environment,
- run `pip check`,
- run a minimal standard GPR example,
- run a minimal physics-informed GPR example,
- open the GitHub release page,
- open the documentation site if deployed,
- verify citation and license links,
- record the release hash, DOI if available, and package source in the project
  log.

## Release Notes Template

````markdown
# matgpr v0.1.0

matgpr v0.1.0 is the first public release of a Gaussian Process Regression
toolkit for materials informatics, with physics-informed mean functions,
uncertainty-aware prediction, reusable validation workflows, materials
featurization, and published-paper examples.

## Highlights

- Standard and physics-informed GPR workflows.
- GPyTorch exact GPR with predictive uncertainty.
- Scikit-learn-style estimators and featurizers.
- RDKit polymer and molecule fingerprints.
- Composition and structure featurization utilities.
- Validation, uncertainty diagnostics, and plotting helpers.
- Optional BoTorch finite-pool Bayesian optimization.
- OPV and solvent diffusivity examples with dataset and model cards.

## Install

```bash
python -m pip install "matgpr[examples] @ git+https://github.com/harikrishna-chem/matgpr.git@v0.1.0"
```

## Cite

Please cite `matgpr` using `CITATION.cff`. For example notebooks, also cite
the original dataset or paper listed in the example report and cards.

## Known Limitations

- `matgpr` is an active-development `0.x` package.
- APIs may change before `1.0.0`.
- The public examples are tutorials and should not be treated as universal
  production models without domain validation.
````
