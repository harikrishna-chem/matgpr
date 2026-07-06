# Versioning And Stability

`matgpr` is currently an active-development `0.x` package. The public goal is
to keep the project useful and reproducible while the API is still being shaped
around materials-informatics workflows.

## Current Status

- Current package version: `0.1.1`.
- Python support: Python 3.10 or newer.
- License: Apache-2.0.
- Repository: <https://github.com/harikrishna-chem/matgpr>.

The package already has CI, tests, examples, documentation, and a changelog, but
it should still be treated as an early public release. APIs may change before
`1.0.0` as the package adds more physics-informed workflows, Bayesian
optimization examples, and additional materials-informatics integrations.

## Versioning Policy

`matgpr` follows semantic-versioning conventions in spirit:

- Patch releases, such as `0.1.1`, are intended for bug fixes,
  documentation fixes, and small backward-compatible improvements.
- Minor releases, such as `0.2.0`, may add new APIs and may include breaking
  changes while the package remains in `0.x`.
- A future `1.0.0` release will mark the point where the core public API is
  considered stable.

Before `1.0.0`, breaking changes should still be intentional, documented in
`CHANGELOG.md`, and limited to changes that materially improve the package
quality, clarity, or scientific correctness.

## What Users Should Pin

For reproducible research, avoid installing from a moving branch such as `main`.
Pin a release tag or commit hash instead.

For a tagged GitHub release:

```bash
python -m pip install "matgpr[examples] @ git+https://github.com/harikrishna-chem/matgpr.git@v0.1.1"
```

For an exact commit:

```bash
python -m pip install "matgpr[examples] @ git+https://github.com/harikrishna-chem/matgpr.git@<commit-hash>"
```

When `matgpr` is published to PyPI, use exact package pins for manuscripts,
benchmarks, and production workflows:

```bash
python -m pip install "matgpr[examples]==0.1.1"
```

For long-lived projects, also save the Python version and dependency state:

```bash
python -m pip freeze > requirements-lock.txt
```

or, for conda-based workflows:

```bash
conda env export > environment-lock.yml
```

## Dependency Policy

The library keeps install requirements reasonably broad so `matgpr` can work
with modern scientific Python environments. Reproducibility should come from
pinning the `matgpr` version or commit and saving an environment lock file for
the project that used it.

Core dependencies are kept lightweight enough for common GPR workflows. Heavier
materials-informatics backends are exposed as optional extras, for example:

```bash
python -m pip install "matgpr[materials-extra]"
python -m pip install "matgpr[bo]"
python -m pip install "matgpr[all-fingerprints]"
```

## API Stability Guidelines

The following APIs are intended to remain as stable as practical during the
`0.x` phase:

- top-level imports documented in the README and API reference,
- scikit-learn-style estimators and featurizers,
- `PhysicsInformedMean` and GPyTorch GPR helpers,
- validation, uncertainty, and plotting utilities used by public examples,
- optional BoTorch finite-pool Bayesian-optimization helpers.

When a public API must change, the preferred approach is:

- document the change in `CHANGELOG.md`,
- keep a compatibility alias when it does not add meaningful maintenance risk,
- provide a clear error or warning when old behavior is no longer supported,
- update examples and documentation in the same change.

## Citation Guidance

When publishing results built with `matgpr`, cite:

- the `matgpr` version, release tag, or commit hash,
- the package citation from `CITATION.cff`,
- the original paper or dataset associated with each example or dataset used.

Once release DOIs are enabled, prefer citing the DOI for the exact release used
in the study.
