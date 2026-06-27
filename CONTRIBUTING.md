# Contributing to matgpr

Thank you for your interest in improving `matgpr`.

`matgpr` is currently in early development. The project is being shaped around
high-quality materials-informatics workflows, physics-informed GPR models,
well-documented examples, and reproducible benchmarks.

## Current Licensing Status

The final public license has not been selected yet. The current preferred
direction is free academic/research use with a separate paid commercial license,
citation expectations, and restrictions on redistribution/modification.

Because the license model is still being finalized:

- do not submit large external code contributions unless contribution terms
  have been agreed in writing,
- small issue reports, documentation suggestions, and reproducibility feedback
  are welcome,
- a final contribution policy will be added after the license decision is made.

See `docs/license_strategy.md` for the current licensing notes.

## Development Setup

From a local checkout:

```bash
python -m pip install -U pip
python -m pip install -e ".[dev,examples]"
```

For optional fingerprinting backends:

```bash
python -m pip install -e ".[structures]"
python -m pip install -e ".[molecular-extra]"
python -m pip install -e ".[all-fingerprints]"
```

## Quality Checks

Run these before proposing changes:

```bash
python -m ruff check matgpr tests scripts
python -m pytest
python -m build
```

For notebook integration smoke checks:

```bash
python scripts/smoke_notebooks.py
```

The smoke script executes reduced versions of the notebooks. It is not a
replacement for full scientific benchmark runs.

## Adding New Examples

Each example should include:

- `dataset.csv` or a clear data-preparation script,
- a notebook with standard GPR and at least one physics-informed GPR model,
- learning curves with repeated splits,
- 90/10 validation with 10-fold cross-validation,
- train/test parity with uncertainty bars,
- production-model interpretation such as SHAP or sensitivity analysis,
- a short report explaining physics, equations, learned parameters, and
  performance.

Do not commit paper PDFs unless redistribution rights are clear.

## Pull Request Expectations

Pull requests should:

- be focused on one coherent change,
- include tests for new package behavior,
- update docs when public APIs or workflows change,
- avoid unrelated formatting churn,
- explain any scientific assumptions.
