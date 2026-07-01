# Contributing to matgpr

Thank you for your interest in improving `matgpr`.

`matgpr` is currently in early development. The project is being shaped around
high-quality materials-informatics workflows, physics-informed GPR models,
well-documented examples, and reproducible benchmarks.

## Licensing

`matgpr` is released under the Apache License 2.0. See `LICENSE`.

Small issue reports, documentation suggestions, and reproducibility feedback
are welcome. Larger external code contributions should be discussed first so
scope, testing expectations, and attribution are clear.

## Development Setup

From a local checkout:

```bash
python -m pip install -U pip
python -m pip install -e ".[dev,examples,bo]"
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

- `dataset.pkl` or a clear data-preparation script,
- a notebook with standard GPR and at least one physics-informed GPR model,
- learning curves with repeated splits,
- 90/10 validation with 10-fold cross-validation,
- train/test parity with uncertainty bars when it clarifies model behavior,
- production-model interpretation such as SHAP or sensitivity analysis when it
  is computationally practical and scientifically useful,
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
