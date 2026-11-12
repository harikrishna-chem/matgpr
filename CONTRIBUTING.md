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
python -m ruff format --check matgpr tests scripts
python -m pytest --cov=matgpr --cov-branch --cov-report=term-missing --cov-fail-under=79
python -m build
```

The suite registers `unit`, `contract`, `component`, `integration`,
`scientific`, `smoke`, `slow`, and `optional` markers. Every test must have
exactly one primary marker from the first five; collection fails for missing
or conflicting ownership. The `smoke`, `slow`, and `optional` markers are
orthogonal modifiers. The unfiltered command above remains the authoritative
full-suite gate. Select focused tiers with:

```bash
python -m pytest -m unit
python -m pytest -m contract
python -m pytest -m component
python -m pytest -m integration
python -m pytest -m scientific
python -m pytest -m smoke
python -m pytest -m optional
```

CI runs the complete branch-coverage suite on Python 3.11 and the focused
`unit or contract` compatibility selection on Python 3.10 and 3.12. A separate
package-quality job owns lint, formatting, distribution build, Twine metadata
validation, and isolated wheel smoke. The minimum-dependency job still runs
the full suite at the declared lower bounds.

A dedicated `MCP integration smoke` job installs `matgpr[mcp]`, validates the
optional dependency graph, and runs the packaged MCP client against the
packaged stdio server. Keep MCP-specific dependencies out of core and ordinary
test jobs.

### Property-Based Tests

Hypothesis is included in the `dev` extra. The shared `matgpr` test profile
uses 50 deterministic examples per property, disables timing deadlines, and
keeps generated inputs bounded for reliable pull-request execution. Run the
current property-test pilot with:

```bash
python -m pytest tests/test_target_transform_properties.py
```

Mutation testing is a manual diagnostic, not a pull-request CI gate. Install
the dedicated extra and run the current module-scoped pilot with:

```bash
python -m pip install -e ".[mutation]"
python -m mutmut run
python -m mutmut results
```

The checked-in configuration mutates only `matgpr/target_transforms.py` and
selects its example, scientific, and property tests. Review meaningful
survivors before adding tests; raw mutation-score increases are not a reason to
encode implementation details as behavior.

### Scientific Convergence

Longer numerical checks remain outside ordinary pull-request testing. The
`Scientific convergence` workflow runs the current physics-informed GPR pilot
every Monday at 17:23 UTC and supports manual dispatch. Run the same check
locally with:

```bash
python scripts/check_pi_gpr_convergence.py
```

The pilot learns a thermal-expansion slope while holding the reference lattice
parameter fixed, then reports loss, parameter error, extrapolation error,
latent uncertainty, interval coverage, and runtime across optimizer budgets up
to 1,600 iterations. Its assertions protect scientifically meaningful endpoint
behavior; intermediate RMSE values are allowed to vary. Runtime is diagnostic
evidence, not a pass/fail threshold.

CI also installs the built wheel without dependencies in a fresh virtual
environment and runs `scripts/smoke_wheel_install.py` from outside the source
tree. This ensures editable installs cannot hide missing package files or
incorrect package metadata.

For notebook integration smoke checks:

```bash
python scripts/smoke_notebooks.py
```

The smoke script executes every public example notebook: the OPV and solvent
diffusivity GPR notebooks, and the three OPV Bayesian-optimization notebooks.
The GPR notebooks run with reduced repeats, iterations, and sample sizes; the
Bayesian-optimization notebooks are small enough to run unreduced. It is not a
replacement for full scientific benchmark runs.

When you add an example notebook, add it to `NOTEBOOKS` in
`scripts/smoke_notebooks.py` so CI executes it.

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

Physics-informed examples should only be proposed for public inclusion when
the validation protocol shows a clear, reproducible reason to prefer the
physics-informed model over the standard baseline in the stated low-data
regime. Include the low-data gate metric in the model card and link the source
paper or dataset citation in the dataset card. See
`docs/benchmark_summary.md` for the current public-example standard.

Do not commit paper PDFs unless redistribution rights are clear.

## Pull Request Expectations

Pull requests should:

- be focused on one coherent change,
- include tests for new package behavior,
- update docs when public APIs or workflows change,
- avoid unrelated formatting churn,
- explain any scientific assumptions.
