# Public Target-Transform Property Pilot

**Date:** 2026-11-05
**Plan step:** Step 9, Slice 1
**Status:** Completed

## Scope

Introduce a bounded property-based testing pilot for pure target-transform
mathematics. Hypothesis is a development-only dependency and does not change
the installed runtime package.

The pilot covers:

- identity, standardization, log, bounded-logit, and physics-residual round
  trips;
- shape, finiteness, row-order, and monotonicity invariants;
- log-normal uncertainty finiteness and non-negativity;
- bounded logistic-normal means and standard deviations;
- optional prediction-field alignment; and
- deterministic pull-request execution with 50 examples per property and no
  timing deadline.

## Boundary

No GPR model is fitted. Generated arrays contain at most 20 rows, and numerical
ranges avoid overflow-only stress cases already owned by example-based
validation tests. Production code changes are allowed only for a reproducible
mathematical or shape defect exposed by the generated cases.

## TDD Evidence

- The initial bounded property suite passed all eight properties.
- Refining the paired residual/baseline generator exposed a subtraction and
  addition round-trip difference of about `1.82e-12` at a magnitude of
  `10,000`.
- The difference was normal floating-point cancellation, not a production
  defect. The test now uses `rtol=5e-15` and `atol=1e-12` for that invariant;
  production mathematics was not relaxed or changed.
- Each run accepts 50 examples for each of eight properties, for 400 generated
  examples in the pilot.

## Results

- All 8 property tests pass; the focused target-transform selection has 29
  passing tests.
- `target_transforms.py` branch-aware coverage increased from 90.8% to 92.6%.
- The complete public suite increased from 849 to 857 passing tests, retained
  56 passing subtests and 9 expected warnings, and reached 83.66% combined
  branch-aware coverage.
- Ruff lint and format, strict MkDocs, sdist/wheel builds, and Twine checks all
  pass.
- No production code, public export, function signature, or runtime dependency
  changed.

## Dependency and Execution Policy

- `hypothesis>=6.100` is present only in the `dev` extra.
- Built-wheel metadata records `Requires-Dist: hypothesis>=6.100; extra ==
  "dev"`; an ordinary installation cannot pull it as an unconditional
  dependency.
- `tests/conftest.py` registers the deterministic `matgpr` profile with 50
  examples and no timing deadline. `HYPOTHESIS_PROFILE` can select another
  explicitly registered profile.
- `.hypothesis/` is ignored so local example databases do not enter source
  control.

## Decision

The pilot is fast and catches useful numerical boundary assumptions, so the
property suite remains part of ordinary development tests. Expansion is not
automatic: the next bounded task is a manual mutation-testing pilot scoped to
`target_transforms.py` and its existing example, scientific, and property
tests.
