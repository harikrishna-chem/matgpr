# Public Target-Transform Mutation Pilot

**Date:** 2026-11-05
**Plan step:** Step 9, Slice 2
**Status:** Completed

## Scope

Run a bounded manual mutation assessment of `matgpr/target_transforms.py`
against its three owning test modules:

- `tests/test_target_transforms.py`;
- `tests/test_target_transform_scientific.py`; and
- `tests/test_target_transform_properties.py`.

Only covered lines are mutated. Mutation testing remains outside ordinary CI
and is installed through the dedicated `mutation` extra.

## Tool and Execution Policy

- `mutmut 3.8.1` is pinned to the compatible `>=3.8,<4` range in the
  dedicated `mutation` extra.
- The isolated mutation workspace receives the complete `matgpr/` package,
  while `only_mutate` limits mutations to `matgpr/target_transforms.py`.
- Test selection is limited to the three owning modules listed above and
  `mutate_only_covered_lines` prevents uncovered defensive code from
  distorting the pilot.
- The pilot is a manual quality tool. It is documented for contributors but
  is not part of ordinary pull-request CI.

## Decision Rules

- Add a test when a survivor represents a missing scientific, validation, or
  compatibility decision.
- Do not add brittle assertions solely to increase a mutation score.
- Classify equivalent, unreachable, logging-only, and defensive mutations
  explicitly rather than treating them as test failures.
- Change production code only when the pilot exposes a reproducible defect.

## Mutation Evidence

The clean baseline generated 827 covered-line mutants:

- 509 killed;
- 318 survived;
- 0 timed out or produced suspicious outcomes; and
- 61.55% mutation score.

The survivor review identified missing tests for public discovery records,
registry filtering, constructor boundaries, exact validation contracts,
optional prediction fields, and a small-noise scientific limit. Thirteen
focused tests were added before the clean pilot was repeated.

The final run generated 850 covered-line mutants:

- 717 killed;
- 133 survived;
- 0 timed out or produced suspicious outcomes; and
- 84.35% mutation score.

The final clean run completed in 8.8 seconds. The changed mutant count reflects
additional lines becoming covered by the new contract tests.

## Meaningful Gaps Closed

The added tests now lock:

- every documented low-level transform factory alias and its keyword
  handling;
- exact unknown-transform choices and registry ordering/filtering;
- immutable specification normalization and copy isolation;
- JSON-safe summary, discovery, and dataframe shapes;
- bounded-transform constructor limits and the three-point quadrature
  boundary;
- exact public array-validation messages;
- missing standardized fitted-statistic failures;
- bounded mean-only optional-field preservation; and
- the bounded logistic-normal delta-method limit at small noise.

## Remaining Survivors

The 133 survivors are dominated by equivalent or intentionally unobservable
changes:

- substitutions in internal validation labels whose public error behavior is
  already exercised at the owning public boundary;
- removal of unused `**kwargs` in transforms that deliberately accept a
  uniform factory call shape;
- sign reversal in symmetric Gauss-Hermite quadrature;
- algebraically equivalent sigmoid branches and dtype choices after inputs
  have already been normalized to finite floating-point arrays; and
- optional-field initialization forms that produce the same record.

Adding implementation-shaped assertions for these survivors would make the
tests more brittle without protecting a scientific or compatibility decision.

## Quality Gates

- The three owning modules pass 42 tests and 22 subtests.
- The complete public suite passes 870 tests with 9 expected third-party
  warnings and 83.78% combined branch-aware coverage.
- `matgpr/target_transforms.py` reaches 97.5% branch-aware coverage.
- Ruff lint and format, strict MkDocs, sdist/wheel builds, Twine checks, wheel
  dependency metadata, and `git diff --check` all pass.
- `mutmut` is conditional on the `mutation` extra; Hypothesis remains
  conditional on `dev`. Neither is an unconditional runtime dependency.
- No production code, public export, function signature, or runtime behavior
  changed in this slice.

## Decision

Manual module-scoped mutation testing found useful contract gaps at practical
cost, so the documented pilot remains available to contributors. It will not
be added to ordinary CI or expanded package-wide without another bounded
decision. Step 9 continues with a repeated-seed determinism pilot for one
stochastic public workflow.
