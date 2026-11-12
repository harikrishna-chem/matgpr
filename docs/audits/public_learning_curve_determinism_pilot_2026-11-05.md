# Public Learning-Curve Determinism Pilot

**Date:** 2026-11-05
**Plan step:** Step 9, Slice 3
**Status:** Completed

## Scope

Characterize the seed contract of `matgpr.validation.learning_curve` without
fitting a GPR. This workflow is a stronger determinism owner than the simple
train/test helper because one root seed controls:

- repeated holdout splits;
- train-pool subsampling at each requested learning-curve size; and
- derived `random_state` values assigned to cloned estimators.

The pilot uses 24 fixed nonlinear samples, two train sizes, two repeats, and a
small randomized decision tree. The estimator is intentionally seed-aware but
CPU-light.

## Contracts

Three focused tests require:

1. Two calls with the same seed produce exactly equal run, prediction, and
   summary dataframes, together with identical result metadata.
2. Calls with different seeds retain the same schemas but change sampled row
   assignments.
3. Runs across seeds 0 through 9 retain stable schemas, finite numerical
   outputs, correct sample labels and targets, and train/test row counts that
   agree with every run record.

The multi-seed test does not require scores to improve, degrade, or remain
equal. Different seeds are expected to vary the sampled data and fitted
estimator while preserving the result contract.

## TDD Evidence

- The initial deterministic-estimator characterization passed.
- The estimator was then strengthened to
  `DecisionTreeRegressor(max_depth=3, splitter="random")`, so the same root
  seed also exercises matgpr's derived estimator-seed policy.
- All three contracts continued to pass in 1.21 seconds.
- No reproducible defect or flaky outcome was found, so production code was
  not changed.

## Quality Gates

- The complete public suite passes 873 tests with 9 expected third-party
  warnings.
- Combined statement and branch-aware coverage is 83.81%, above the 78%
  project floor.
- Ruff lint and format checks pass.
- Strict MkDocs, sdist/wheel builds, Twine checks, and diff hygiene pass.
- No public export, function signature, dependency, or runtime behavior
  changed.

## Decision

The existing `learning_curve` seed policy is reproducible and suitable for
ordinary pull-request testing. The three focused tests remain in the normal
unit suite because their fixed workloads are small and do not depend on wall
clock thresholds.

Step 9 continues with a bounded performance-regression pilot. That pilot will
prefer deterministic operation or call-count contracts and use wall-clock
measurements only as non-blocking audit evidence.
