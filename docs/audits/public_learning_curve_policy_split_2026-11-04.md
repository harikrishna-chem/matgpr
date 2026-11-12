# Public Learning-Curve Policy Split Audit

**Date:** 2026-11-04
**Plan step:** Step 8, Slice 7
**Status:** Completed

## Scope

Step 8, Slice 7 isolates the pure input and metric policy used by the public
learning-curve workflows in `matgpr.validation`.

The extraction covers:

- explicit and generated train-size resolution;
- percent, fraction, and count unit aliases;
- inclusive train-size range generation;
- RMSE, MAE, R2, and Pearson-r metric aliases;
- train/test metric selection; and
- learning-curve metric column-name generation and validation.

## Boundary

The private policy module must not own estimator fitting, random splitting,
subsampling, prediction, multifidelity logic, co-kriging logic, or public
result assembly. Those responsibilities remain in `matgpr.validation`.

## Compatibility Contract

- Existing public signatures and exports remain unchanged.
- Explicit train sizes preserve caller order and duplicates.
- Generated ranges include a reachable stop value and round floating-point
  accumulation to 12 decimal places.
- Metric and split selections preserve first canonical occurrence order.
- Existing validation errors and generated column names remain stable.
- The combined split aliases `both`, `train-test`, and `train_test` are
  case-insensitive and tolerate surrounding whitespace.
- `matgpr.validation` exposes exact aliases to the extracted helper objects so
  internal callers and downstream users of the private names see one policy
  implementation.

## TDD Evidence

- Before production edits, 53 characterization tests passed. The two intended
  failures identified the missing internal module and inconsistent whitespace
  handling for the combined `both` split alias.
- The completed focused suite has 55 tests and covers all 86 statements and 40
  branches in `_learning_curve_policy.py`.
- The focused policy plus neighboring validation selection has 77 passing
  tests.

## Implementation

- Added private `_learning_curve_policy.py` with no public exports or internal
  package dependencies.
- Moved eight pure train-size, metric, split, and column helpers out of
  `validation.py`.
- Retained all eight former parent private names as exact aliases to the new
  module.
- Preserved caller ordering and duplicate explicit train sizes, inclusive
  generated ranges, canonical metric ordering, errors, and column names.
- Made `both`, `train-test`, and `train_test` consistently tolerate outer
  whitespace, matching all other train-size, metric, and split aliases.
- Kept estimator cloning and fitting, split generation, subsampling,
  multifidelity/co-kriging execution, and public result assembly in the parent
  module.

## Structural Result

- `validation.py` now has 2,979 physical lines, 1,008 statements, and 374
  branches. Its branch-aware coverage is 74.2%.
- The new policy module has 160 physical lines and 100% direct statement and
  branch coverage.
- The ordered 293-name public export inventory and its frozen digest remain
  exact.
- The package has 50 top-level modules and 68 internal import edges with no
  cycle. The policy module has no package-internal dependency.

## Verification

- Full branch-aware suite: 810 tests and 56 subtests passed at 83.32% coverage,
  with 9 expected third-party warnings.
- Ruff lint, Ruff formatting, and `git diff --check`: passed.
- Strict MkDocs build: passed.
- Offline source distribution and wheel build plus Twine metadata checks:
  passed.

## Next Boundary

Proceed to a security-first characterization of the safe-equation parser and
evaluation boundary. Do not split that module unless adversarial parser tests
demonstrate a cohesive, acyclic extraction that preserves every public
signature and numerical contract.
