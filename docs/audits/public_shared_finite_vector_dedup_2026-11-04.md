# Shared Finite-Vector Validation Deduplication

**Date:** 2026-11-04
**Plan step:** Step 8, Slice 2
**Status:** Completed

## Scope

Consolidate the exact one-dimensional finite float-array validation copies in
reporting, validation, and visualization under the existing private canonical
owner, `_validation.to_1d_finite`.

## Compatibility Requirements

- Preserve all 293 ordered package exports.
- Preserve the local private helper names used inside each module.
- Preserve float conversion, row-major flattening, non-empty/finite checks,
  exception types, and exact error messages.
- Preserve visualization's relative-import fallback for direct module use.
- Do not change workflow outputs or numerical behavior.

## TDD Sequence

1. Characterize canonical success and failure behavior directly.
2. Require all three local helpers to be the canonical function object.
3. Observe identity-test failure before the refactor.
4. Replace duplicate definitions with import aliases.
5. Run focused and full quality gates before closing the slice.

## TDD Evidence

Before the production edit, the six-test characterization module produced the
expected result: five canonical behavior tests passed and the alias-identity
test failed at `reporting._to_1d_numeric`.

The characterization locks:

- float conversion and row-major flattening;
- one-dimensional output shape and float dtype;
- empty-input rejection and exact message;
- `NaN`, positive infinity, and negative infinity rejection and exact message;
- canonical function-object identity at all three local module surfaces.

## Implementation

- Imported `_validation.to_1d_finite` as `_to_1d_numeric` in reporting.
- Imported `_validation.to_1d_finite` as `_to_1d_float` in validation.
- Imported `_validation.to_1d_finite` as `_to_1d_float_array` in both relative
  and direct-module visualization import paths.
- Removed the three exact duplicate function bodies.
- Changed no public function, class, signature, export, error, or workflow
  result.

## Result

| Measure | Before | After |
| --- | ---: | ---: |
| Tests | 582 | 588 |
| Statements | 12,314 | 12,296 |
| Branches | 4,484 | 4,472 |
| Branch-aware coverage | 81.19% | 81.25% |
| Function definitions | 1,204 | 1,201 |
| Exact duplicate groups | 19 | 18 |
| Ordered public exports | 293 | 293 |

The top-level import graph now has 58 edges and remains acyclic. The ordered
public export list and frozen SHA-256 are unchanged.

## Verification

- Pre-edit TDD checkpoint: 5 passed, 1 expected failure.
- Focused helper/reporting/validation/visualization/API gate: 41 passed plus 45
  subtests.
- Full branch-aware suite: 588 passed, 9 expected third-party warnings,
  81.25% coverage.
- Ruff lint, Ruff formatting, `git diff --check`, strict MkDocs, and offline
  source/wheel builds: passed.

## Decision

Step 8, Slice 2 is complete. The next meaningful cleanup target is the pure
saved-prediction acquisition policy currently embedded in
`bayesian_optimization.py`. It should receive intent-, uncertainty-,
constraint-, ranking-, and numerical-characterization tests before extraction
behind the unchanged public `rank_bo_candidate_predictions(...)` facade.
