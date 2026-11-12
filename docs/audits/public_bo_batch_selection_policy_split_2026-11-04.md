# BO Diversity-Aware Batch Selection Policy Split

**Date:** 2026-11-04
**Plan step:** Step 8, Slice 5
**Status:** Completed

## Scope

Extract pure finite-pool batch-selection policy from
`bayesian_optimization.py` into a private internal module after current score,
feature, distance, and tie behavior is characterized.

## Extraction Boundary

Move only:

- reserved generated-output column names used for feature inference;
- finite score conversion, normalization, stable top-k, and descending ranks;
- explicit and inferred diversity-feature resolution;
- constant-feature removal and optional standardization;
- stable argmax and greedy diversity-aware position selection.

Keep in the parent:

- public `select_diverse_batch(...)` validation and dataframe facade;
- batch annotation columns, output sorting, and returned table assembly;
- sequential multi-objective selection, acquisition workflows, and fitting.

## Compatibility Requirements

- Preserve public signatures, result schemas, stable row ordering, selection
  scores, distances, errors, and numerical behavior.
- Retain every moved parent private name and constant as an exact alias.
- Give the internal module no public exports and no package-internal imports.
- Correct duplicate input-index annotation by working in positional row space;
  returned outputs already reset their indexes.

## TDD Sequence

1. Characterize vector conversion, normalization, stable ties, feature
   resolution, matrix preparation, greedy selection, and stopping rules.
2. Demonstrate the duplicate-index defect and missing-module adoption failure.
3. Fix positional annotation and extract the pure helpers without rewriting
   their mathematics.
4. Run focused, neighboring BO, public API, full-suite, docs, and build gates.

## TDD Evidence

- Before production edits, 18 behavior tests passed. Two intended failures
  reproduced duplicate-index over-selection and the missing internal module.
- The completed focused suite has 20 tests and covers all 109 statements and
  44 branches in the new module.
- Neighboring BO and public API selection: 165 tests and 48 subtests passed
  with four expected third-party warnings.

## Implementation

- Added private `_bo_batch_selection.py` for the reserved output-column tuple,
  finite score conversion, normalization, stable ranking, diversity-feature
  preparation, and greedy position selection.
- Retained the moved constant and all eight moved functions as exact parent
  aliases. All eight function bodies are AST-identical to their prior
  implementations.
- Kept public validation, annotation columns, output sorting, and dataframe
  result assembly in `select_diverse_batch(...)`.
- Fixed duplicate-index over-selection by resetting only the facade's internal
  working index. Public outputs already reset their indexes, so unique-index
  behavior and output schemas are unchanged.
- Declared an empty module `__all__`; the new module imports no package module.

## Structural Result

| Measure | Before | After |
| --- | ---: | ---: |
| Public BO module physical lines | 3,729 | 3,540 |
| Public BO module statements | 1,373 | 1,278 |
| Public BO module branches | 572 | 528 |
| Public BO module branch-aware coverage | 78.97% | 79.40% |
| Internal batch-policy statements | - | 109 |
| Internal batch-policy branches | - | 44 |
| Internal batch-policy coverage | - | 100% |
| Complete public tests | 670 | 690 |
| Complete branch-aware coverage | 82.57% | 82.81% |

The parent module lost 189 physical lines, 95 statements, and 44 branches.
All 293 ordered exports and all 15 public BO function signatures remain exact.
The package now has 48 top-level modules and 66 internal import edges with no
cycle; the new module has no package-internal edge.

## Verification

- Full branch-aware suite: 690 passed, 54 subtests passed, and 9 expected
  third-party warnings.
- Ruff lint, Ruff formatting, and `git diff --check`: passed.
- Strict MkDocs build: passed.
- Offline source-distribution and wheel build: passed.

## Next Boundary

Characterize single- and multi-objective BoTorch acquisition-name
normalization and factory construction. Keep posterior evaluation, candidate
ranking, sequential selection, and public facades in the parent module.
