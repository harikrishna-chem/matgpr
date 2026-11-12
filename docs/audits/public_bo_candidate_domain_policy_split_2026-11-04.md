# BO Candidate-Domain Policy Split

**Date:** 2026-11-04
**Plan step:** Step 8, Slice 4
**Status:** Completed

## Scope

Extract deterministic finite-pool candidate-domain policy from
`bayesian_optimization.py` into a private internal module after current
behavior is directly characterized.

## Extraction Boundary

Move only:

- candidate policy, output-name, and distance-metric normalization;
- numeric dataframe/array feature-matrix construction;
- raw, standard-deviation, and explicit feature-scale handling;
- Euclidean, Manhattan, and Chebyshev nearest-reference distances;
- stable existing-candidate IDs and exact row-key construction;
- trust-region and duplicate annotations.

Keep in the parent:

- public candidate policy dataclasses and their module identity;
- public annotation/filtering functions and runtime type validation;
- combined domain-policy orchestration and result assembly;
- acquisition scoring, ranking, batch selection, and BoTorch fitting.

## Compatibility Requirements

- Preserve all public imports, signatures, dataclass identities, errors,
  generated columns, row order, masks, and numerical results.
- Retain every moved parent private name as an exact alias.
- Give the new internal module no public exports and no dependency on the
  parent BO facade.
- Keep the package import graph acyclic.

## TDD Sequence

1. Characterize normalizers, feature matrices, scales, distance metrics,
   empty inputs, IDs, row keys, trust regions, and duplicate combinations.
2. Require exact parent aliases to a not-yet-created internal module and retain
   that intended adoption failure.
3. Extract the characterized helpers without changing their mathematics.
4. Run focused, neighboring BO, public API, full-suite, docs, and build gates.

## TDD Evidence

- After correcting two test expectation mistakes about nearest-reference
  distance, 50 characterization tests passed and only the intended missing
  internal-module adoption test failed.
- The completed focused suite has 52 tests and covers all 191 statements and
  100 branches in the new module.
- Neighboring BO and public API selection: 145 tests and 47 subtests passed
  with four expected third-party warnings.

## Implementation

- Added private `_bo_candidate_domain.py` with structural protocols for the
  parent-owned trust-region and duplicate-policy dataclasses.
- Moved 13 deterministic helpers for policy/name normalization, matrix and
  scale preparation, distances, IDs, row keys, and annotations.
- Retained all 13 prior parent private names as exact aliases.
- Kept public dataclasses, public apply functions, runtime type checks,
  combined policy orchestration, fitting, ranking, and result assembly in
  `bayesian_optimization.py`.
- Declared an empty module `__all__`; the internal module imports no package
  module and cannot form a facade dependency cycle.
- Verified that all 13 extracted function bodies are AST-identical to their
  pre-extraction implementations.

## Structural Result

| Measure | Before | After |
| --- | ---: | ---: |
| Public BO module physical lines | 4,076 | 3,729 |
| Public BO module statements | 1,531 | 1,373 |
| Public BO module branches | 672 | 572 |
| Public BO module branch-aware coverage | 75.35% | 78.97% |
| Internal domain-policy statements | - | 191 |
| Internal domain-policy branches | - | 100 |
| Internal domain-policy coverage | - | 100% |
| Complete public tests | 618 | 670 |
| Complete branch-aware coverage | 81.74% | 82.57% |

The parent module lost 347 physical lines, 158 statements, and 100 branches.
All 293 ordered exports retain their exact order and frozen digest. All 15
public BO function signatures match the pre-split source. The package now has
47 top-level modules and 65 internal import edges with no cycle.

## Verification

- Full branch-aware suite: 670 passed, 53 subtests passed, and 9 expected
  third-party warnings.
- Ruff lint, Ruff formatting, and `git diff --check`: passed.
- Strict MkDocs build: passed.
- Offline source-distribution and wheel build: passed.

## Next Boundary

Characterize diversity-aware finite-pool batch selection before extracting its
pure scoring, feature-matrix, stable ordering, and greedy selection helpers.
Keep the public `select_diverse_batch(...)` result-assembly facade in place.
