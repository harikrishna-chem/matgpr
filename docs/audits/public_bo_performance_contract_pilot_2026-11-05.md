# Public BO Performance-Contract Pilot

**Date:** 2026-11-05
**Plan step:** Step 9, Slice 4
**Status:** Completed

## Scope

Add a bounded, hardware-independent performance contract for
`rank_bo_candidate_pool`. Finite-pool Bayesian optimization is a credible
scalability boundary because candidate pools may contain many rows, while
surrogate fitting and posterior evaluation should remain batched.

The pilot compares pools with 16 and 1,024 candidates, each containing three
numeric features. A lightweight counting surrogate replaces numerical GPR
fitting so the test measures workflow call shape rather than optimizer speed.

## Contract

Each public ranking request must perform:

- exactly one surrogate-construction call;
- exactly one model `eval()` call;
- exactly one posterior call containing the complete candidate matrix; and
- exactly one vectorized `rank_bo_candidate_predictions` call containing the
  complete prediction vector.

The output must contain one ranked row per input candidate. Increasing the
pool from 16 to 1,024 rows may increase vectorized numerical work, but it must
not introduce per-candidate surrogate fitting, posterior calls, or scoring
dispatch.

## TDD Evidence

- The focused public workflow contract passes in 1.05 seconds, including
  imports and test setup.
- Both candidate-pool sizes use one fit, one batched posterior call, and one
  saved-prediction scoring call.
- The posterior receives exact shapes `(16, 3)` and `(1024, 3)`.
- No repeated fitting or row-wise model evaluation was found, so production
  code was not changed.

## Non-Blocking Timing Evidence

After one warm-up call, five real `fit_model=False` runs produced these local
medians:

- 16 candidates: 2.207 ms;
- 1,024 candidates: 6.565 ms.

These measurements are environment-specific audit context. They are not test
thresholds and do not gate CI.

## Quality Gates

- The complete public suite passes 874 tests with 9 expected third-party
  warnings.
- Combined statement and branch-aware coverage is 83.81%, above the 78%
  project floor.
- Ruff lint and format checks pass.
- Strict MkDocs, sdist/wheel builds, Twine checks, and diff hygiene pass.
- No public export, function signature, dependency, or runtime behavior
  changed.

## Decision

Deterministic operation-count contracts provide useful performance protection
without hardware-sensitive failures. This bounded BO contract remains in the
ordinary component suite. Wider benchmarking is not justified by this pilot.

Step 9 continues with a manual scientific-convergence pilot for a small GPR or
PI-GPR workflow. Longer numerical evidence will remain outside ordinary
pull-request tests until Step 10 defines the slow-suite execution policy.
