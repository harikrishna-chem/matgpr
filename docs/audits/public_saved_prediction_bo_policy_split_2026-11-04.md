# Saved-Prediction BO Acquisition Policy Split

**Date:** 2026-11-04
**Plan step:** Step 8, Slice 3
**Status:** Completed

## Scope

Extract deterministic saved-prediction acquisition policy and numerical
helpers from `bayesian_optimization.py` into a private internal module.

The public `rank_bo_candidate_predictions(...)` facade remains responsible for
input coercion, candidate-table construction, generated columns, stable
sorting, and top-k selection.

## Extraction Boundary

Move only:

- supported acquisition constants and intent resolution;
- posterior-mean, UCB/LCB, EI/PI, and target-probability values;
- objective/constraint event intervals and intersections;
- feasibility weighting and joint same-target probability;
- normal CDF/PDF, interval probability, target closeness, and unit scaling;
- incumbent, target-tolerance, and same-target constraint validation.

Keep in the parent:

- public result classes and ranking functions;
- dataframe and candidate-index construction;
- constraint probability-column naming;
- generated-column assembly, stable sorting, and top-k selection;
- BoTorch model fitting and all multi-objective workflows.

## Compatibility Requirements

- Preserve all public signatures and the ordered 293-name export inventory.
- Preserve every moved parent private name as an exact function/constant alias.
- Preserve acquisition values, effective names, generated columns, errors,
  stable tie ordering, and numerical tolerances.
- Keep the new module dependent only on NumPy, SciPy through the existing lazy
  CDF import, and BO specification records.
- Introduce no import cycle.

## TDD Sequence

1. Add direct policy characterization tests against current parent helpers.
2. Require the parent helpers to alias a new internal module and observe the
   intended pre-extraction adoption failure.
3. Move the characterized helpers without rewriting their mathematics.
4. Run focused, public-facade, scientific, full-suite, documentation, and
   package-build gates.

## TDD Evidence

- Before extraction, 22 behavior tests passed and the module-adoption identity
  test failed as intended because `_bo_saved_prediction.py` did not exist.
- The focused saved-prediction and BO scientific selection finished with 90
  passing tests and four expected third-party warnings.
- The new internal module has direct coverage for all 193 statements and 86
  branches.

## Implementation

- Added private module `matgpr._bo_saved_prediction` for deterministic
  acquisition intent, analytical scoring, interval, feasibility, and
  same-target constraint policy.
- Retained every moved parent private name as an exact alias, including both
  acquisition constant sets and all 19 helper functions.
- Kept candidate dataframe assembly, generated-column naming, stable sorting,
  top-k selection, public results, BoTorch fitting, and multi-objective
  workflows in `bayesian_optimization.py`.
- Declared an empty module `__all__` so the internal split adds no public API.
  The full public-API gate caught this requirement after the first extraction
  pass, and the omission was fixed before completion.

## Structural Result

| Measure | Before | After |
| --- | ---: | ---: |
| Public BO module physical lines | 4,514 | 4,076 |
| Public BO module statements | 1,697 | 1,531 |
| Public BO module branches | 758 | 672 |
| Public BO module branch-aware coverage | 74.83% | 75.35% |
| Internal policy statements | - | 193 |
| Internal policy branches | - | 86 |
| Internal policy coverage | - | 100% |
| Complete public tests | 588 | 618 |
| Complete branch-aware coverage | 81.25% | 81.74% |

The parent module lost 438 physical lines, 166 statements, and 86 branches.
All 293 ordered package exports exactly match the frozen baseline and retain
SHA-256 `b64f5af236df40169a73e93bb6031e3f7afb85ea866c8531746edd7c8eb46204`.
All 15 public BO function signatures match the pre-split source. The package
now has 46 top-level modules and 64 internal import edges with no cycle; the
new module imports only `bayesian_optimization_specs` internally.

## Verification

- Full branch-aware suite: 618 passed, 52 subtests passed, and 9 expected
  third-party warnings.
- Ruff lint, Ruff formatting, and `git diff --check`: passed.
- Strict MkDocs build: passed.
- Offline source-distribution and wheel build: passed.

## Next Boundary

Proceed with a separately characterized candidate-domain policy slice. Keep
batch selection, acquisition evaluation, BoTorch fitting, and public facades
outside that extraction.
