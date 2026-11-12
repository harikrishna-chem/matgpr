# Step 8 Public Structural Closeout

**Date:** 2026-11-05
**Plan step:** Step 8, Slice 9
**Status:** Completed

## Scope

Close the targeted public-package cleanup after remeasuring quality,
structure, exact duplication, dependency direction, public compatibility, and
direct test ownership. This is an audit-only slice; it makes no production-code
change.

## Baseline Comparison

| Measure | Step 8 start | Step 8 closeout | Change |
| --- | ---: | ---: | ---: |
| Tests | 582 | 849 | +267 |
| Subtests | - | 56 | - |
| Expected warnings | 9 | 9 | 0 |
| Statements | 12,314 | 12,455 | +141 |
| Branches | 4,484 | 4,488 | +4 |
| Branch-aware coverage | 81.19% | 83.62% | +2.43 points |
| Top-level modules | 45 | 50 | +5 private policy modules |
| Internal import edges | 57 | 68 | +11 |
| Import cycles | 0 | 0 | 0 |
| Functions/methods | 1,204 | 1,202 | -2 |
| Exact duplicate groups | 19 | 10 | -9 |
| Public exports | 293 | 293 | 0 |

The statement increase comes from scientific validation, security hardening,
and contract checks completed during the broader TDD program. The targeted
module extractions reduced the two intended facades while adding directly
tested private owners.

## Major Modules

| Module | Start lines | Closeout lines | Closeout statements | Branches | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bayesian_optimization.py` | 4,514 | 3,339 | 1,228 | 504 | 81.1% |
| `validation.py` | 3,128 | 2,979 | 1,008 | 374 | 74.2% |
| `safe_equations.py` | 2,015 | 2,139 | 947 | 362 | 86.4% |
| `multifidelity_gpr.py` | 1,647 | 1,647 | 664 | 176 | 85.0% |
| `mcp_tools.py` | 1,313 | 1,313 | 449 | 178 | 80.5% |
| `candidate_generation.py` | 1,271 | 1,271 | 551 | 230 | 81.8% |

The BO facade lost 1,175 physical lines, 469 statements, and 254 branches.
Validation lost 149 physical lines, 79 statements, and 44 branches. The
safe-equation module grew because bounded complexity, strict metadata, and
finite-output defenses were added; its parser and interpreters remain one
cohesive security owner.

## Step 8 Boundaries

| Boundary | Direct focused evidence | Result |
| --- | --- | --- |
| Shared finite-vector validation | 6 focused checks | Three duplicate bodies replaced by canonical aliases |
| Saved-prediction BO policy | 30 focused tests | 193 statements and 86 branches isolated at 100% |
| BO candidate-domain policy | 52 focused tests | 191 statements and 100 branches isolated at 100% |
| BO batch-selection policy | 20 focused tests | 109 statements and 44 branches isolated at 100% |
| BoTorch acquisition policy | 65 focused tests | 67 statements and 24 branches isolated at 100% |
| Learning-curve policy | 55 focused tests | 86 statements and 40 branches isolated at 100% |
| Safe-equation security | 39 focused tests | Security hardened; extraction deliberately rejected |

The five extracted modules have no dependency on their parent facade. Four
have no package-internal dependency; `_bo_saved_prediction` depends only on
the lower-level BO specification module. Parent-to-private dependency direction
is acyclic.

## Remaining Exact Duplicates

The strict signature-plus-body AST scan finds ten groups:

- three-line `n_features` properties on observation records;
- three-line train/test prediction projections on validation result records;
- four-line objective slug helpers in BO and multi-objective modules;
- four-line dataframe-copy validators in BO audit and candidate generation;
- eight-line numeric-matrix adapters in derivative and virtual-observation
  modules;
- two sklearn featurizer `get_feature_names_out` methods;
- three trivial sklearn kernel property groups; and
- one three-line target-transform inverse method group.

None warrants extraction now. The record, sklearn, kernel, and transform
methods are local protocol implementations. The slug, dataframe, and numeric
matrix helpers are only 4–8 lines and own domain-specific future behavior;
sharing them would add a generic dependency without reducing meaningful risk.

## Remaining Hotspots

- `bayesian_optimization.py` is now primarily a public facade plus fitting and
  workflow orchestration. Further splitting requires a new behavior-driven
  task, not cleanup by line count.
- `validation.py` retains estimator fitting, split generation, multifidelity,
  co-kriging, and result assembly after pure request policy moved out.
- `safe_equations.py` intentionally retains public records, allowlist policy,
  both interpreters, and the mean adapter as one security boundary.
- Large fitting and explanation functions remain cohesive scientific
  workflows with direct tests. They are candidates only when a scientific,
  performance, or maintenance defect establishes a better boundary.

## Compatibility

- The ordered 293-name export digest remains
  `b64f5af236df40169a73e93bb6031e3f7afb85ea866c8531746edd7c8eb46204`.
- Public BO, validation, learning-curve, and safe-equation signatures remain
  unchanged.
- The 50-module, 68-edge import graph is acyclic.
- Every extracted policy has direct focused tests and exact facade aliases
  where a prior private name existed.

## Decision

Step 8 is complete. No production edit is justified by this closeout. Move to
a bounded property-based test pilot instead of continuing structural work.

## Next Task

Step 9, Slice 1: use Hypothesis to test target-transform round trips,
monotonicity, finite shapes, bounded-domain behavior, and uncertainty
invariants. Keep the profile deterministic and small enough for ordinary pull
requests.
