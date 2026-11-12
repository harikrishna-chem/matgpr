# BoTorch Acquisition Construction Policy Split

**Date:** 2026-11-04
**Plan step:** Step 8, Slice 6
**Status:** Completed

## Scope

Extract deterministic acquisition-name normalization and lazy BoTorch factory
construction from `bayesian_optimization.py` into a private internal module.

## Extraction Boundary

Move only:

- single-objective acquisition aliases and constructor selection;
- multi-objective acquisition aliases and constructor selection;
- sampler, partitioning, baseline, incumbent, beta, and pending-point wiring.

Keep in the parent:

- public surrogate dataclasses and their module identity;
- Monte-Carlo sample-count validation;
- posterior evaluation and conversion back to physical objective directions;
- candidate ranking, sequential batch selection, and public BO facades.

## Compatibility Requirements

- Preserve every supported alias, exact rejection message, lazy import, and
  constructor argument.
- Retain all four moved parent private names as exact aliases.
- Use structural protocols so the internal module does not import the parent
  facade or change public dataclass identity.
- Give the internal module no public exports and keep the import graph acyclic.

## TDD Sequence

1. Freeze all alias tables and invalid-name behavior.
2. Use deterministic fake constructors to verify every factory branch and
   constructor argument without fitting a model.
3. Require exact facade aliases to a not-yet-created module and retain the
   intended adoption failure.
4. Extract the four characterized functions without rewriting their bodies.
5. Run focused, neighboring BO, full-suite, docs, and package-build gates.

## TDD Evidence

- Before production edits, 64 alias and constructor-wiring tests passed and
  only the intended missing-module adoption test failed.
- The completed focused suite has 65 tests and covers all 67 statements and 24
  branches in the new module.
- Neighboring BO and public API selection: 230 tests and 49 subtests passed
  with four expected third-party warnings, including real BoTorch workflows.

## Implementation

- Added private `_bo_acquisition.py` with structural protocols for the
  parent-owned single- and multi-objective surrogate records.
- Moved both acquisition-name normalizers and both lazy acquisition factories.
- Retained all four prior parent private names as exact aliases. All moved
  function bodies are AST-identical to their prior implementations.
- Kept sample-count validation, posterior evaluation, physical-direction
  conversion, ranking, sequential selection, and public facades in the parent.
- Declared an empty module `__all__`; the module has no package-internal import.

## Structural Result

| Measure | Before | After |
| --- | ---: | ---: |
| Public BO module physical lines | 3,540 | 3,339 |
| Public BO module statements | 1,278 | 1,228 |
| Public BO module branches | 528 | 504 |
| Public BO module branch-aware coverage | 79.40% | 81.12% |
| Internal acquisition statements | - | 67 |
| Internal acquisition branches | - | 24 |
| Internal acquisition coverage | - | 100% |
| Complete public tests | 690 | 755 |
| Complete branch-aware coverage | 82.81% | 83.09% |

The parent module lost 201 physical lines, 50 statements, and 24 branches.
All 293 ordered exports and all 15 public BO function signatures remain exact.
The package now has 49 top-level modules and 67 internal import edges with no
cycle; the new acquisition module has no package-internal edge.

## Verification

- Full branch-aware suite: 755 passed, 55 subtests passed, and 9 expected
  third-party warnings.
- Ruff lint, Ruff formatting, and `git diff --check`: passed.
- Strict MkDocs build: passed.
- Offline source-distribution and wheel build: passed.

## Next Boundary

The planned BO internal-policy split is complete. Proceed to learning-curve
train-size, metric, split, and generated-column policy before changing any
learning-curve orchestration.
