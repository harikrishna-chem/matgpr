# Step 8 Public Structural Inventory

**Date:** 2026-11-04
**Plan step:** Step 8, Slice 1
**Status:** Completed

## Scope

Characterize public `matgpr` before targeted cleanup. This slice measures
structure, duplication, dependency direction, test ownership, coverage, and
released API constraints. It does not change production behavior.

## Review Matrix

1. Full branch-aware test and warning baseline.
2. Package exports, public signatures, and facade identity.
3. Module, class, and function size hotspots.
4. Top-level imports, cycles, and dependency direction.
5. Exact AST-normalized duplicate function bodies.
6. Direct test ownership and hotspot coverage.
7. BO, validation/conversion, learning-curve, and safe-equation boundaries.

## Decision Rule

- Preserve every released import and signature.
- Consolidate only behavior proven semantically identical by tests.
- Prefer a small internal extraction with one clear owner over a generic helper
  layer.
- Keep trivial protocol methods and domain-specific validation local.
- Select at most one candidate for the next TDD slice; make no production code
  change during this inventory.

## Current Quality Baseline

| Measure | Result |
| --- | ---: |
| Tests | 582 passed |
| Expected third-party warnings | 9 |
| Runtime | 16.37 seconds |
| Statements | 12,314 |
| Branches | 4,484 |
| Combined branch-aware coverage | 81.19% |
| Released package exports | 293 |

The current ordered export list is an exact match to the frozen baseline. Its
newline-delimited SHA-256 remains
`b64f5af236df40169a73e93bb6031e3f7afb85ea866c8531746edd7c8eb46204`.
No export was added, removed, or reordered.

## Structural Findings

- The package contains 45 top-level Python modules and 57 top-level internal
  import edges. The import graph has no cycle.
- The three largest modules are large because they own several related public
  workflows, not because of one unbounded function. Their largest individual
  definitions are 163, 157, and 97 lines, respectively.
- Each major hotspot has a direct test owner. The shared private validation
  module is exercised indirectly but lacks focused ownership for several error
  branches.

| Module | Physical lines | Statements | Branches | Coverage | Direct owner |
| --- | ---: | ---: | ---: | ---: | --- |
| `bayesian_optimization.py` | 4,514 | 1,697 | 758 | 74.83% | BO API and scientific tests |
| `validation.py` | 3,128 | 1,087 | 418 | 73.69% | `test_validation.py` |
| `safe_equations.py` | 2,015 | 893 | 346 | 82.41% | `test_safe_equations.py` |
| `multifidelity_gpr.py` | 1,647 | 664 | 176 | 85.00% | multifidelity tests |
| `mcp_tools.py` | 1,313 | 449 | 178 | 80.54% | MCP contract/tool tests |
| `candidate_generation.py` | 1,271 | 551 | 230 | 81.82% | candidate-generation tests |

Largest definitions relevant to future cleanup include:

- `select_sequential_multi_objective_batch`: 163 lines.
- `cokriging_learning_curve`: 157 lines.
- `suggest_next_experiments`: 142 lines.
- `multifidelity_learning_curve`: 138 lines.
- `learning_curve`: 137 lines.
- `_validate_expression_node`: 97 lines.

These sizes justify later boundary review, but not immediate movement without
additional characterization at the intended extraction seam.

## Exact-Duplicate Classification

The strict AST scan found 19 groups among 1,204 functions and methods.

Four groups remain plausible cleanup candidates:

1. One-dimensional finite numeric conversion: the canonical
   `_validation.to_1d_finite` plus exact copies in reporting, validation, and
   visualization.
2. Dataframe validation in BO audit and candidate generation.
3. Objective-name slugging in BO and multi-objective utilities.
4. Numeric-matrix validation in derivative and virtual-observation workflows.

The other 15 groups are intentionally short protocol methods or
contract-sensitive behavior: validation-result projections, target-transform
methods, estimator distribution methods, featurizer feature-name methods,
kernel properties, observation counts, and report-frame properties. Sharing
them would add indirection without establishing a better owner.

## Selected Next Slice

Select the one-dimensional finite-array helper for Step 8, Slice 2.

Reasons:

- The four implementations are exactly identical in signature AST and body.
- `_validation.to_1d_finite` is already the canonical low-level owner and is
  used by scientific modules.
- The three copies are private helpers, so no released import changes.
- Existing workflow tests protect callers, while focused tests can first lock
  success, empty-input, non-finite-input, shape-flattening, dtype, and exact
  error-message behavior.
- Local private names can remain as direct aliases where that improves review
  clarity and compatibility.

## Deferred Candidates

- Split saved-prediction acquisition and candidate-selection internals from
  `bayesian_optimization.py` only after seam-specific tests cover acquisition
  intent, uncertainty requirements, constraints, stable ranking, and output
  columns.
- Split learning-curve request normalization from execution only after exact
  train-size, seed, split, metric, and estimator-cloning behavior is frozen.
- Separate safe-equation validation/evaluation only in a security-focused
  slice; parser allowlists, numerical semantics, and error records must remain
  together until that boundary is directly characterized.

## Decision

Step 8, Slice 1 is complete with no production-code change. Proceed with the
focused shared finite-vector validation slice before attempting any large
module extraction.

## Verification

- Full branch-aware suite: 582 passed, 9 expected third-party warnings,
  81.19% coverage.
- Ordered 293-name public export inventory: exact baseline match.
- Top-level package import graph: no cycle.
- Ruff lint, Ruff formatting, and `git diff --check`: passed.
- Strict MkDocs build and offline source/wheel package build: passed.
