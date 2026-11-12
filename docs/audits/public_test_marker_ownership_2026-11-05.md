# Public Test-Marker Ownership Audit

**Date:** 2026-11-05
**Repository:** `matgpr`
**Scope:** Step 10, Slice 2

## Purpose

Give every collected public test one explicit primary owner before CI begins
selecting tests by marker. Primary markers describe the behavior under test:

- `unit`: one focused function or class without workflow orchestration.
- `contract`: public API, schema, serialization, or compatibility behavior.
- `component`: one in-memory package workflow or service component.
- `integration`: behavior crossing modules or an external package boundary.
- `scientific`: numerical, statistical, or physical correctness.

The `smoke`, `slow`, and `optional` markers remain orthogonal modifiers and do
not replace primary ownership.

## TDD Sequence

1. Added a collection guard requiring exactly one primary marker.
2. Confirmed collection failed because 647 tests lacked an owner.
3. Added module-level primary markers to 43 uniformly owned test modules.
4. Preserved existing per-test markers in the genuinely mixed modules.
5. Added `optional` to the two tests that execute the optional matminer backend.
6. Confirmed collection succeeds and direct guard probes reject both missing
   and conflicting primary markers.

The guard reports at most 20 individual violations and the number of remaining
violations, keeping accidental large regressions readable in CI logs.

## Ownership Result

| Primary tier | Tests | Local selected runtime |
| --- | ---: | ---: |
| Unit | 346 | 2.70 s |
| Contract | 90 | 2.00 s |
| Component | 217 | 4.20 s |
| Integration | 12 | 1.84 s |
| Scientific | 209 | 3.50 s |
| **Total** | **874** | - |

The primary counts are disjoint and sum to the complete 874-test collection.
Modifier ownership is 12 `smoke` tests and 2 `optional` tests; no ordinary
test is currently marked `slow`.

## Assignment Rationale

- Pure BO policy helpers, data cleaning, fingerprint helpers, PCA,
  target-transform mechanics, and learning-curve policy are unit-owned.
- BO request records, MCP registrations, optional-dependency behavior,
  experiment-log schema behavior, public exports, and safe-equation security
  boundaries are contract-owned where their primary purpose is interface
  stability.
- BO workflows, candidate generation, estimators, featurizer pipelines,
  reporting, safe-equation workflows, validation, and plotting are
  component-owned.
- MCP executable smoke behavior remains integration-owned with the `smoke`
  modifier.
- GPR variants, kernels, metrics, uncertainty, physics equations and
  constraints, noise models, and numerical BO checks are scientific-owned.

## Commands

```bash
python -m pytest --collect-only -q
python -m pytest -q -m unit
python -m pytest -q -m contract
python -m pytest -q -m component
python -m pytest -q -m integration
python -m pytest -q -m scientific
python -m pytest -q -m smoke
python -m pytest -q -m optional
```

## Decision

Marker ownership is now safe for CI selection. This slice does not change any
workflow selection, dependency, coverage threshold, production behavior, or
public API. The next independently reviewable slice may establish one
canonical full-coverage job and fast unit/contract compatibility jobs on the
other supported Python versions.

## Verification

- `ruff check matgpr tests scripts`: passed.
- `ruff format --check matgpr tests scripts`: 113 files already formatted.
- Full pytest with branch coverage: 874 passed in 9.67 seconds.
- Combined branch-aware coverage: 83.81% against the unchanged 78% floor.
- Strict MkDocs build: passed.
- Source distribution and wheel build: passed.
- Twine validation of both distributions: passed.
- Isolated no-dependency wheel smoke: passed from `site-packages` at version
  `0.2.0`.
- `git diff --check`: passed.
