# Public CI Workflow Split Audit

**Date:** 2026-11-05
**Repository:** `matgpr`
**Scope:** Step 10, Slice 3

## Purpose

Stop repeating every test, lint, format, build, and wheel-install check on all
three supported Python versions while preserving full scientific validation,
supported-version compatibility, minimum-dependency coverage, and established
branch-protection check names.

## Workflow Topology

| Job | Python | Responsibility |
| --- | --- | --- |
| `Package quality` | 3.11 | Ruff lint/format, build, Twine, isolated no-dependency wheel smoke |
| `Tests (3.11)` | 3.11 | All 874 tests with branch coverage and the unchanged 78% floor |
| `Tests (3.10)` | 3.10 | The 436 unit-or-contract compatibility tests |
| `Tests (3.12)` | 3.12 | The 436 unit-or-contract compatibility tests |
| `Minimum supported dependencies` | 3.10 | Full suite at declared lower dependency bounds |
| `Notebook smoke checks` | 3.11 | Five reduced example notebook executions |

The three existing `Tests (3.x)` display names are preserved so repositories
using those names as required status checks do not need an avoidable branch
protection migration.

## Efficiency Decisions

- Package quality installs only `build`, `ruff`, and `twine`; it does not
  install the full PyTorch, GPyTorch, BoTorch, RDKit, and pymatgen runtime.
- Test jobs install `matgpr[dev]` without the notebook-only `examples` extra.
- Notebook smoke remains the only CI job that installs `matgpr[dev,examples]`.
- Notebook smoke now waits for canonical tests and package quality, rather
  than waiting for the complete compatibility matrix.
- The minimum-dependency job remains a full-suite job and retains its exact
  lower-bound pins.

## Preserved Gates

- All primary test tiers still run on every pull request through the canonical
  Python 3.11 job.
- Branch-aware coverage retains the 78% failure floor.
- Python 3.10 and 3.12 retain focused API and behavior compatibility checks.
- Distribution build, metadata validation, and isolated wheel import remain
  blocking checks.
- The five notebook smoke checks remain blocking.

## Local Verification

- Workflow YAML parsed and the job matrix/dependency assertions passed.
- Compatibility command: 436 passed, 438 deselected, in 3.61 seconds.
- Canonical command: 874 passed at 83.81% branch-aware coverage in 9.01
  seconds.
- Ruff lint and format checks passed.
- Source distribution and wheel build passed.
- Twine validated both distributions.
- Isolated no-dependency wheel smoke imported `matgpr 0.2.0` from
  `site-packages`.
- All five notebook smoke executions passed.

Python 3.10 and 3.12 are not installed in the local development environment,
so the first GitHub Actions run remains the authoritative supported-version
execution check.

## Decision

The first workflow split is ready for remote validation. No production code,
public API, dependency declaration, minimum-dependency pin, or coverage
threshold changed. The next CI hardening slice should install the real
`matgpr[mcp]` extra and execute the packaged MCP client/server smoke path.
