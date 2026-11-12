# Public Coverage-Floor Ratchet

**Date:** 2026-11-06
**Repository:** `matgpr`
**Scope:** Step 10, Slice 11
**Branch:** `agent/matgpr/tdd-baseline`

## Scope

- Raised the combined branch-aware coverage floor from 78% to 79%.
- Aligned `pyproject.toml` and the canonical Python 3.11 CI command.
- Preserved every test selection, supported-version compatibility job,
  minimum-dependency job, MCP smoke, notebook smoke, strict-docs gate, package
  build, and installed-wheel check.
- Made no production-code, test-behavior, public-API, dependency, or workflow
  selection changes.

## Safety Margin

- Accepted local baseline: 83.81% combined branch-aware coverage.
- Enforced floor: 79.00%.
- Remaining headroom: 4.81 percentage points.

The floor remains deliberately below the accepted baseline. Any later increase
requires a separate review after remote CI confirms this one-point ratchet.

## Verification

- Primary tiers passed independently: 346 unit, 90 contract, 217 component,
  12 integration, and 209 scientific tests.
- Canonical gate: 874 tests passed at 83.81% combined branch-aware coverage
  with the new 79% floor enforced.
- Ruff lint passed; all 113 checked files were already formatted.
- Workflow assertions confirmed the unchanged package-quality, MCP,
  compatibility, minimum-dependency, and notebook-smoke jobs.
- Strict MkDocs built successfully. Existing informational output about
  Material for MkDocs and intentionally unlisted audit pages remains unchanged.
- Source distribution and wheel built successfully; Twine accepted both.
- A clean environment imported `matgpr 0.2.0` from the installed wheel.
- `git diff --check` passed.

The first remote workflow remains the final evidence needed before considering
another public-package coverage increase.

## Closeout Timeout Hardening

The cross-repository closeout found that ordinary public workflows still used
GitHub's default six-hour job limit. The approved bounded policy is now:

- 15 minutes for package quality, MCP smoke, documentation build, publishing
  build, and scheduled PI-GPR convergence;
- 20 minutes for canonical tests, supported-version compatibility tests, and
  notebook smoke;
- 30 minutes for the Python 3.10 minimum-dependency suite; and
- 10 minutes for documentation deployment and each trusted-publishing job.

Every executable public workflow job is now bounded. The intentionally disabled
Jekyll sample remains unchanged because it cannot execute.
