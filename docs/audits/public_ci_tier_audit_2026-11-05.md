# Public CI-Tier Audit

**Date:** 2026-11-05
**Plan step:** Step 10, Slice 1
**Status:** Completed

## Scope

Inventory the public repository's GitHub Actions jobs, supported Python and
dependency claims, pytest marker ownership, local tier runtimes, and recent CI
durations before changing workflow behavior.

The branch HEAD and `origin/main` both point to `5743e871`, while the complete
TDD quality work remains uncommitted. Recent GitHub Actions runs therefore
measure the pre-TDD remote tree, not the current 874-test working tree.

## Workflow Inventory

| Workflow/job | Runtime/dependency surface | Current responsibility |
| --- | --- | --- |
| CI: Tests 3.10, 3.11, 3.12 | `.[dev,examples]` on each Python | Full tests, lint, format, and three package builds |
| CI: Minimum supported dependencies | Python 3.10 with every declared lower bound pinned | Full source-tree suite without coverage |
| CI: Notebook smoke | Python 3.11, `.[dev,examples]`; waits for the whole test matrix | Two reduced GPR notebooks and three real BO notebooks |
| Docs: Build documentation | Python 3.11, `.[docs,examples]` | Strict MkDocs build on pushes, pull requests, and manual dispatch |
| Docs: Deploy documentation | GitHub Pages trusted deployment | Deploys only for non-pull-request events |
| Publish: Build distributions | Manual target/version inputs | Version/tag validation, sdist/wheel build, Twine check, artifact upload |
| Publish: TestPyPI/PyPI | Trusted publishing environments | Publishes the validated artifact to the selected index |
| Disabled Jekyll sample | Manual-only explanatory job | No deployment responsibility; retained as a disabled sample |

The accumulated working-tree CI change already strengthens the test matrix
with branch coverage, a 78% floor, and an isolated no-dependency wheel smoke.
Remote `main` still has line-only coverage and no wheel smoke until this branch
is reviewed and merged.

## Missing Explicit Tiers

- No real MCP-extra installation job exists. Unit/contract tests use package
  helpers or fake MCP modules; CI does not install `matgpr[mcp]` and launch the
  installed server through the public smoke client.
- No `optional` or `slow` tests are selected because no collected tests carry
  those markers.
- Mutation testing is correctly manual and scoped to target transforms.
- The PI-GPR convergence check is correctly manual pending a scheduled-tier
  decision.
- Build, lint, format, and the full suite are repeated across all three Python
  matrix jobs. In the working tree, wheel smoke would also run three times.
- Notebook smoke cannot start until all three matrix jobs finish, even though
  it needs only one canonical tested environment.

## Marker Ownership

Pytest collected 874 tests with this ownership:

| Marker | Collected tests |
| --- | ---: |
| `unit` | 87 |
| `contract` | 19 |
| `component` | 2 |
| `scientific` | 107 |
| `integration` | 12 |
| `smoke` | 12 |
| `slow` | 0 |
| `optional` | 0 |
| Unclassified | 647 |

The 12 integration tests are also marked smoke. Only 227 unique tests are
classified, leaving 647 tests across 43 files unclassified. A marker-selected
compatibility matrix would silently omit 74.0% of the suite and is therefore
not safe.

The intended taxonomy should use exactly one primary ownership marker from
`unit`, `contract`, `component`, `integration`, or `scientific`. The `smoke`,
`slow`, and `optional` markers are orthogonal modifiers and may overlap a
primary marker.

## Duration Evidence

Local pytest-reported durations on the current working tree were:

| Selection | Tests | Duration |
| --- | ---: | ---: |
| Full suite with branch coverage | 874 | 10.01 s |
| `unit` | 87 | 2.18 s |
| `contract` | 19 | 1.82 s |
| `component` | 2 | 1.82 s |
| `scientific` | 107 | 3.46 s |
| `integration or smoke` | 12 | 1.85 s |
| Unclassified remainder | 647 | 4.44 s |

Collection and heavy-library import time dominate the small selections, so
splitting tests into many separate jobs would not automatically make CI
cheaper.

The latest successful remote CI run on 2026-10-17 completed in 7 minutes
5 seconds. Its individual jobs took approximately:

- Python 3.10 tests: 2 minutes 28 seconds;
- Python 3.11 tests: 2 minutes 41 seconds;
- Python 3.12 tests: 2 minutes 54 seconds;
- minimum dependencies: 1 minute 59 seconds; and
- notebook smoke: 4 minutes 3 seconds, including 1 minute 50 seconds of
  notebook execution after the matrix completed.

A nearby pull-request run completed in 6 minutes 22 seconds. Across these
runs, dependency installation consumed roughly 80-105 seconds per job, while
test steps consumed 15-48 seconds. The largest wall-clock opportunity is job
structure and dependency reuse, not weakening scientific coverage.

## Safe Ratchet Order

1. Complete primary marker ownership for all 874 tests and add a collection
   guard that rejects unclassified tests.
2. Preserve one canonical Python 3.11 full suite with branch coverage and the
   78% floor.
3. After marker ownership is complete, change Python 3.10 and 3.12 to fast
   compatibility selections rather than full scientific duplication.
4. Run lint, format, package build, Twine metadata checks, and isolated wheel
   smoke once in a canonical package-quality job.
5. Let notebook smoke depend on that canonical job rather than the entire
   compatibility matrix.
6. Add a dedicated Python 3.11 MCP job that installs the `mcp` extra and runs
   the installed entrypoints against the real optional dependency.
7. Keep minimum-dependency coverage explicit; reduce it only after its own
   selected contract is documented and measured.
8. Consider scheduled/manual execution for the PI-GPR convergence script;
   keep mutation testing manual unless a future bounded pilot justifies more.

## Decision

Do not change the test matrix yet. Marker ownership is the prerequisite for a
safe compatibility/full-suite split. Step 10, Slice 2 will classify every
collected test under one primary owner, retain orthogonal modifiers, and add a
collection guard. CI workflow restructuring follows only after that guard is
green.
