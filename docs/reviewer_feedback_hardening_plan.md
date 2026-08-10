# Reviewer Feedback Hardening Plan

Date: 2026-08-10

## Scope

Address actionable independent-review comments for the public `matgpr` package
without changing existing scientific defaults or touching draft BO notebooks.

## Ownership Boundary

- `matgpr` owns reusable logging, metrics, uncertainty, packaging, and release
  documentation.
- Draft BO notebooks remain untracked until they are intentionally promoted to
  public examples with cards, smoke tests, and scrubbed outputs.
- GenMatics app integration work is out of scope for this pass.

## Ordered Steps

1. Harden `log_experiment_result` so schema changes do not corrupt CSV logs.
2. Add JSON-safe companion helpers for metrics and uncertainty diagnostics.
3. Add tests for schema-evolving experiment logs and strict JSON serialization.
4. Update user/API documentation to explain safe logging and JSON-safe helpers.
5. Add packaging manifest policy for citation/changelog files in source
   distributions.
6. Tighten release commands so stale `dist/*` artifacts are not uploaded by
   accident.
7. Update stale changelog wording about documentation deployment.
8. Append a concise implementation reply to the reviewer handoff file.

## Acceptance Checks

- CSV experiment logs round-trip after same-schema, extra-column,
  missing-column, and reordered-column appends.
- `json.dumps(..., allow_nan=False)` succeeds for one-sample/constant metric
  edge cases and constant-uncertainty diagnostics.
- Release docs use clean builds and exact release artifact globs.
- Source distribution policy includes `CITATION.cff` and `CHANGELOG.md`.
- Existing public APIs remain backward compatible.

## Verification Commands

```bash
python -m ruff check matgpr tests scripts
python -m pytest
python -m mkdocs build --strict
python -m build
python -m twine check dist/matgpr-0.1.1*
```

## Handoff Notes

Report back to `/Users/hari/projects/genmatics/reviewer/projects/matgpr.md`
with the APIs added, tests run, docs changed, and any deferred items.
