# Public Safe-Equation Security Review

**Date:** 2026-11-05
**Plan step:** Step 8, Slice 8
**Status:** Completed

## Scope

Characterize the public safe-equation parser, NumPy interpreter, torch
interpreter, and PI-GPR mean adapter before deciding whether any structural
extraction is justified.

The review covers:

- the AST syntax, operator, symbol, function, and constant allowlists;
- injection-shaped expressions and forbidden Python constructs;
- bounded expression length, AST node count, and AST depth;
- NumPy/torch agreement across every allowlisted operation;
- gradient flow through variables and learnable parameters;
- scalar-output row expansion; and
- finite-output enforcement before torch-backed GPR training.

## Non-Goals

- Do not add richer expression syntax or user-defined functions.
- Do not change the public equation records or function signatures.
- Do not move portal workflow or proprietary equation policy into `matgpr`.
- Do not split the module merely to reduce line count. Security policy must
  retain one clear owner and an acyclic dependency direction.

## TDD Decision Rule

Production changes are allowed only for a demonstrated security or numerical
contract gap. A module extraction is allowed only if the characterized parser
and evaluator contracts can move without duplicating public validation records
or introducing a facade dependency cycle.

## TDD Evidence

- The first adversarial run had 22 passing tests and eight failures. Seven
  failures demonstrated missing expression-complexity or direct torch finite
  guards; the eighth confirmed that the existing PI-GPR mean adapter already
  rejected non-finite values with its own message.
- Boundary probes then exposed unstructured overflow from oversized integer
  literals/defaults, malformed Unicode, and cyclic or deeply nested metadata.
- The completed security suite has 39 tests. Together with the original
  safe-equation suite, 71 tests and 6 subtests pass with 86.4% branch-aware
  coverage of `safe_equations.py`.

## Hardening Changes

- Expressions are limited to 4,096 characters, 256 AST nodes, and AST depth
  32 before recursive validation or evaluation.
- Parser recursion and malformed Unicode now return structured validation
  issues instead of escaping as runtime exceptions.
- Oversized numeric literals, defaults, bounds, and metadata values now follow
  the existing finite-value error contract.
- Strict-JSON metadata rejects cyclic references and nesting deeper than 32
  levels.
- Direct torch evaluation rejects non-finite output tensors before they can
  reach the existing PI-GPR mean guard.
- NumPy and torch parity is characterized across every allowlisted operator and
  function, scalar row expansion, variable gradients, and parameter gradients.
- `safe_equation_schema_snapshot()` now publishes the enforced expression and
  metadata limits without changing equation-record fields.

## Architecture Decision

Do not extract the parser or evaluators in this slice.

The parser consumes `SafeEquationSpec` and returns
`SafeEquationValidationIssue`, while both evaluators share the same public
records, symbol rules, finite conversion, and interpreter grammar. Moving the
parser alone would require importing those records back from the public facade
or duplicating them in a private module. Either choice would worsen dependency
direction or create two security-policy owners. The module is large but
cohesive at this boundary; the security improvements are more valuable than a
line-count-only split.

## Compatibility

- All public safe-equation function signatures remain exact.
- The ordered 293-name package export inventory and frozen digest remain exact.
- The schema-snapshot additions are additive; existing record fields and
  schema version remain unchanged.
- The package still has 50 top-level modules and 68 internal import edges with
  no cycle.

## Verification

- Full branch-aware suite: 849 tests and 56 subtests passed at 83.62% coverage,
  with 9 expected third-party warnings.
- Ruff lint, Ruff formatting, and `git diff --check`: passed.
- Strict MkDocs, offline package build, Twine metadata, and wheel contents are
  verified in the final quality gate.

## Next Boundary

Perform a Step 8 structural closeout audit. Remeasure module sizes,
duplication, import direction, exports, signatures, and test ownership before
marking public targeted cleanup complete and moving to Step 9 test-quality
work.
