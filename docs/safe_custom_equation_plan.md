# Safe Custom Equation Plan

Date: 2026-08-13

## Goal

Add reusable, public `matgpr` infrastructure for safe user-defined
physics-informed mean functions. This should let downstream tools validate,
preview, and eventually train PI-GPR models from structured equation specs
without executing arbitrary Python.

## Ownership Boundary

`matgpr` should own general scientific infrastructure:

- JSON-safe custom equation specs,
- allowlisted expression parsing and validation,
- NumPy preview evaluation,
- torch-compatible evaluation for GPyTorch mean functions,
- finite-output validation on sample data,
- conversion into physics-informed GPR mean-function objects,
- tests and public documentation.

`matgpr` should not own GenMatics project persistence, user ownership,
database records, job orchestration, artifact storage, or portal UI.

## Non-Negotiable Safety Rules

- Do not evaluate arbitrary Python.
- Do not accept imports, lambdas, callbacks, attributes, comprehensions,
  indexing, mutation, assignments, or user-defined functions.
- Treat equation expressions as structured data.
- Allow only explicit variables, parameters, constants, and allowlisted math
  functions.
- Validate syntax, symbols, parameter bounds, and finite outputs before
  fitting.
- Return JSON-safe payloads with clear warnings/errors.

## Proposed Public API

Module: `matgpr.safe_equations`

Records:

- `SafeEquationSpec`
- `SafeEquationVariable`
- `SafeEquationParameter`
- `SafeEquationConstant`
- `SafeEquationValidationResult`
- `SafeEquationEvaluationResult`
- `SafeEquationMeanPreview`

Functions:

- `validate_safe_equation_spec(...)`
- `evaluate_safe_equation_numpy(...)`
- `evaluate_safe_equation_torch(...)`
- `preview_safe_equation_mean(...)`
- `build_safe_equation_mean_function(...)`
- `safe_equation_schema_snapshot(...)`

The API names can be adjusted during implementation if existing `matgpr`
patterns suggest a better fit.

## Expression Grammar

Initial grammar should be intentionally small:

- operators: `+`, `-`, `*`, `/`, `**`
- unary operators: `+`, `-`
- parentheses through normal expression grouping
- numeric constants
- variable names declared in the spec
- parameter names declared in the spec
- constant names declared in the spec
- functions: `exp`, `log`, `sqrt`, `abs`, `min`, `max`, `pow`

Explicitly reject:

- `eval`, `exec`, `open`, imports, attributes, subscripts, comprehensions,
  lambdas, conditionals, assignments, comparisons, boolean operators, and any
  unknown function name.

Piecewise expressions, comparisons, and richer unit algebra should be deferred
until the first safe core is stable.

## Step-By-Step Implementation

1. Add schema dataclasses and `.to_dict()` methods.
   - Validate names with a conservative identifier rule.
   - Enforce unique names across variables, parameters, constants, and allowed
     functions.
   - Store units, descriptions, assumptions, validity limits, initial values,
     and bounds as JSON-safe metadata.
   - Status: implemented in `matgpr.safe_equations`.

2. Add expression parser and validator.
   - Parse with Python `ast.parse(..., mode="eval")`.
   - Walk the AST manually.
   - Accept only allowlisted nodes, operators, names, and functions.
   - Return structured syntax/symbol/function/bounds errors.
   - Status: implemented in `matgpr.safe_equations`.

3. Add NumPy evaluator.
   - Implement a recursive AST interpreter.
   - Map allowed functions to NumPy operations.
   - Accept dataframe/array feature inputs through explicit variable mapping.
   - Return values plus finite/non-finite counts and warnings.
   - Status: implemented as `evaluate_safe_equation_numpy(...)`.

4. Add torch evaluator.
   - Reuse the validated AST.
   - Map allowed functions to torch operations.
   - Preserve differentiability for learnable parameters.
   - Add tests that gradients flow through parameters.
   - Status: implemented as `evaluate_safe_equation_torch(...)` and
     `initialize_safe_equation_torch_parameters(...)`.

5. Add finite-output preview helpers.
   - Validate missing variables and non-finite input rows.
   - Evaluate the physics mean on sample data.
   - Optionally compute target residual summaries when a target is supplied.
   - Return JSON-safe summary statistics.
   - Status: implemented as `preview_safe_equation_mean(...)`.

6. Add PI-GPR mean-function conversion.
   - Convert a validated spec into a callable mean compatible with
     `PhysicsInformedMean` or a new small adapter class.
   - Support fixed parameters and learnable parameter initial values.
   - Support parameter bounds metadata first; enforce fitting bounds only when
     compatible with the existing GPyTorch training path.
   - Status: implemented as `SafeEquationMeanFunction` and
     `build_safe_equation_mean_function(...)`.

7. Add tests.
   - Valid simple equations: Arrhenius-like, power law, Hall-Petch-like.
   - Invalid syntax.
   - Missing variables.
   - Unsupported functions.
   - Injection attempts.
   - Invalid parameter bounds.
   - Non-finite output detection.
   - Strict JSON serialization.
   - Torch evaluation and gradient flow.

8. Add documentation.
   - User guide section for safe custom equations.
   - API docs page.
   - Safety limitations.
   - Example spec and preview workflow.
   - Clear note that GenMatics app should use `genmatics-matgpr` wrappers, not
     call lower-level evaluators directly.

## Acceptance Checks

- No user-provided expression can execute Python code.
- Valid specs can be serialized with `json.dumps(..., allow_nan=False)`.
- Invalid specs return structured errors before fitting.
- NumPy and torch evaluators agree on representative examples.
- Non-finite physics-mean outputs are detected before training.
- A validated safe equation can be converted into a PI-GPR mean-function path.
- Existing curated equation APIs remain backward compatible.

## Verification Commands

```bash
python -m ruff check matgpr tests scripts
python -m pytest
python -m mkdocs build --strict
```

## Handoff To genmatics-matgpr

After the public `matgpr.safe_equations` core is implemented and tested,
`genmatics-matgpr` should add platform-facing service methods for custom
equation validation, form specs, physics-mean previews, comparison jobs,
model-run records, and artifact manifests.
