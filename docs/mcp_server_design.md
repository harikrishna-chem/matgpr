# MCP Server Design

Date: 2026-09-08

Status: optional MCP entry-point scaffold, pure-Python read-only helper layer,
safe-equation preview helper, read-only prompt/resource layer, user-facing MCP
guide, unit tests, package build check, packaged MCP smoke command, and real
local command smoke test are complete. Full host UI configuration is deferred
until after the first read-only server is reviewed.

## Goal

Create an optional local Model Context Protocol server that lets AI coding
agents use `matgpr` more reliably. The server should expose package knowledge
and small, deterministic helper operations through MCP tools, resources, and
prompts while keeping the core Python package usable without MCP dependencies.

The first MCP server should help users and agents:

- discover `matgpr` capabilities,
- choose appropriate materials featurizers,
- search physics equation templates,
- validate safe custom physics equations,
- preview physics mean functions on small sample rows,
- choose validation and Bayesian-optimization workflows,
- cite and install the correct package release.

## Why MCP Helps

MCP is useful for `matgpr` because the package has many scientific workflows
that users may not remember exactly:

- An agent can ask the server which `matgpr` tool fits a dataset instead of
  guessing function names from memory.
- The server can return structured feature recommendations for composition,
  molecule SMILES, polymer SMILES, structure, processing-condition, and target
  columns.
- The server can expose physics equation metadata, required features, learned
  parameters, units, assumptions, and citation guidance.
- Safe-equation validation can happen before a notebook or app tries to train a
  physics-informed GPR model.
- Small previews can catch unit, missing-column, or non-finite-output problems
  early.
- AI agents can generate better scripts and notebooks by calling structured
  `matgpr` metadata instead of copying prose from documentation.

## Repository Decision

Start inside the public `matgpr` repository, not a separate repository.

Reasons:

- The MCP server is a thin interface over public `matgpr` APIs.
- The server should version with the package so tool behavior matches the
  installed package version.
- Tests, docs, CI, release notes, and PyPI extras can stay in one place.
- Users can install one package and get both the library and the local server.
- A separate repo would add release and compatibility overhead before there is
  a real need.

Create a separate repository later only if:

- the server becomes a hosted service instead of a local package companion,
- authentication, user accounts, or managed cloud deployment are added,
- the server needs an independent release cadence,
- the server grows beyond the public `matgpr` API surface.

The public `matgpr` MCP server should expose only open-source `matgpr`
capabilities. Downstream product-specific or deployment-specific workflows
should use their own separate MCP servers or application adapters.

## Transport Decision

Use local STDIO first.

The first server should run as a local process on the user's machine:

```bash
pip install "matgpr[mcp]"
matgpr-mcp
```

Local STDIO is the best first transport because:

- it is simple to install and debug,
- it uses the user's installed `matgpr` version,
- it avoids hosting, authentication, and cloud cost,
- it keeps user datasets local,
- it works naturally with local AI coding agents.

A Streamable HTTP server can be considered later if there is a clear public
use case for hosted or remote access.

## Proposed Package Layout

Add the MCP implementation as optional code:

```text
matgpr/
  mcp_prompts.py
  mcp_resources.py
  mcp_server.py
  mcp_smoke.py
  mcp_tools.py
tests/
  test_mcp_prompts.py
  test_mcp_resources.py
  test_mcp_smoke.py
  test_mcp_tools.py
  test_mcp_server.py
docs/
  mcp_server_design.md
  mcp_server.md
```

Add `mcp_schemas.py` later only if the helper payloads grow enough to justify a
dedicated schema module.

Add an optional dependency extra:

```toml
[project.optional-dependencies]
mcp = [
    "mcp[cli]",
]
```

Add a console entry point:

```toml
[project.scripts]
matgpr-mcp = "matgpr.mcp_server:main"
```

The regular import path must stay lightweight:

```python
import matgpr
```

This should not import the MCP SDK. Import `mcp` only inside MCP-specific
modules or the `matgpr-mcp` entry point.

## First Tool Set

Start with read-only, low-risk tools.

### `get_matgpr_info`

Return package metadata:

- installed version,
- Python version,
- docs URL,
- PyPI URL,
- GitHub URL,
- current DOI if available,
- high-level capabilities,
- optional dependency groups.

### `list_capabilities`

Return a structured capability map:

- data cleaning,
- composition featurization,
- molecule featurization,
- polymer featurization,
- structure featurization,
- standard GPR,
- physics-informed GPR,
- multitask GPR,
- multi-fidelity GPR,
- uncertainty diagnostics,
- validation,
- finite-pool Bayesian optimization,
- visualization.

### `recommend_featurizers`

Input:

- column names,
- optional dtype summary,
- one to five sample rows,
- optional target column,
- optional task type.

Output:

- likely formula/composition columns,
- likely molecule SMILES columns,
- likely polymer SMILES columns,
- likely structure columns,
- likely processing-condition columns,
- likely target columns,
- recommended `matgpr` featurizers,
- optional dependency notes,
- warnings for ambiguous or invalid columns.

The tool should use conservative heuristics and should report uncertainty. It
should not hard-code domain-specific column names beyond common patterns.

### `list_physics_equations`

Wrap the public physics-equation registry and return:

- template names,
- domains,
- required features,
- learned parameters,
- fixed constants,
- assumptions,
- units,
- when to use,
- limitations.

### `get_physics_equation`

Return one template in detail:

- equation expression,
- feature map requirements,
- learnable parameter initial guesses,
- positive-parameter constraints,
- fixed constants,
- implementation notes,
- reporting checklist.

### `validate_safe_equation`

Wrap `matgpr.safe_equations` validation.

Input:

- expression,
- variables,
- parameters,
- constants,
- allowed functions.

Output:

- valid or invalid,
- normalized equation spec if valid,
- parsed symbols,
- errors and warnings,
- implementation hints.

### `preview_safe_equation`

Evaluate a valid safe custom equation on small user-provided sample rows.

Rules:

- accept rows directly in the MCP call,
- also accept direct variable arrays for agents that already have structured
  inputs,
- do not read arbitrary files,
- cap rows and columns,
- return JSON-safe finite counts and preview values,
- report non-finite outputs and missing features.

Output:

- validation diagnostics,
- normalized equation spec,
- input mode and row-count metadata,
- physics-mean summary statistics,
- clipped physics mean values,
- optional target-minus-physics residual summary,
- warnings and implementation hints.

### `suggest_validation_workflow`

Recommend a validation protocol from dataset evidence:

- low-data repeated learning curve,
- train/test split,
- cross-validation,
- parity plot,
- uncertainty coverage,
- SHAP or feature-impact analysis when appropriate.

The tool should describe what `matgpr` functions to call, not run a large model
fit in the first MCP release.

### `suggest_bo_workflow`

Recommend finite-pool Bayesian-optimization setup:

- required measured-data fields,
- candidate-pool requirements,
- objective direction,
- uncertainty requirements,
- duplicate policy,
- feasibility constraints,
- diversity or trust-region options,
- audit/reporting outputs.

This tool should be advisory in the first release. Actual BO execution can be
added later after the read-only server is stable.

## Resources

Expose compact static resources for agent context:

- `matgpr://guide/capabilities`
- `matgpr://guide/featurization`
- `matgpr://guide/physics-informed-gpr`
- `matgpr://guide/validation`
- `matgpr://guide/bayesian-optimization`

The first implementation provides short built-in Markdown summaries plus
canonical hosted-doc paths in prompt text. Because full documentation is not
currently installed inside the wheel, do not silently bundle the full docs tree
unless package contents and PyPI size policy are reviewed.

## Prompts

Expose reusable prompts for common agent workflows:

- `plan_featurization_workflow`
- `plan_physics_informed_gpr_workflow`
- `plan_validation_workflow`
- `plan_bayesian_optimization_workflow`

Prompts should direct the agent to use `matgpr` APIs and cite relevant docs.
They should not embed downstream product-specific workflows.

## Safety And Guardrails

The first public MCP server should be conservative:

- read-only by default,
- no arbitrary Python execution,
- no shell execution,
- no destructive file writes,
- no automatic package installation,
- no network calls except optional links returned to the user,
- no training jobs or BO execution in the first release,
- row limits for preview tools,
- JSON-safe outputs only,
- clear errors for missing optional dependencies,
- no imports from downstream project packages,
- no secret, credential, or environment-variable exposure.

If later releases add modeling execution tools, they should require:

- explicit small-data limits or explicit artifact paths,
- deterministic seeds,
- timeouts,
- model-size limits,
- saved configuration summaries,
- artifact manifests,
- opt-in file writes,
- clear user approval in the MCP host.

## User Setup Examples

Codex project-level config example:

```toml
[mcp_servers.matgpr]
command = "matgpr-mcp"
args = []
enabled = true
```

Claude Code local setup example:

```bash
claude mcp add --transport stdio matgpr -- matgpr-mcp
```

Users can then ask an AI coding agent questions such as:

- "Which `matgpr` featurizer should I use for these columns?"
- "Search physics equations for diffusion or Arrhenius behavior."
- "Validate this custom PI-GPR equation before I train it."
- "Suggest a low-data validation protocol for this dataset."
- "Help me set up finite-pool BO with duplicate avoidance."

## Implementation Sequence

1. Keep this design page in the public docs and gather review feedback.
2. Add an optional `mcp` extra and `matgpr-mcp` console entry point.
3. Add pure-Python helper functions for package info, capability summaries,
   featurizer recommendations, physics-template lookup, safe-equation
   validation, and safe-equation preview.
4. Add unit tests for the helper functions without requiring an MCP runtime.
5. Add read-only prompt templates and compact static guide resources.
6. Add the MCP server wrapper using the official Python MCP SDK.
7. Add MCP smoke tests that verify the server object exposes the expected
   tools, prompts, and resources when the optional dependency is installed.
8. Maintain `docs/mcp_server.md` with install, Codex, Claude Code, and
   troubleshooting instructions.
9. Run local validation:

```bash
python -m ruff check matgpr tests scripts
python -m pytest
python -m mkdocs build --strict
python -m build
python -m twine check dist/*
```

10. Add a packaged `matgpr-mcp-smoke` command that verifies initialize,
   list-tools, list-prompts, list-resources, a lightweight package-info tool
   call, one prompt fetch, and one resource read through the real MCP SDK.
11. Test with at least one local MCP host or MCP Inspector before announcing
    host-specific setup as fully validated.
12. Release as a future patch version only after the MCP server is stable.

## Acceptance Criteria

- `pip install matgpr` remains lightweight and does not require the MCP SDK.
- `pip install "matgpr[mcp]"` installs the server dependency.
- `matgpr-mcp` starts successfully as a local STDIO server.
- `matgpr-mcp-smoke` verifies a real MCP client can initialize the server and
  list the expected tools, prompts, and resources.
- The server exposes a small set of documented read-only tools, prompt
  templates, and static guide resources.
- Tool outputs are deterministic and JSON-safe.
- Invalid equations, invalid sample rows, missing features, and missing
  optional dependencies produce clear errors.
- Tests cover helper logic and server tool registration.
- MkDocs includes the MCP user guide and design page.
- No downstream project-specific logic is exposed.

## Open Questions

- Should compact docs resources be bundled as package data, or should the MCP
  server return hosted documentation links only?
- Should any modeling execution tools be added before `v0.3.0`, or should the
  `v0.2.x` line remain read-only/advisory for MCP?
- Should a hosted MCP server ever be part of the public `matgpr` project, or
  should public `matgpr` stay local-only?

## Recommended First Implementation

Build the first MCP server in `matgpr` as a local STDIO, read-only, optional
extra with these tools:

- `get_matgpr_info`
- `list_capabilities`
- `recommend_featurizers`
- `list_physics_equations`
- `get_physics_equation`
- `validate_safe_equation`
- `preview_safe_equation`
- `suggest_validation_workflow`
- `suggest_bo_workflow`

Add prompt templates for:

- `plan_featurization_workflow`
- `plan_physics_informed_gpr_workflow`
- `plan_validation_workflow`
- `plan_bayesian_optimization_workflow`

Add static guide resources for:

- `matgpr://guide/capabilities`
- `matgpr://guide/featurization`
- `matgpr://guide/physics-informed-gpr`
- `matgpr://guide/validation`
- `matgpr://guide/bayesian-optimization`

Defer model fitting, file writes, and hosted HTTP transport until the local
advisory server is stable.

## References

- MCP tools specification: <https://modelcontextprotocol.io/specification/2025-06-18/server/tools>
- MCP resources specification: <https://modelcontextprotocol.io/specification/2025-06-18/server/resources>
- MCP Python SDK: <https://py.sdk.modelcontextprotocol.io/>
- OpenAI Codex MCP documentation: <https://learn.chatgpt.com/docs/extend/mcp>
- Claude Code MCP documentation: <https://code.claude.com/docs/en/mcp>
