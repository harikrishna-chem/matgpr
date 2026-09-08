# MCP Server

`matgpr` includes an optional local Model Context Protocol server for AI coding
agents. The server exposes public `matgpr` package knowledge as structured,
read-only tools so agents can choose featurizers, inspect physics-equation
templates, validate safe custom equations, and design validation or
Bayesian-optimization workflows without guessing API names.

This server is a companion to the open-source `matgpr` library. It does not
expose downstream project-specific workflows.

## Current Status

The MCP server is currently available on the development branch and is planned
for a future `v0.2.x` release. Until that release is published, install from a
local checkout when testing the MCP server.

The first MCP version is intentionally conservative:

- local STDIO transport,
- read-only tools,
- JSON-safe outputs,
- no model fitting,
- no Bayesian-optimization execution,
- no shell execution,
- no arbitrary Python execution,
- no file writes,
- no automatic package installation,
- no imports from downstream project packages.

## Install

For a released version that includes MCP support:

```bash
python -m pip install "matgpr[mcp]"
```

For the current development branch:

```bash
git clone https://github.com/harikrishna-chem/matgpr.git
cd matgpr
python -m pip install -e ".[mcp]"
```

Check that the command is available:

```bash
matgpr-mcp
```

The command starts a local STDIO MCP server. Most users will not run it
directly; their MCP host will start it when needed.

## Configure Codex

Codex can run local STDIO MCP servers from its `config.toml` file. Add this to
your Codex MCP configuration:

```toml
[mcp_servers.matgpr]
command = "matgpr-mcp"
args = []
enabled = true
```

For project-scoped setup, place the configuration in `.codex/config.toml` for a
trusted project. For user-wide setup, use the default Codex configuration file.

You can also add the server from the Codex CLI:

```bash
codex mcp add matgpr -- matgpr-mcp
codex mcp list
```

## Configure Claude Code

Claude Code can also launch local STDIO MCP servers. Add `matgpr` with:

```bash
claude mcp add --transport stdio matgpr -- matgpr-mcp
```

The double dash separates Claude Code's MCP options from the command used to
start the server.

## Tool Overview

The initial server exposes read-only helper tools.

| Tool | Purpose |
| --- | --- |
| `get_matgpr_info` | Returns version, install, documentation, DOI, and capability metadata. |
| `list_capabilities` | Lists public `matgpr` workflows and the main APIs for each. |
| `recommend_featurizers` | Suggests composition, molecule SMILES, polymer SMILES, structure, numeric, and target-column handling from schema evidence. |
| `list_physics_equations` | Lists built-in physics-equation templates, optionally filtered by query, application, tag, or required features. |
| `get_physics_equation` | Returns detailed metadata for one physics-equation template. |
| `validate_safe_equation` | Validates a safe custom equation spec and returns schema-level diagnostics. |
| `suggest_validation_workflow` | Recommends learning-curve, train/test, cross-validation, and uncertainty-diagnostic protocols. |
| `suggest_bo_workflow` | Recommends a finite-pool Bayesian-optimization setup without executing BO. |

## Example Agent Prompts

After connecting the server, useful prompts include:

```text
Ask matgpr which featurizer I should use for these columns:
formula, polymer_smiles, solvent_smiles, temperature_C, diffusivity
```

```text
Ask matgpr to list physics equations related to diffusion.
```

```text
Ask matgpr to validate this safe custom equation before I use it as a
physics-informed GPR mean function.
```

```text
Ask matgpr to suggest a low-data validation workflow for a 75-row regression
dataset.
```

## Featurizer Recommendation Inputs

`recommend_featurizers` is designed for small schema previews, not full
datasets. Pass:

- column names,
- optional dtype summaries,
- one to five sample rows,
- optional target-column name,
- optional task type.

The tool uses conservative heuristics. It can recognize common evidence for:

- inorganic composition/formula columns,
- molecule SMILES columns,
- polymer repeat-unit SMILES columns with `[*]` dummy atoms,
- crystal-structure columns or file names,
- numeric process-condition columns,
- likely target columns.

Always review the recommendation before training. Ambiguous columns should be
confirmed by the user.

## Safe Equation Validation

`validate_safe_equation` wraps the public safe-equation schema and validator.
The input is the same JSON-like spec used by `matgpr.safe_equations`.

Minimal example:

```json
{
  "name": "linear_temperature_mean",
  "expression": "offset + slope * temperature_k",
  "variables": [
    {"name": "temperature_k", "units": "K"}
  ],
  "parameters": [
    {"name": "offset", "initial_value": 0.0},
    {"name": "slope", "initial_value": 1.0}
  ]
}
```

The tool returns:

- whether the spec is valid,
- validation errors and warnings,
- normalized spec when valid,
- the safe-equation schema snapshot,
- implementation hints.

The validator does not use `eval`. It accepts only declared symbols,
allowlisted operators, and allowlisted mathematical functions.

## What The Server Does Not Do Yet

The first public MCP server is advisory. It does not yet:

- fit GPR models,
- run Bayesian optimization,
- read arbitrary files,
- write artifacts,
- execute notebooks,
- install dependencies,
- upload data,
- call external services.

Those capabilities can be considered later only after the read-only server is
stable and the safety model is reviewed.

## Troubleshooting

If `matgpr-mcp` is not found:

```bash
python -m pip install -e ".[mcp]"
```

or, after a release that includes MCP support:

```bash
python -m pip install "matgpr[mcp]"
```

If the command reports that MCP is missing, install the MCP extra:

```bash
python -m pip install "matgpr[mcp]"
```

If your MCP host cannot find `matgpr-mcp`, use the full path from:

```bash
which matgpr-mcp
```

and place that absolute command path in the host configuration.

If an MCP host starts but does not show tools, restart the host after changing
its MCP configuration.

## Development Checks

Before releasing MCP support, run:

```bash
python -m ruff check matgpr tests scripts
python -m pytest
python -m mkdocs build --strict
python -m build
```

For package-upload readiness, also run `twine check` on the generated
distribution files.

## References

- [Model Context Protocol tools specification](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)
- [OpenAI Codex MCP documentation](https://learn.chatgpt.com/docs/extend/mcp)
- [Claude Code MCP documentation](https://code.claude.com/docs/en/mcp)
- [MCP Python SDK](https://py.sdk.modelcontextprotocol.io/)
