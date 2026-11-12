# Public MCP CI Smoke Audit

**Date:** 2026-11-05
**Repository:** `matgpr`
**Scope:** Step 10, Slice 4

## Purpose

Exercise the real optional Model Context Protocol boundary in CI. The ordinary
MCP tests intentionally use fake registrations and process adapters for speed
and deterministic failure coverage; they do not prove that the current MCP SDK
can initialize the packaged server and communicate over stdio.

## CI Contract

The dedicated `MCP integration smoke` job:

1. Uses Python 3.11.
2. Installs the package with `.[mcp]` in the isolated GitHub runner.
3. Runs `pip check` to reject an inconsistent optional dependency graph.
4. Resolves the packaged `matgpr-mcp` console entry point.
5. Runs the packaged `matgpr-mcp-smoke` client against that server over stdio.
6. Requires the smoke command to initialize and expose every expected tool,
   prompt, and resource, read one resource, invoke one prompt, and call
   `get_matgpr_info` successfully.

The job is independent of core tests and package quality. MCP remains an
optional dependency and is not installed in ordinary test jobs.

## Local Wheel Validation

The local validation used an isolated temporary environment with shared core
scientific dependencies, installed the built `matgpr 0.2.0` wheel without
dependencies, and installed the current optional MCP SDK separately.

- MCP SDK version: `2.3.0`.
- Dependency check: passed.
- Server initialization: passed.
- Tools: 9 observed and 9 expected.
- Prompts: 4 observed and 4 expected.
- Resources: 5 observed and 5 expected.
- Prompt invocation: passed.
- Resource read: passed.
- `get_matgpr_info`: passed with JSON-safe package metadata.
- MCP-focused pytest selection: 58 passed.

The result confirms the package's compatibility path for the MCP 2.x
`MCPServer` API. Existing tests continue to cover the fallback used by MCP 1.x
`FastMCP` releases.

## Decision

No MCP source fix or version pin is needed. The real client/server smoke is now
a blocking public CI job without changing core dependencies, public APIs,
coverage thresholds, existing check names, or ordinary test behavior.

## Final Verification

- Workflow YAML and MCP job structure assertions passed.
- Real packaged client/server smoke passed with MCP SDK 2.3.0.
- All 58 focused MCP tests passed with the real SDK installed.
- All 874 public tests passed at 83.81% branch-aware coverage.
- Ruff lint and format checks passed.
- Strict MkDocs build passed.
- `git diff --check` passed.
