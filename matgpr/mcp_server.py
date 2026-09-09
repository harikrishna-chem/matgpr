"""Local MCP server entry point for optional AI-agent access to matgpr."""

from __future__ import annotations

from typing import Any

from ._version import __version__
from .mcp_prompts import MCP_PROMPT_REGISTRATIONS
from .mcp_resources import MCP_RESOURCE_REGISTRATIONS
from .mcp_tools import (
    get_matgpr_info,
    get_physics_equation,
    list_capabilities,
    list_physics_equations,
    preview_safe_equation,
    recommend_featurizers,
    suggest_bo_workflow,
    suggest_validation_workflow,
    validate_safe_equation,
)
from .optional_dependencies import require_optional_dependency

__all__ = [
    "MCP_PROMPT_REGISTRATIONS",
    "MCP_RESOURCE_REGISTRATIONS",
    "MCP_TOOL_FUNCTIONS",
    "MCP_SERVER_NAME",
    "create_server",
    "main",
]


MCP_SERVER_NAME = "matgpr"
MCP_TOOL_FUNCTIONS = (
    get_matgpr_info,
    list_capabilities,
    recommend_featurizers,
    list_physics_equations,
    get_physics_equation,
    validate_safe_equation,
    preview_safe_equation,
    suggest_validation_workflow,
    suggest_bo_workflow,
)


def create_server() -> Any:
    """Create the local `matgpr` MCP server.

    The MCP SDK is an optional dependency. Importing `matgpr` or this module
    should not require MCP; only server construction does.
    """
    server_class = _mcp_server_class()

    server = server_class(
        MCP_SERVER_NAME,
        title="matgpr",
        description="Read-only public helpers for the matgpr materials-informatics GPR toolkit.",
        version=__version__,
    )
    for tool_function in MCP_TOOL_FUNCTIONS:
        server.tool()(tool_function)
    for registration in MCP_PROMPT_REGISTRATIONS:
        server.prompt(
            name=registration.name,
            title=registration.title,
            description=registration.description,
        )(registration.function)
    for registration in MCP_RESOURCE_REGISTRATIONS:
        server.resource(
            registration.uri,
            name=registration.name,
            title=registration.title,
            description=registration.description,
            mime_type=registration.mime_type,
        )(registration.function)

    return server


def main() -> None:
    """Run the local STDIO MCP server."""
    try:
        server = create_server()
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc
    server.run()


def _mcp_server_class() -> type[Any]:
    require_optional_dependency("mcp")
    try:
        from mcp.server import MCPServer

        return MCPServer
    except ImportError:
        from mcp.server.fastmcp import FastMCP

        return FastMCP
