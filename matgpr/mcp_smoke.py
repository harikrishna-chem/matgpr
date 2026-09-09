"""Command-line smoke test for the optional local `matgpr` MCP server."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .mcp_server import (
    MCP_PROMPT_REGISTRATIONS,
    MCP_RESOURCE_REGISTRATIONS,
    MCP_SERVER_NAME,
    MCP_TOOL_FUNCTIONS,
)

__all__ = [
    "EXPECTED_PROMPT_NAMES",
    "EXPECTED_RESOURCE_URIS",
    "EXPECTED_TOOL_NAMES",
    "main",
]

EXPECTED_PROMPT_NAMES = tuple(registration.name for registration in MCP_PROMPT_REGISTRATIONS)
EXPECTED_RESOURCE_URIS = tuple(registration.uri for registration in MCP_RESOURCE_REGISTRATIONS)
EXPECTED_TOOL_NAMES = tuple(function.__name__ for function in MCP_TOOL_FUNCTIONS)


def main(argv: Sequence[str] | None = None) -> int:
    """Run a real MCP stdio handshake against an installed server command."""
    args = _parse_args(argv)
    try:
        import anyio
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
    except ImportError:
        print(_missing_dependency_message(), file=sys.stderr)
        return 2

    try:
        payload = anyio.run(
            _run_smoke,
            args,
            ClientSession,
            StdioServerParameters,
            stdio_client,
        )
    except FileNotFoundError:
        print(
            "matgpr MCP smoke failed: command not found. "
            "Install `matgpr[mcp]` or pass --command with the absolute path to matgpr-mcp.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(f"matgpr MCP smoke failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_format_human_summary(payload))
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test the installed matgpr MCP server over stdio.",
    )
    parser.add_argument(
        "--command",
        default="matgpr-mcp",
        help="Server executable or absolute path. Defaults to matgpr-mcp.",
    )
    parser.add_argument(
        "--cwd",
        default=None,
        help="Optional working directory for the server process.",
    )
    parser.add_argument(
        "--skip-info-call",
        action="store_true",
        help="Only initialize and list tools; skip the get_matgpr_info tool call.",
    )
    parser.add_argument(
        "--skip-prompt-resource-checks",
        action="store_true",
        help="Only check tools; skip prompt and resource listing checks.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a JSON-safe smoke-test payload instead of a human summary.",
    )
    parser.add_argument(
        "server_args",
        nargs=argparse.REMAINDER,
        help="Optional arguments forwarded to the server command after --.",
    )
    return parser.parse_args(argv)


async def _run_smoke(
    args: argparse.Namespace,
    client_session_class: Any,
    stdio_parameters_class: Any,
    stdio_client_factory: Any,
) -> dict[str, object]:
    server_args = _normalize_server_args(args.server_args)
    cwd = Path(args.cwd).expanduser() if args.cwd else None
    parameters = stdio_parameters_class(command=args.command, args=server_args, cwd=cwd)

    async with stdio_client_factory(parameters) as (read_stream, write_stream):
        async with client_session_class(read_stream, write_stream) as session:
            initialization = await session.initialize()
            tools_result = await session.list_tools()
            tool_names = [str(tool.name) for tool in tools_result.tools]
            _raise_if_missing("tool", EXPECTED_TOOL_NAMES, tool_names)

            prompt_names: list[str] = []
            resource_uris: list[str] = []
            checked_prompt: str | None = None
            checked_resource: str | None = None
            if not args.skip_prompt_resource_checks:
                prompts_result = await session.list_prompts()
                prompt_names = [str(prompt.name) for prompt in prompts_result.prompts]
                _raise_if_missing("prompt", EXPECTED_PROMPT_NAMES, prompt_names)
                await session.get_prompt(EXPECTED_PROMPT_NAMES[0], arguments={})
                checked_prompt = EXPECTED_PROMPT_NAMES[0]

                resources_result = await session.list_resources()
                resource_uris = [str(resource.uri) for resource in resources_result.resources]
                _raise_if_missing("resource", EXPECTED_RESOURCE_URIS, resource_uris)
                await session.read_resource(EXPECTED_RESOURCE_URIS[0])
                checked_resource = EXPECTED_RESOURCE_URIS[0]

            info_payload: Mapping[str, object] | None = None
            if not args.skip_info_call:
                call_result = await session.call_tool("get_matgpr_info", {})
                if getattr(call_result, "is_error", False):
                    raise RuntimeError("get_matgpr_info returned an MCP error")
                info_payload = _extract_tool_payload(call_result)

    server_info = getattr(initialization, "serverInfo", None)
    if server_info is None:
        server_info = getattr(initialization, "server_info", None)

    return {
        "ok": True,
        "server_name": MCP_SERVER_NAME,
        "command": str(args.command),
        "command_args": server_args,
        "cwd": str(cwd) if cwd is not None else None,
        "tool_count": len(tool_names),
        "tool_names": tool_names,
        "expected_tool_names": list(EXPECTED_TOOL_NAMES),
        "prompt_count": len(prompt_names),
        "prompt_names": prompt_names,
        "expected_prompt_names": list(EXPECTED_PROMPT_NAMES),
        "checked_prompt": checked_prompt,
        "resource_count": len(resource_uris),
        "resource_uris": resource_uris,
        "expected_resource_uris": list(EXPECTED_RESOURCE_URIS),
        "checked_resource": checked_resource,
        "package_info": dict(info_payload) if info_payload is not None else None,
        "server_info": _object_to_json_safe_dict(server_info),
    }


def _normalize_server_args(server_args: Sequence[str]) -> list[str]:
    args = list(server_args)
    if args and args[0] == "--":
        return args[1:]
    return args


def _extract_tool_payload(call_result: Any) -> Mapping[str, object] | None:
    structured_content = getattr(call_result, "structured_content", None)
    if isinstance(structured_content, Mapping):
        return structured_content

    for content in getattr(call_result, "content", ()):
        text = getattr(content, "text", None)
        if not isinstance(text, str):
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            return payload
    return None


def _format_human_summary(payload: Mapping[str, object]) -> str:
    package_info = payload.get("package_info")
    package_line = None
    if isinstance(package_info, Mapping):
        name = package_info.get("name")
        version = package_info.get("version")
        if name and version:
            package_line = f"Package: {name} {version}"

    tool_names = _string_list(payload.get("tool_names", ()))
    prompt_names = _string_list(payload.get("prompt_names", ()))
    resource_uris = _string_list(payload.get("resource_uris", ()))
    lines = [
        "matgpr MCP smoke passed.",
        f"Command: {payload.get('command', '')}",
        f"Tools: {payload.get('tool_count', len(tool_names))} ({', '.join(tool_names)})",
    ]
    if prompt_names:
        lines.append(
            f"Prompts: {payload.get('prompt_count', len(prompt_names))} ({', '.join(prompt_names)})"
        )
    if resource_uris:
        lines.append(
            f"Resources: {payload.get('resource_count', len(resource_uris))} "
            f"({', '.join(resource_uris)})"
        )
    if package_line:
        lines.append(package_line)
    return "\n".join(lines)


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return []


def _raise_if_missing(kind: str, expected: Sequence[str], observed: Sequence[str]) -> None:
    missing = [name for name in expected if name not in observed]
    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(f"server did not expose expected {kind}(s): {missing_text}")


def _object_to_json_safe_dict(value: Any) -> dict[str, object] | None:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
    elif hasattr(value, "dict"):
        dumped = value.dict()
    elif isinstance(value, Mapping):
        dumped = dict(value)
    else:
        dumped = {
            key: getattr(value, key) for key in ("name", "version", "title") if hasattr(value, key)
        }
    return _json_roundtrip(dumped) if isinstance(dumped, Mapping) else None


def _json_roundtrip(payload: Mapping[str, object]) -> dict[str, object]:
    return json.loads(json.dumps(payload, default=str, allow_nan=False))


def _missing_dependency_message() -> str:
    return (
        "The matgpr MCP smoke test requires the optional MCP dependency.\n"
        'Install it with: python -m pip install "matgpr[mcp]"\n'
        'For an editable checkout: python -m pip install -e ".[mcp]"'
    )


if __name__ == "__main__":
    raise SystemExit(main())
