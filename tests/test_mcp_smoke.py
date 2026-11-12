from __future__ import annotations

import asyncio
import json
import types
import unittest

import pytest

from matgpr.mcp_server import MCP_TOOL_FUNCTIONS
from matgpr.mcp_smoke import (
    EXPECTED_PROMPT_NAMES,
    EXPECTED_RESOURCE_URIS,
    EXPECTED_TOOL_NAMES,
    _extract_tool_payload,
    _format_human_summary,
    _missing_dependency_message,
    _normalize_server_args,
    _object_to_json_safe_dict,
    _parse_args,
    _raise_if_missing,
    _run_smoke,
)

pytestmark = [pytest.mark.integration, pytest.mark.smoke]


class MCPSmokeTests(unittest.TestCase):
    def test_expected_tool_names_match_server_registry(self):
        self.assertEqual(EXPECTED_TOOL_NAMES, tuple(tool.__name__ for tool in MCP_TOOL_FUNCTIONS))
        self.assertIn("preview_safe_equation", EXPECTED_TOOL_NAMES)
        self.assertIn("plan_featurization_workflow", EXPECTED_PROMPT_NAMES)
        self.assertIn("matgpr://guide/capabilities", EXPECTED_RESOURCE_URIS)

    def test_parse_args_supports_absolute_command_and_forwarded_args(self):
        args = _parse_args(
            [
                "--command",
                "/tmp/matgpr-mcp",
                "--cwd",
                "/tmp",
                "--json",
                "--skip-prompt-resource-checks",
                "--",
                "--server-option",
            ]
        )

        self.assertEqual(args.command, "/tmp/matgpr-mcp")
        self.assertEqual(args.cwd, "/tmp")
        self.assertTrue(args.json)
        self.assertTrue(args.skip_prompt_resource_checks)
        self.assertEqual(_normalize_server_args(args.server_args), ["--server-option"])

    def test_extract_tool_payload_prefers_structured_content(self):
        result = types.SimpleNamespace(
            structured_content={"name": "matgpr", "version": "0.2.0"},
            content=[types.SimpleNamespace(text='{"name": "fallback"}')],
        )

        self.assertEqual(_extract_tool_payload(result)["name"], "matgpr")

    def test_extract_tool_payload_can_parse_text_content(self):
        result = types.SimpleNamespace(
            content=[types.SimpleNamespace(text=json.dumps({"name": "matgpr"}))]
        )

        self.assertEqual(_extract_tool_payload(result)["name"], "matgpr")

    def test_human_summary_lists_command_tools_and_package(self):
        summary = _format_human_summary(
            {
                "command": "matgpr-mcp",
                "tool_count": 2,
                "tool_names": ["get_matgpr_info", "list_capabilities"],
                "prompt_count": 1,
                "prompt_names": ["plan_featurization_workflow"],
                "resource_count": 1,
                "resource_uris": ["matgpr://guide/capabilities"],
                "package_info": {"name": "matgpr", "version": "0.2.0"},
            }
        )

        self.assertIn("matgpr MCP smoke passed", summary)
        self.assertIn("Command: matgpr-mcp", summary)
        self.assertIn("Tools: 2", summary)
        self.assertIn("Prompts: 1", summary)
        self.assertIn("Resources: 1", summary)
        self.assertIn("Package: matgpr 0.2.0", summary)

    def test_human_summary_accepts_sequence_like_name_payloads(self):
        summary = _format_human_summary(
            {
                "command": "matgpr-mcp",
                "tool_names": ("get_matgpr_info",),
                "prompt_names": ("plan_featurization_workflow",),
                "resource_uris": ("matgpr://guide/capabilities",),
                "package_info": None,
            }
        )

        self.assertIn("Tools: 1", summary)
        self.assertIn("Prompts: 1", summary)
        self.assertIn("Resources: 1", summary)

    def test_missing_dependency_message_mentions_mcp_extra(self):
        message = _missing_dependency_message()

        self.assertIn("matgpr[mcp]", message)
        self.assertIn("editable checkout", message)

    def test_run_smoke_exercises_tools_prompts_resources_and_info(self):
        args = _parse_args(["--command", "fake-matgpr-mcp", "--cwd", "/tmp"])
        session = _FakeSession()

        payload = asyncio.run(
            _run_smoke(
                args,
                lambda read, write: _AsyncContext(session),
                _FakeStdioParameters,
                lambda parameters: _AsyncContext(("read", "write")),
            )
        )

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["tool_names"], list(EXPECTED_TOOL_NAMES))
        self.assertEqual(payload["prompt_names"], list(EXPECTED_PROMPT_NAMES))
        self.assertEqual(payload["resource_uris"], list(EXPECTED_RESOURCE_URIS))
        self.assertEqual(payload["checked_prompt"], EXPECTED_PROMPT_NAMES[0])
        self.assertEqual(payload["checked_resource"], EXPECTED_RESOURCE_URIS[0])
        self.assertEqual(payload["package_info"]["name"], "matgpr")
        self.assertEqual(payload["server_info"]["version"], "0.2.0")
        self.assertEqual(session.requested_prompts, [(EXPECTED_PROMPT_NAMES[0], {})])
        self.assertEqual(session.requested_resources, [EXPECTED_RESOURCE_URIS[0]])
        json.dumps(payload, allow_nan=False)

    def test_run_smoke_can_skip_info_prompts_and_resources(self):
        args = _parse_args(
            [
                "--command",
                "fake-matgpr-mcp",
                "--skip-info-call",
                "--skip-prompt-resource-checks",
            ]
        )
        session = _FakeSession()

        payload = asyncio.run(
            _run_smoke(
                args,
                lambda read, write: _AsyncContext(session),
                _FakeStdioParameters,
                lambda parameters: _AsyncContext(("read", "write")),
            )
        )

        self.assertIsNone(payload["package_info"])
        self.assertEqual(payload["prompt_names"], [])
        self.assertEqual(payload["resource_uris"], [])
        self.assertEqual(session.requested_prompts, [])
        self.assertEqual(session.requested_resources, [])

    def test_run_smoke_rejects_mcp_tool_error(self):
        args = _parse_args(["--command", "fake-matgpr-mcp"])
        session = _FakeSession(tool_error=True)

        with self.assertRaisesRegex(RuntimeError, "get_matgpr_info returned an MCP error"):
            asyncio.run(
                _run_smoke(
                    args,
                    lambda read, write: _AsyncContext(session),
                    _FakeStdioParameters,
                    lambda parameters: _AsyncContext(("read", "write")),
                )
            )

    def test_raise_if_missing_reports_contract_name(self):
        with self.assertRaisesRegex(RuntimeError, "expected tool.*list_capabilities"):
            _raise_if_missing(
                "tool",
                ("get_matgpr_info", "list_capabilities"),
                ("get_matgpr_info",),
            )

    def test_object_to_json_safe_dict_supports_common_sdk_records(self):
        model_record = types.SimpleNamespace(model_dump=lambda: {"name": "matgpr", "count": 1})

        self.assertEqual(
            _object_to_json_safe_dict(model_record),
            {"name": "matgpr", "count": 1},
        )
        self.assertEqual(
            _object_to_json_safe_dict({"name": "matgpr"}),
            {"name": "matgpr"},
        )
        self.assertEqual(
            _object_to_json_safe_dict(types.SimpleNamespace(name="matgpr", version="0.2.0")),
            {"name": "matgpr", "version": "0.2.0"},
        )
        self.assertIsNone(_object_to_json_safe_dict(None))


class _AsyncContext:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _FakeStdioParameters:
    def __init__(self, *, command, args, cwd):
        self.command = command
        self.args = args
        self.cwd = cwd


class _FakeSession:
    def __init__(self, *, tool_error: bool = False):
        self.tool_error = tool_error
        self.requested_prompts: list[tuple[str, dict[str, object]]] = []
        self.requested_resources: list[str] = []

    async def initialize(self):
        return types.SimpleNamespace(
            serverInfo=types.SimpleNamespace(name="matgpr", version="0.2.0")
        )

    async def list_tools(self):
        return types.SimpleNamespace(
            tools=[types.SimpleNamespace(name=name) for name in EXPECTED_TOOL_NAMES]
        )

    async def list_prompts(self):
        return types.SimpleNamespace(
            prompts=[types.SimpleNamespace(name=name) for name in EXPECTED_PROMPT_NAMES]
        )

    async def get_prompt(self, name, arguments):
        self.requested_prompts.append((name, arguments))

    async def list_resources(self):
        return types.SimpleNamespace(
            resources=[types.SimpleNamespace(uri=uri) for uri in EXPECTED_RESOURCE_URIS]
        )

    async def read_resource(self, uri):
        self.requested_resources.append(uri)

    async def call_tool(self, name, arguments):
        assert name == "get_matgpr_info"
        assert arguments == {}
        return types.SimpleNamespace(
            is_error=self.tool_error,
            structured_content={"name": "matgpr", "version": "0.2.0"},
            content=[],
        )


if __name__ == "__main__":
    unittest.main()
