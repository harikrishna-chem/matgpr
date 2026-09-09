from __future__ import annotations

import json
import types
import unittest

from matgpr.mcp_server import MCP_TOOL_FUNCTIONS
from matgpr.mcp_smoke import (
    EXPECTED_TOOL_NAMES,
    _extract_tool_payload,
    _format_human_summary,
    _missing_dependency_message,
    _normalize_server_args,
    _parse_args,
)


class MCPSmokeTests(unittest.TestCase):
    def test_expected_tool_names_match_server_registry(self):
        self.assertEqual(EXPECTED_TOOL_NAMES, tuple(tool.__name__ for tool in MCP_TOOL_FUNCTIONS))
        self.assertIn("preview_safe_equation", EXPECTED_TOOL_NAMES)

    def test_parse_args_supports_absolute_command_and_forwarded_args(self):
        args = _parse_args(
            [
                "--command",
                "/tmp/matgpr-mcp",
                "--cwd",
                "/tmp",
                "--json",
                "--",
                "--server-option",
            ]
        )

        self.assertEqual(args.command, "/tmp/matgpr-mcp")
        self.assertEqual(args.cwd, "/tmp")
        self.assertTrue(args.json)
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
                "package_info": {"name": "matgpr", "version": "0.2.0"},
            }
        )

        self.assertIn("matgpr MCP smoke passed", summary)
        self.assertIn("Command: matgpr-mcp", summary)
        self.assertIn("Tools: 2", summary)
        self.assertIn("Package: matgpr 0.2.0", summary)

    def test_missing_dependency_message_mentions_mcp_extra(self):
        message = _missing_dependency_message()

        self.assertIn("matgpr[mcp]", message)
        self.assertIn("editable checkout", message)


if __name__ == "__main__":
    unittest.main()
