from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from matgpr._version import __version__
from matgpr.mcp_server import MCP_SERVER_NAME, MCP_TOOL_FUNCTIONS, create_server, main


class FakeFastMCP:
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.kwargs = kwargs
        self.tools: dict[str, object] = {}
        self.ran = False

    def tool(self):
        def decorator(function):
            self.tools[function.__name__] = function
            return function

        return decorator

    def run(self):
        self.ran = True


def fake_mcp_modules() -> dict[str, types.ModuleType]:
    mcp_module = types.ModuleType("mcp")
    mcp_module.__path__ = []
    server_module = types.ModuleType("mcp.server")
    server_module.__path__ = []
    fastmcp_module = types.ModuleType("mcp.server.fastmcp")
    fastmcp_module.FastMCP = FakeFastMCP
    mcp_module.server = server_module
    server_module.fastmcp = fastmcp_module
    return {
        "mcp": mcp_module,
        "mcp.server": server_module,
        "mcp.server.fastmcp": fastmcp_module,
    }


def fake_mcp_v2_modules() -> dict[str, types.ModuleType]:
    modules = fake_mcp_modules()
    modules["mcp.server"].MCPServer = FakeFastMCP
    return modules


class MCPServerTests(unittest.TestCase):
    def test_create_server_registers_package_info_tool(self):
        with patch.dict(sys.modules, fake_mcp_v2_modules()):
            server = create_server()

        self.assertEqual(server.name, MCP_SERVER_NAME)
        self.assertEqual(server.kwargs["title"], "matgpr")
        self.assertEqual(set(server.tools), {function.__name__ for function in MCP_TOOL_FUNCTIONS})

        info = server.tools["get_matgpr_info"]()
        self.assertEqual(info["name"], "matgpr")
        self.assertEqual(info["version"], __version__)
        self.assertIn("physics-informed GPR", info["capabilities"])

    def test_create_server_supports_legacy_fastmcp_import_path(self):
        with patch.dict(sys.modules, fake_mcp_modules()):
            server = create_server()

        self.assertEqual(server.name, MCP_SERVER_NAME)
        self.assertIn("get_matgpr_info", server.tools)

    def test_create_server_reports_missing_optional_dependency(self):
        with patch(
            "matgpr.optional_dependencies.importlib.import_module",
            side_effect=ImportError("missing mcp"),
        ):
            with self.assertRaises(ImportError) as context:
                create_server()

        message = str(context.exception)
        self.assertIn("Model Context Protocol server support", message)
        self.assertIn("matgpr[mcp]", message)
        self.assertIn("optional dependency `mcp`", message)

    def test_main_runs_created_server(self):
        server = FakeFastMCP(MCP_SERVER_NAME)
        with patch("matgpr.mcp_server.create_server", return_value=server):
            main()

        self.assertTrue(server.ran)

    def test_main_converts_missing_dependency_to_system_exit(self):
        with patch("matgpr.mcp_server.create_server", side_effect=ImportError("install mcp")):
            with self.assertRaises(SystemExit) as context:
                main()

        self.assertEqual(str(context.exception), "install mcp")


if __name__ == "__main__":
    unittest.main()
