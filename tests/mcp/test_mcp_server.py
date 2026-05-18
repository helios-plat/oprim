"""Tests for oprim.mcp.mcp_server."""
from __future__ import annotations

import pytest

from oprim.mcp.mcp_server import create_mcp_server, register_tool


class TestMCPServer:
    def test_create_server_returns_fastmcp(self):
        server = create_mcp_server("test-server", "1.0.0")
        # Check by class name to be independent of which fastmcp variant is loaded
        assert type(server).__name__ == "FastMCP"

    def test_server_has_name(self):
        server = create_mcp_server("my-server", "0.1.0")
        assert server.name == "my-server"

    def test_register_tool_callable(self):
        server = create_mcp_server("test-server", "1.0.0")

        def my_tool(x: int) -> str:
            return str(x)

        # register_tool must not raise
        register_tool(server, "my_tool", my_tool, description="A test tool")

    def test_register_tool_no_description(self):
        server = create_mcp_server("test-server", "1.0.0")

        def echo(text: str) -> str:
            return text

        register_tool(server, "echo", echo)

    def test_multiple_servers_are_independent(self):
        s1 = create_mcp_server("server-1", "1.0.0")
        s2 = create_mcp_server("server-2", "2.0.0")
        assert s1 is not s2
        assert s1.name == "server-1"
        assert s2.name == "server-2"

    def test_register_lambda_tool(self):
        server = create_mcp_server("test-server", "1.0.0")
        fn = lambda x: x * 2  # noqa: E731

        # lambdas have no __name__ but FastMCP uses the `name` parameter
        register_tool(server, "double", fn, description="Double a value")
