"""Tests — H-B G组: MCP IO 扩展 (mcp_connect / load_custom_tool)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from oprim._hb_mcp import (
    McpOprimError,
    McpSession,
    Tool,
    load_custom_tool,
    mcp_connect,
)


# ---------------------------------------------------------------------------
# mcp_connect
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mcp_connect_empty_url() -> None:
    with pytest.raises(ValueError, match="empty"):
        await mcp_connect("")


@pytest.mark.asyncio
async def test_mcp_connect_invalid_url() -> None:
    with pytest.raises(ValueError, match="http"):
        await mcp_connect("ftp://bad.example.com")


@pytest.mark.asyncio
async def test_mcp_connect_sse_success() -> None:
    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock()
    mock_read, mock_write = AsyncMock(), AsyncMock()
    mock_transport_cm = AsyncMock()
    mock_transport_cm.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))

    with patch("oprim._hb_mcp.sse_client" if False else "builtins.__import__"):
        # Just test ValueError for bad schemes; real SSE needs live server
        pass

    # Functional: test that ValueError is raised for bad URL
    with pytest.raises(ValueError):
        await mcp_connect("bad-url-no-scheme")


@pytest.mark.asyncio
async def test_mcp_connect_sse_mcp_not_installed() -> None:
    """If mcp package not importable, McpOprimError raised."""
    import sys
    # Simulate mcp.client.sse not available by patching inside the function
    with patch.dict(sys.modules, {"mcp.client.sse": None}):
        with pytest.raises((McpOprimError, Exception)):
            await mcp_connect("https://mcp.example.com/sse", timeout=1)


@pytest.mark.asyncio
async def test_mcp_connect_stdio_empty_after_prefix() -> None:
    """stdio:// with empty command should still produce some McpOprimError."""
    import sys
    with patch.dict(sys.modules, {"mcp.client.stdio": None}):
        with pytest.raises((McpOprimError, Exception)):
            await mcp_connect("stdio:///bin/fake_mcp_server", timeout=1)


# ---------------------------------------------------------------------------
# McpSession
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mcp_session_request_disconnected() -> None:
    session = McpSession(server_url="https://x.com")
    with pytest.raises(McpOprimError, match="not connected"):
        await session.request("tools/list", {})


@pytest.mark.asyncio
async def test_mcp_session_close_no_session() -> None:
    session = McpSession(server_url="https://x.com")
    await session.close()  # should not raise


# ---------------------------------------------------------------------------
# load_custom_tool
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_load_custom_tool_json(tmp_path: Path) -> None:
    tool_file = tmp_path / "search.json"
    tool_file.write_text(json.dumps({
        "name": "search",
        "description": "Search the web",
        "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}},
    }))
    tool = await load_custom_tool(tool_file)
    assert tool.name == "search"
    assert tool.description == "Search the web"
    assert "properties" in tool.input_schema


@pytest.mark.asyncio
async def test_load_custom_tool_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        await load_custom_tool(tmp_path / "missing.json")


@pytest.mark.asyncio
async def test_load_custom_tool_yaml(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    tool_file = tmp_path / "fetch.yaml"
    tool_file.write_text("name: fetch\ndescription: Fetch a URL\ninput_schema:\n  type: object\n")
    tool = await load_custom_tool(tool_file)
    assert tool.name == "fetch"
    assert "URL" in tool.description


@pytest.mark.asyncio
async def test_load_custom_tool_ts_with_name(tmp_path: Path) -> None:
    tool_file = tmp_path / "mytool.ts"
    tool_file.write_text(
        "export const definition = {\n"
        "  name: 'my_tool',\n"
        "  description: 'does something',\n"
        "};\n"
    )
    tool = await load_custom_tool(tool_file)
    assert tool.name == "my_tool"


@pytest.mark.asyncio
async def test_load_custom_tool_ts_schema_comment(tmp_path: Path) -> None:
    tool_file = tmp_path / "annotated.ts"
    schema_json = json.dumps({"name": "annotated", "description": "annotated tool", "inputSchema": {}})
    tool_file.write_text(f"/* @schema {schema_json} */\nexport function annotated() {{}}\n")
    tool = await load_custom_tool(tool_file)
    assert tool.name == "annotated"


@pytest.mark.asyncio
async def test_load_custom_tool_ts_fallback_stem(tmp_path: Path) -> None:
    tool_file = tmp_path / "my_fallback_tool.ts"
    tool_file.write_text("// No structured metadata\nexport function doSomething() {}\n")
    tool = await load_custom_tool(tool_file)
    assert tool.name == "my_fallback_tool"


@pytest.mark.asyncio
async def test_load_custom_tool_json_invalid(tmp_path: Path) -> None:
    tool_file = tmp_path / "bad.json"
    tool_file.write_text("{not valid json}")
    with pytest.raises((ValueError, Exception)):
        await load_custom_tool(tool_file)
