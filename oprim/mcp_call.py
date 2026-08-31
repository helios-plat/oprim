"""Atomic MCP tool invocation through an injected client transport."""

from __future__ import annotations

import inspect
from typing import Any


class McpCallError(RuntimeError):
    """Safe MCP transport error; remote error text is not exposed."""


async def mcp_call(
    tool: str,
    *,
    arguments: dict[str, Any],
    client: Any,
    server: str = "",
    credential: str | None = None,
) -> Any:
    """Call one MCP tool; credential stays inside the injected transport."""
    del server
    try:
        if credential is not None and hasattr(client, "call_tool_with_credential"):
            result = client.call_tool_with_credential(tool, arguments, credential)
        else:
            result = client.call_tool(tool, arguments)
        if inspect.isawaitable(result):
            result = await result
        return result
    except Exception as exc:
        raise McpCallError(f"MCP transport failed ({type(exc).__name__})") from exc


__all__ = ["McpCallError", "mcp_call"]
