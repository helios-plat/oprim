"""oprim.mcp_register_tool — MCP 工具注册.

把 (name, description, input_schema, handler) 注册到注入的 MCP server
（obase.mcp_server.MCPServer 或兼容协议），返回注册结果。

Example:
    >>> r = mcp_register_tool(
    ...     server, name="veya.greet", description="打招呼",
    ...     input_schema={"type": "object", "properties": {"who": {"type": "string"}}},
    ...     handler=async_handler,
    ... )
    >>> r["registered"]
    True
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

from oprim._exceptions import OprimError, OprimValidationError


class McpRegisterError(OprimError):
    """工具注册失败。"""


@runtime_checkable
class McpServerHandle(Protocol):
    """MCP server 注册协议（对齐 obase.mcp_server.MCPServer）。"""

    def register_skill(self, skill_def: Any) -> None: ...


def mcp_register_tool(
    server: McpServerHandle,
    *,
    name: str,
    description: str,
    input_schema: dict[str, Any],
    handler: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
    output_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """注册单个 MCP 工具。

    Args:
        server: MCP server 句柄（obase.mcp_server.MCPServer 兼容）。
        name: 工具全名（如 "veya.greet"）。
        description: 工具描述（LLM 可见）。
        input_schema: JSON Schema 输入描述。
        handler: async (args: dict) -> dict 处理器。
        output_schema: 输出 JSON Schema（可选，MCP 2025-06-18+）。

    Returns:
        {"status": "ok", "registered": True, "tool_name": str, "server": str}

    Raises:
        McpRegisterError: 注册失败 / server 不兼容。
        OprimValidationError: name / description / input_schema 缺失。
    """
    if not name or not name.strip():
        raise OprimValidationError("mcp_register_tool: name must not be empty")
    if not description:
        raise OprimValidationError("mcp_register_tool: description must not be empty")
    if not isinstance(input_schema, dict) or "type" not in input_schema:
        raise OprimValidationError(
            "mcp_register_tool: input_schema must be a JSON Schema object"
        )
    if server is None:
        raise OprimValidationError("mcp_register_tool: server must be injected")

    register_fn = getattr(server, "register_skill", None)
    if register_fn is None or not callable(register_fn):
        raise McpRegisterError(
            "mcp_register_tool: server has no register_skill() (need obase.mcp_server.MCPServer)"
        )

    try:
        # 兼容 obase.mcp_server.SkillDef 与裸 dict 两种形态
        try:
            from obase.mcp_server import SkillDef

            skill = SkillDef(
                name=name,
                description=description,
                input_schema=input_schema,
                output_schema=output_schema,
                handler=handler,
            )
        except ImportError:  # pragma: no cover - obase 缺失时退回 dict 形态
            skill = {
                "name": name,
                "description": description,
                "input_schema": input_schema,
                "output_schema": output_schema,
                "handler": handler,
            }
        register_fn(skill)
    except Exception as exc:
        raise McpRegisterError(
            f"mcp_register_tool failed for {name!r}: {type(exc).__name__}: {exc}",
            cause=exc,
        ) from exc

    return {
        "status": "ok",
        "registered": True,
        "tool_name": name,
        "server": getattr(server, "name", type(server).__name__),
    }
