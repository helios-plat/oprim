"""oprim.frontend_tool_forward — 前端工具转发.

把一次工具调用转发给前端网关（gateway Protocol 注入），返回网关响应。

Example:
    >>> r = await frontend_tool_forward("notify", payload={"msg": "hi"}, gateway=gw)
    >>> r["delivered"]
    True
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from oprim._exceptions import OprimError, OprimValidationError


class FrontendForwardError(OprimError):
    """前端转发失败。"""


@runtime_checkable
class FrontendGateway(Protocol):
    """前端网关协议（注入面）。"""

    async def forward(self, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]: ...


async def frontend_tool_forward(
    tool_name: str,
    *,
    payload: dict[str, Any],
    gateway: FrontendGateway,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """转发工具调用到前端。

    Args:
        tool_name: 工具名。
        payload: 调用载荷。
        gateway: 前端网关（注入）。
        timeout: 超时秒数。

    Returns:
        {"status": "ok", "tool": str, "delivered": True, "response": dict}

    Raises:
        FrontendForwardError: 转发失败 / 超时。
        OprimValidationError: tool_name 为空 / gateway 未注入。
    """
    if not tool_name or not tool_name.strip():
        raise OprimValidationError("frontend_tool_forward: tool_name must not be empty")
    if gateway is None:
        raise OprimValidationError("frontend_tool_forward: gateway must be injected")

    import asyncio

    try:
        response = await asyncio.wait_for(
            gateway.forward(tool_name, payload), timeout=timeout
        )
    except TimeoutError as exc:
        raise FrontendForwardError(
            f"frontend_tool_forward timed out after {timeout}s: {tool_name}", cause=exc
        ) from exc
    except Exception as exc:
        raise FrontendForwardError(
            f"frontend_tool_forward failed: {type(exc).__name__}: {exc}", cause=exc
        ) from exc

    return {
        "status": "ok",
        "tool": tool_name,
        "delivered": True,
        "response": response if isinstance(response, dict) else {"result": response},
    }
