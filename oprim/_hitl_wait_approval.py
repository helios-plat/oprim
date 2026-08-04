"""oprim.hitl_wait_approval — 人工确认等待.

经注入的 hitl 总线（obase.hitl_signal_bus.HitlSignalBus 兼容协议）等待
人工审批决策，返回标准化 dict。bus 缺省取 ContextVar 当前绑定。

Example:
    >>> d = await hitl_wait_approval("hitl-abc", timeout=30.0)
    >>> d["outcome"]
    'approved'
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from oprim._exceptions import OprimError, OprimValidationError


class HitlWaitError(OprimError):
    """审批等待失败。"""


@runtime_checkable
class HitlBusHandle(Protocol):
    """审批总线协议（对齐 obase.hitl_signal_bus.HitlSignalBus）。"""

    async def wait_for_decision(
        self, request_id: str, *, timeout: float = 60.0
    ) -> dict[str, Any]: ...


async def hitl_wait_approval(
    request_id: str,
    *,
    bus: HitlBusHandle | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """等待人工确认。

    Args:
        request_id: 审批请求 ID（由总线 request_approval 创建）。
        bus: 审批总线（注入）；None 时取 ContextVar 当前绑定。
        timeout: 等待超时秒数（超时返回 outcome="timed_out"，不抛）。

    Returns:
        {
            "status": "ok",
            "request_id": str,
            "outcome": "approved" | "rejected" | "timed_out" | "cancelled" | "unknown",
            "note": str,
            "scope": str | None,
            "summary": str | None,
        }

    Raises:
        HitlWaitError: 无可用总线。
        OprimValidationError: request_id 为空 / timeout 非法。
    """
    if not request_id:
        raise OprimValidationError("hitl_wait_approval: request_id must not be empty")
    if timeout <= 0:
        raise OprimValidationError("hitl_wait_approval: timeout must be > 0")

    if bus is None:
        try:
            from obase.hitl_signal_bus import get_bus
        except ImportError as exc:  # pragma: no cover - 环境相关
            raise HitlWaitError("hitl_wait_approval: obase unavailable", cause=exc) from exc
        bus = get_bus()
    if bus is None:
        raise HitlWaitError(
            "hitl_wait_approval: no bus bound — pass bus=... or bind_bus() first"
        )

    decision = await bus.wait_for_decision(request_id, timeout=timeout)
    return {
        "status": "ok",
        "request_id": decision.get("request_id", request_id),
        "outcome": decision.get("outcome", "unknown"),
        "note": decision.get("note", ""),
        "scope": decision.get("scope"),
        "summary": decision.get("summary"),
    }
