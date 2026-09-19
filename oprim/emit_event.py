"""事件发布原子操作 (3O 范式全栈 — oprim 元素 11).

单次原子 IO: 向守护总线发布全局事件, 不感知消费方语义。
"""

from __future__ import annotations

from obase import DaemonBusProvider


async def emit_global_event(
    topic: str,
    *,
    payload: dict,
    bus: DaemonBusProvider,
) -> None:
    """单次原子 IO: 向守护总线发布全局事件。

    Args:
        topic: 事件主题。
        payload: 事件载荷。
        bus: 守护总线契约实现 (依赖注入)。
    """
    await bus.publish(topic, payload)
