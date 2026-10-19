"""oprim: 下发硬件指令原子操作 (≤1 位置参数).

铁律: ≤1 个位置参数，其余 keyword-only.

组合: MQTTEdgeGateway (Protocol 注入，不硬 import obase).
"""

from __future__ import annotations

from typing import Any, Protocol


class MQTTGatewayProtocol(Protocol):
    """MQTT 网关 Protocol — 不 import obase，由调用方注入."""

    connected: bool

    def publish(self, topic: str, payload: bytes, qos: int = 1) -> bool: ...


def _publish_hardware_command(
    topic: str,
    *,
    gateway: MQTTGatewayProtocol,
    payload: bytes,
    qos: int = 1,
) -> dict[str, Any]:
    """向指定的主题下发二进制指令，不抛出异常.

    签名遵循 oprim 铁律：最多 1 个位置参数，其余 kw-only.

    Args:
        topic: MQTT 主题
        gateway: MQTT 网关（注入 MQTTGatewayProtocol）
        payload: 二进制载荷
        qos: QoS 级别

    Returns:
        {
            "status": "success" | "error",
            "topic": str,
            "bytes_sent": int,
            "error": str | None,
        }
    """
    try:
        if not gateway.connected:
            return {"status": "error", "error": "网关未连接", "topic": topic}

        success = gateway.publish(topic, payload, qos=qos)
        if success:
            return {"status": "success", "topic": topic, "bytes_sent": len(payload)}
        else:
            return {"status": "error", "error": "MQTT publish 确认失败", "topic": topic}
    except Exception as e:
        return {"status": "error", "error": str(e), "topic": topic}
