"""oprim._heartbeat_emit — 向外部死人开关打常燃心跳 (aegis DESIGN §6 L1)."""

from __future__ import annotations

import time

import httpx
from pydantic import BaseModel


class HeartbeatResult(BaseModel):
    url: str
    delivered: bool  # 2xx 视为送达
    status_code: int | None  # None = 请求未完成(网络错误)
    elapsed_ms: float
    error: str | None  # "timeout" / "connect_failed:..." / "http_4xx" / ...


def heartbeat_emit(
    *,
    url: str,
    method: str = "GET",
    timeout_sec: float = 5.0,
    user_agent: str = "obase-heartbeat/1.0",
) -> HeartbeatResult:
    """向 healthchecks.io 类外部死人开关打一次常燃心跳。

    **绝不抛异常**:心跳缺失本身即信号,失败以 delivered=False + error 返回,由外部服务
    侧在静默超时后触发告警。follow_redirects=False(SSRF 防护)。

    Args:
        url: 心跳 ping URL。
        method: HTTP 方法(通常 GET;某些服务用 POST)。
        timeout_sec: 请求超时秒数。
        user_agent: UA 头。

    Returns:
        HeartbeatResult(delivered / status_code / elapsed_ms / error)。
    """
    started = time.monotonic()
    try:
        with httpx.Client(follow_redirects=False, timeout=timeout_sec) as client:
            response = client.request(method, url, headers={"User-Agent": user_agent})
        elapsed_ms = (time.monotonic() - started) * 1000
        if response.is_success:
            return HeartbeatResult(
                url=url,
                delivered=True,
                status_code=response.status_code,
                elapsed_ms=elapsed_ms,
                error=None,
            )
        sc = response.status_code
        error_class = "http_3xx" if 300 <= sc < 400 else "http_4xx" if sc < 500 else "http_5xx"
        return HeartbeatResult(
            url=url,
            delivered=False,
            status_code=sc,
            elapsed_ms=elapsed_ms,
            error=error_class,
        )
    except httpx.TimeoutException:
        elapsed_ms = (time.monotonic() - started) * 1000
        return HeartbeatResult(
            url=url, delivered=False, status_code=None, elapsed_ms=elapsed_ms, error="timeout"
        )
    except httpx.ConnectError as e:
        elapsed_ms = (time.monotonic() - started) * 1000
        return HeartbeatResult(
            url=url,
            delivered=False,
            status_code=None,
            elapsed_ms=elapsed_ms,
            error=f"connect_failed: {e}",
        )
    except Exception as e:
        elapsed_ms = (time.monotonic() - started) * 1000
        return HeartbeatResult(
            url=url,
            delivered=False,
            status_code=None,
            elapsed_ms=elapsed_ms,
            error=f"unexpected: {type(e).__name__}: {e}",
        )
