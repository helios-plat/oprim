"""oprim.loki_query — Execute a Loki LogQL range query."""
from __future__ import annotations

import httpx
from obase.tool_registry import register_tool
from pydantic import BaseModel, Field

from oprim._exceptions import OprimError


class LokiLogLine(BaseModel):
    ts: float = Field(description="Timestamp (Unix seconds)")
    labels: dict[str, str]
    line: str = Field(description="Raw log line text")


class LokiQueryResult(BaseModel):
    query: str
    stream_count: int
    log_lines: list[LokiLogLine]
    total_lines: int
    truncated: bool = Field(description="True if hit limit")


@register_tool(  # type: ignore[untyped-decorator]
    permission="read",
    requires_secrets=[],
)
def loki_query(
    *,
    loki_url: str,
    query: str,
    start_unix: float,
    end_unix: float,
    limit: int = 100,
    direction: str = "backward",
    timeout_seconds: float = 30.0,
) -> LokiQueryResult:
    """Execute a Loki LogQL range query.

    Args:
        loki_url: Loki base URL
        query: LogQL query string
        start_unix: Start timestamp (Unix seconds)
        end_unix: End timestamp (Unix seconds)
        limit: Max lines per request
        direction: "forward" or "backward"
        timeout_seconds: HTTP timeout

    Returns:
        LokiQueryResult with log lines + labels.

    Raises:
        OprimError: HTTP non-2xx / LogQL parse error / timeout.
    """
    url = f"{loki_url.rstrip('/')}/loki/api/v1/query_range"
    params = {
        "query": query,
        "start": str(int(start_unix * 1e9)),
        "end": str(int(end_unix * 1e9)),
        "limit": str(limit),
        "direction": direction,
    }

    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.get(url, params=params)
    except httpx.TimeoutException as exc:
        raise OprimError(f"Loki query timeout after {timeout_seconds}s: {exc}") from exc
    except httpx.ConnectError as exc:
        raise OprimError(f"Failed to connect to Loki: {exc}") from exc

    if not response.is_success:
        try:
            err_data = response.json()
            err_msg = err_data.get("error", response.text[:200])
        except Exception:
            err_msg = response.text[:200]
        raise OprimError(f"Loki query failed (HTTP {response.status_code}): {err_msg}")

    data = response.json()
    if data.get("status") != "success":
        raise OprimError(f"Loki returned error: {data.get('error', 'unknown')}")

    streams = data.get("data", {}).get("result", [])
    log_lines: list[LokiLogLine] = []
    for stream in streams:
        labels = stream.get("stream", {})
        for value_pair in stream.get("values", []):
            ts_ns_str, line_text = value_pair
            log_lines.append(
                LokiLogLine(
                    ts=int(ts_ns_str) / 1e9,
                    labels=labels,
                    line=line_text,
                )
            )

    return LokiQueryResult(
        query=query,
        stream_count=len(streams),
        log_lines=log_lines,
        total_lines=len(log_lines),
        truncated=len(log_lines) >= limit,
    )
