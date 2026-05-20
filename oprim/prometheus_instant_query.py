"""oprim.prometheus_instant_query — Execute a Prometheus instant query."""
from __future__ import annotations

import httpx
from obase.tool_registry import register_tool
from pydantic import BaseModel, Field

from oprim._exceptions import OprimError


class PrometheusInstantQueryResult(BaseModel):
    query: str
    result_type: str
    samples: list[dict[str, object]] = Field(description="Raw samples with metric labels + value")
    sample_count: int


@register_tool(  # type: ignore[untyped-decorator]
    permission="read",
    requires_secrets=[],
)
def prometheus_instant_query(
    *,
    prom_url: str,
    query: str,
    time_unix: float | None = None,
    timeout_seconds: float = 30.0,
    bearer_token: str | None = None,
) -> PrometheusInstantQueryResult:
    """Execute a Prometheus instant query (PromQL).

    Args:
        prom_url: Prometheus base URL
        query: PromQL query string
        time_unix: Evaluation timestamp (Unix seconds, None = now)
        timeout_seconds: HTTP timeout
        bearer_token: Optional Authorization header value

    Returns:
        PrometheusInstantQueryResult.

    Raises:
        OprimError: HTTP non-2xx / PromQL parse error / timeout.
    """
    url = f"{prom_url.rstrip('/')}/api/v1/query"
    params: dict[str, str] = {"query": query}
    if time_unix is not None:
        params["time"] = str(time_unix)

    headers: dict[str, str] = {}
    if bearer_token is not None:
        headers["Authorization"] = f"Bearer {bearer_token}"

    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.get(url, params=params, headers=headers)
    except httpx.TimeoutException as exc:
        raise OprimError(f"Prometheus query timeout after {timeout_seconds}s: {exc}") from exc
    except httpx.ConnectError as exc:
        raise OprimError(f"Failed to connect to Prometheus: {exc}") from exc

    if not response.is_success:
        try:
            err_data = response.json()
            err_msg = err_data.get("error", response.text[:200])
        except Exception:
            err_msg = response.text[:200]
        raise OprimError(f"Prometheus query failed (HTTP {response.status_code}): {err_msg}")

    data = response.json()
    if data.get("status") != "success":
        raise OprimError(f"Prometheus returned error: {data.get('error', 'unknown')}")

    result_data = data.get("data", {})
    samples = result_data.get("result", [])
    return PrometheusInstantQueryResult(
        query=query,
        result_type=result_data.get("resultType", "vector"),
        samples=samples,
        sample_count=len(samples),
    )
