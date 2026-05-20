"""oprim.docker_logs — Fetch recent logs from a Docker container."""
from __future__ import annotations

import docker
from docker.errors import APIError, DockerException, NotFound
from obase.tool_registry import register_tool
from pydantic import BaseModel, Field

from oprim._exceptions import OprimError


class DockerLogsResult(BaseModel):
    container_id: str
    container_name: str
    lines_fetched: int
    log_lines: list[str] = Field(description="Lines in chronological order")
    stdout_truncated: bool = False
    stderr_truncated: bool = False


@register_tool(  # type: ignore[untyped-decorator]
    permission="read",
    requires_secrets=[],
)
def docker_logs(
    *,
    container: str,
    tail: int = 100,
    since_seconds: int | None = None,
    timestamps: bool = True,
    timeout_seconds: float = 10.0,
    docker_host: str = "unix:///var/run/docker.sock",
) -> DockerLogsResult:
    """Fetch recent logs from a Docker container.

    Args:
        container: Container ID or name
        tail: Number of recent lines to fetch
        since_seconds: Only fetch logs newer than N seconds ago (None = all)
        timestamps: Prefix each line with timestamp
        timeout_seconds: Docker daemon timeout
        docker_host: Docker daemon URL

    Returns:
        DockerLogsResult with log lines.

    Raises:
        OprimError: Container not found / daemon unreachable / permission denied.
    """
    try:
        client = docker.DockerClient(base_url=docker_host, timeout=int(timeout_seconds))
        container_obj = client.containers.get(container)
    except NotFound as exc:
        raise OprimError(f"Container not found: {container!r}: {exc}") from exc
    except DockerException as exc:
        raise OprimError(f"Failed to connect to Docker daemon at {docker_host}: {exc}") from exc

    try:
        logs_kwargs: dict[str, object] = {"tail": tail, "timestamps": timestamps, "stream": False}
        if since_seconds is not None:
            import time

            logs_kwargs["since"] = int(time.time() - since_seconds)

        log_bytes = container_obj.logs(**logs_kwargs)
    except APIError as exc:
        raise OprimError(f"Docker API error fetching logs: {exc}") from exc

    log_text = log_bytes.decode("utf-8", errors="replace")
    log_lines = [line for line in log_text.split("\n") if line.strip()]

    return DockerLogsResult(
        container_id=container_obj.id or "",
        container_name=container_obj.name or container,
        lines_fetched=len(log_lines),
        log_lines=log_lines,
    )
