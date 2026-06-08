"""Docker oprim — Docker container/image/compose operations."""

from __future__ import annotations

import time
from datetime import UTC
from typing import Any, Literal

import docker  # type: ignore[import-untyped]
import docker.errors  # type: ignore[import-untyped]
from pydantic import BaseModel

from oprim._exceptions import (
    OprimAuthError,
    OprimConnectionError,
    OprimNotFoundError,
)

# ---------------------------------------------------------------------------
# Shared models
# ---------------------------------------------------------------------------


class ContainerInfo(BaseModel):
    container_id: str
    name: str
    image: str
    state: Literal["running", "exited", "paused", "restarting", "dead", "created"]
    status: str
    started_at: str | None
    finished_at: str | None
    exit_code: int | None
    health: Literal["healthy", "unhealthy", "starting", "none"] | None
    restart_count: int
    labels: dict[str, str]
    ports: list[dict[str, Any]]
    mounts: list[dict[str, Any]]


class ContainerOpResult(BaseModel):
    container_id: str
    operation: Literal["start", "stop", "restart"]
    success: bool
    elapsed_ms: int
    state_before: str
    state_after: str


class LogLine(BaseModel):
    timestamp: str
    stream: Literal["stdout", "stderr"]
    message: str


class ImagePullResult(BaseModel):
    image: str
    tag: str
    digest: str
    pulled: bool
    size_bytes: int
    elapsed_ms: int


class ContainerStats(BaseModel):
    container_id: str
    cpu_percent: float
    memory_usage_bytes: int
    memory_limit_bytes: int
    memory_percent: float
    network_rx_bytes: int
    network_tx_bytes: int
    block_read_bytes: int
    block_write_bytes: int
    pids: int
    timestamp: str


class ContainerCreateResult(BaseModel):
    container_id: str
    name: str
    warnings: list[str]


class PruneResult(BaseModel):
    containers_removed: int
    images_removed: int
    volumes_removed: int
    space_reclaimed_bytes: int


class NodeInfo(BaseModel):
    docker_host: str
    reachable: bool
    server_version: str | None
    os: str | None
    arch: str | None
    cpus: int | None
    memory_bytes: int | None
    containers_running: int | None
    error: str | None


class ContainerRenameResult(BaseModel):
    container_id: str
    old_name: str
    new_name: str


class NetworkCreateResult(BaseModel):
    network_id: str
    name: str
    driver: str


class NetworkDeleteResult(BaseModel):
    network_id: str
    name: str
    deleted: bool


class VolumeCreateResult(BaseModel):
    name: str
    driver: str
    mountpoint: str
    created_at: str


class ContainerExecResult(BaseModel):
    container_id: str
    command: list[str]
    exit_code: int
    stdout: str
    stderr: str
    elapsed_ms: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(docker_host: str) -> docker.DockerClient:
    try:
        return docker.DockerClient(base_url=docker_host)
    except docker.errors.DockerException as exc:
        raise OprimConnectionError(
            f"Cannot connect to docker daemon at {docker_host}: {exc}"
        ) from exc


def _get_container(client: docker.DockerClient, container_id: str) -> Any:
    try:
        return client.containers.get(container_id)
    except docker.errors.NotFound as exc:
        raise OprimNotFoundError(f"Container not found: {container_id}") from exc
    except docker.errors.DockerException as exc:
        raise OprimConnectionError(f"Docker error while fetching container: {exc}") from exc


def _parse_state(
    attrs: dict[str, Any],
) -> Literal["running", "exited", "paused", "restarting", "dead", "created"]:
    raw = attrs.get("State", {}).get("Status", "").lower()
    valid = {"running", "exited", "paused", "restarting", "dead", "created"}
    return raw if raw in valid else "exited"


def _parse_health(
    attrs: dict[str, Any],
) -> Literal["healthy", "unhealthy", "starting", "none"] | None:
    health = attrs.get("State", {}).get("Health")
    if health is None:
        return None
    status = health.get("Status", "none").lower()
    valid = {"healthy", "unhealthy", "starting", "none"}
    return status if status in valid else "none"


def _parse_ports(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    ports = []
    bindings = attrs.get("HostConfig", {}).get("PortBindings") or {}
    for container_port, host_bindings in bindings.items():
        proto = "tcp"
        cp = container_port
        if "/" in container_port:
            cp, proto = container_port.split("/", 1)
        for hb in host_bindings or []:
            ports.append(
                {
                    "host_port": hb.get("HostPort"),
                    "container_port": cp,
                    "protocol": proto,
                }
            )
    return ports


def _parse_mounts(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "type": m.get("Type"),
            "source": m.get("Source"),
            "destination": m.get("Destination"),
            "mode": m.get("Mode"),
            "rw": m.get("RW"),
        }
        for m in attrs.get("Mounts", [])
    ]


# ---------------------------------------------------------------------------
# 2.1 docker_container_inspect
# ---------------------------------------------------------------------------


def docker_container_inspect(
    *,
    container_id: str,
    docker_host: str = "unix:///var/run/docker.sock",
) -> ContainerInfo:
    """查容器完整状态信息.

    Args:
        container_id: 容器 ID 或 name
        docker_host: docker daemon 地址, 默认本机 socket

    Returns:
        ContainerInfo 含状态 / 启停时间 / 健康 / 端口等

    Raises:
        OprimNotFoundError: container_id 不存在
        OprimConnectionError: docker daemon 不可达
    """
    client = _make_client(docker_host)
    container = _get_container(client, container_id)
    container.reload()
    attrs = container.attrs

    state_attrs = attrs.get("State", {})
    started = state_attrs.get("StartedAt") or None
    finished = state_attrs.get("FinishedAt") or None
    # Docker returns "0001-01-01T00:00:00Z" for never-started containers
    if started and started.startswith("0001"):
        started = None
    if finished and finished.startswith("0001"):
        finished = None

    exit_code = state_attrs.get("ExitCode")

    return ContainerInfo(
        container_id=attrs["Id"],
        name=attrs.get("Name", "").lstrip("/"),
        image=attrs.get("Config", {}).get("Image", ""),
        state=_parse_state(attrs),
        status=container.status,
        started_at=started,
        finished_at=finished,
        exit_code=exit_code,
        health=_parse_health(attrs),
        restart_count=attrs.get("RestartCount", 0),
        labels=attrs.get("Config", {}).get("Labels") or {},
        ports=_parse_ports(attrs),
        mounts=_parse_mounts(attrs),
    )


# ---------------------------------------------------------------------------
# 2.2 docker_container_logs
# ---------------------------------------------------------------------------


def docker_container_logs(
    *,
    container_id: str,
    lines: int = 100,
    since: str | None = None,
    until: str | None = None,
    docker_host: str = "unix:///var/run/docker.sock",
) -> list[LogLine]:
    """读容器日志.

    Args:
        container_id: 容器 ID 或 name
        lines: 取最近 N 行 (since 为 None 时生效)
        since: 起始时间 (ISO 8601 或 "5m" / "1h" relative)
        until: 截止时间
        docker_host: docker daemon 地址

    Returns:
        日志行列表, 按时间升序

    Raises:
        OprimNotFoundError / OprimConnectionError
    """
    client = _make_client(docker_host)
    container = _get_container(client, container_id)

    kwargs: dict[str, Any] = {"timestamps": True, "stream": False}
    if since is not None:
        kwargs["since"] = since
    if until is not None:
        kwargs["until"] = until
    if since is None:
        kwargs["tail"] = lines

    try:
        raw: bytes = container.logs(**kwargs)
    except docker.errors.DockerException as exc:
        raise OprimConnectionError(f"Failed to retrieve logs: {exc}") from exc

    result: list[LogLine] = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if not line:
            continue
        # Docker log format with timestamps: "2026-05-20T10:00:00.000000000Z message"
        parts = line.split(" ", 1)
        if len(parts) == 2:
            ts, msg = parts
        else:
            ts, msg = "", line
        result.append(
            LogLine(
                timestamp=ts,
                stream="stdout",  # Docker multiplexed stream detection requires binary parsing
                message=msg,
            )
        )
    return result


# ---------------------------------------------------------------------------
# 2.3 docker_container_start
# ---------------------------------------------------------------------------


def docker_container_start(
    *,
    container_id: str,
    docker_host: str = "unix:///var/run/docker.sock",
) -> ContainerOpResult:
    """启动容器.

    Args:
        container_id: 容器 ID 或 name
        docker_host: docker daemon 地址

    Returns:
        ContainerOpResult 含 state 变化

    Raises:
        OprimNotFoundError / OprimConnectionError
    """
    client = _make_client(docker_host)
    container = _get_container(client, container_id)

    state_before = container.status
    t0 = time.monotonic()
    try:
        container.start()
    except docker.errors.DockerException as exc:
        raise OprimConnectionError(f"Failed to start container: {exc}") from exc
    elapsed = int((time.monotonic() - t0) * 1000)
    container.reload()
    state_after = container.status

    return ContainerOpResult(
        container_id=container.id,
        operation="start",
        success=True,
        elapsed_ms=elapsed,
        state_before=state_before,
        state_after=state_after,
    )


# ---------------------------------------------------------------------------
# 2.4 docker_container_stop
# ---------------------------------------------------------------------------


def docker_container_stop(
    *,
    container_id: str,
    timeout_sec: int = 10,
    docker_host: str = "unix:///var/run/docker.sock",
) -> ContainerOpResult:
    """停止容器 (优雅停止, 超时后强杀).

    Args:
        container_id: 容器 ID 或 name
        timeout_sec: SIGTERM 后等待秒数, 超时后 SIGKILL
        docker_host: docker daemon 地址

    Returns:
        ContainerOpResult

    Raises:
        OprimNotFoundError / OprimConnectionError
    """
    client = _make_client(docker_host)
    container = _get_container(client, container_id)

    state_before = container.status
    t0 = time.monotonic()
    try:
        container.stop(timeout=timeout_sec)
    except docker.errors.DockerException as exc:
        raise OprimConnectionError(f"Failed to stop container: {exc}") from exc
    elapsed = int((time.monotonic() - t0) * 1000)
    container.reload()
    state_after = container.status

    return ContainerOpResult(
        container_id=container.id,
        operation="stop",
        success=True,
        elapsed_ms=elapsed,
        state_before=state_before,
        state_after=state_after,
    )


# ---------------------------------------------------------------------------
# 2.5 docker_container_restart
# ---------------------------------------------------------------------------


def docker_container_restart(
    *,
    container_id: str,
    timeout_sec: int = 10,
    docker_host: str = "unix:///var/run/docker.sock",
) -> ContainerOpResult:
    """重启容器 (= stop + start).

    Args:
        container_id: 容器 ID 或 name
        timeout_sec: stop 阶段等待秒数
        docker_host: docker daemon 地址

    Returns:
        ContainerOpResult

    Raises:
        OprimNotFoundError / OprimConnectionError
    """
    client = _make_client(docker_host)
    container = _get_container(client, container_id)

    state_before = container.status
    t0 = time.monotonic()
    try:
        container.restart(timeout=timeout_sec)
    except docker.errors.DockerException as exc:
        raise OprimConnectionError(f"Failed to restart container: {exc}") from exc
    elapsed = int((time.monotonic() - t0) * 1000)
    container.reload()
    state_after = container.status

    return ContainerOpResult(
        container_id=container.id,
        operation="restart",
        success=True,
        elapsed_ms=elapsed,
        state_before=state_before,
        state_after=state_after,
    )


# ---------------------------------------------------------------------------
# 2.6 docker_image_pull
# ---------------------------------------------------------------------------


def docker_image_pull(
    *,
    image: str,
    tag: str = "latest",
    docker_host: str = "unix:///var/run/docker.sock",
    auth_config: dict[str, Any] | None = None,
) -> ImagePullResult:
    """拉取 docker 镜像.

    Args:
        image: 镜像名 (e.g. "nginx", "myregistry.io/myapp")
        tag: 标签, 默认 "latest"
        docker_host: docker daemon 地址
        auth_config: 私有仓库认证 (可选) {"username": ..., "password": ...}

    Returns:
        ImagePullResult

    Raises:
        OprimNotFoundError: 镜像 / tag 不存在
        OprimAuthError: 私有仓库认证失败
        OprimConnectionError: docker daemon 或 registry 不可达
    """
    client = _make_client(docker_host)
    ref = f"{image}:{tag}"

    # Check if already local
    already_local = False
    try:
        client.images.get(ref)
        already_local = True
    except docker.errors.ImageNotFound:
        pass
    except docker.errors.DockerException as exc:
        raise OprimConnectionError(f"Docker error checking local image: {exc}") from exc

    t0 = time.monotonic()
    try:
        img = client.images.pull(image, tag=tag, auth_config=auth_config)
    except docker.errors.ImageNotFound as exc:
        raise OprimNotFoundError(f"Image not found: {ref}") from exc
    except docker.errors.APIError as exc:
        msg = str(exc)
        if "unauthorized" in msg.lower() or "authentication" in msg.lower() or "403" in msg:
            raise OprimAuthError(f"Authentication failed for {ref}: {exc}") from exc
        raise OprimConnectionError(f"Failed to pull image {ref}: {exc}") from exc
    except docker.errors.DockerException as exc:
        raise OprimConnectionError(f"Failed to pull image {ref}: {exc}") from exc
    elapsed = int((time.monotonic() - t0) * 1000)

    digest = img.id or ""
    size_bytes = img.attrs.get("Size", 0)

    return ImagePullResult(
        image=image,
        tag=tag,
        digest=digest,
        pulled=not already_local,
        size_bytes=size_bytes,
        elapsed_ms=elapsed,
    )


# ---------------------------------------------------------------------------
# 2.7 docker_container_stats
# ---------------------------------------------------------------------------


def docker_container_stats(
    *,
    container_id: str,
    docker_host: str = "unix:///var/run/docker.sock",
) -> ContainerStats:
    """读容器资源使用快照 (单次, 非流式).

    Returns:
        ContainerStats 含 CPU / 内存 / 网络 / IO

    Raises:
        OprimNotFoundError / OprimConnectionError
    """
    from datetime import datetime

    client = _make_client(docker_host)
    container = _get_container(client, container_id)

    try:
        raw = container.stats(stream=False)
    except docker.errors.DockerException as exc:
        raise OprimConnectionError(f"Failed to get container stats: {exc}") from exc

    # CPU %
    cpu_delta = raw.get("cpu_stats", {}).get("cpu_usage", {}).get("total_usage", 0) - raw.get(
        "precpu_stats", {}
    ).get("cpu_usage", {}).get("total_usage", 0)
    system_delta = raw.get("cpu_stats", {}).get("system_cpu_usage", 0) - raw.get(
        "precpu_stats", {}
    ).get("system_cpu_usage", 0)
    percpu = raw.get("cpu_stats", {}).get("cpu_usage", {}).get("percpu_usage") or [0]
    num_cpus = len(percpu)
    cpu_percent = (cpu_delta / system_delta * num_cpus * 100.0) if system_delta > 0 else 0.0

    # Memory
    mem = raw.get("memory_stats", {})
    mem_usage = mem.get("usage", 0)
    mem_limit = mem.get("limit", 0)
    mem_percent = (mem_usage / mem_limit * 100.0) if mem_limit > 0 else 0.0

    # Network (sum across all interfaces)
    net_rx = net_tx = 0
    for iface in raw.get("networks", {}).values():
        net_rx += iface.get("rx_bytes", 0)
        net_tx += iface.get("tx_bytes", 0)

    # Block IO
    blk_read = blk_write = 0
    for bio in raw.get("blkio_stats", {}).get("io_service_bytes_recursive") or []:
        op = bio.get("op", "").lower()
        if op == "read":
            blk_read += bio.get("value", 0)
        elif op == "write":
            blk_write += bio.get("value", 0)

    # PIDs
    pids = raw.get("pids_stats", {}).get("current", 0)

    return ContainerStats(
        container_id=container_id,
        cpu_percent=round(cpu_percent, 2),
        memory_usage_bytes=mem_usage,
        memory_limit_bytes=mem_limit,
        memory_percent=round(mem_percent, 2),
        network_rx_bytes=net_rx,
        network_tx_bytes=net_tx,
        block_read_bytes=blk_read,
        block_write_bytes=blk_write,
        pids=pids,
        timestamp=datetime.now(UTC).isoformat(),
    )


# ---------------------------------------------------------------------------
# 2.8 docker_image_list
# ---------------------------------------------------------------------------


def docker_image_list(
    *,
    docker_host: str = "unix:///var/run/docker.sock",
) -> list[dict[str, Any]]:
    """列出 docker 镜像.

    Returns:
        镜像列表, 每个含 id, tags, size_bytes, created_at

    Raises:
        OprimConnectionError
    """
    client = _make_client(docker_host)
    try:
        images = client.images.list()
        return [
            {
                "id": img.id,
                "tags": img.tags,
                "size_bytes": img.attrs.get("Size", 0),
                "created_at": img.attrs.get("Created", ""),
            }
            for img in images
        ]
    except docker.errors.DockerException as exc:
        raise OprimConnectionError(f"Docker error listing images: {exc}") from exc


# ---------------------------------------------------------------------------
# 2.9 docker_image_delete
# ---------------------------------------------------------------------------


def docker_image_delete(
    *,
    image: str,
    force: bool = False,
    docker_host: str = "unix:///var/run/docker.sock",
) -> dict[str, Any]:
    """删除 docker 镜像.

    Returns:
        {"deleted": [...], "untagged": [...]}

    Raises:
        OprimNotFoundError / OprimConnectionError
    """
    client = _make_client(docker_host)
    try:
        # returns list of dicts, but let's wrap it for consistency if needed
        # Actually docker-py returns exactly what the API returns
        res = client.images.remove(image, force=force)
        return {"result": res}
    except docker.errors.ImageNotFound as exc:
        raise OprimNotFoundError(f"Image not found: {image}") from exc
    except docker.errors.DockerException as exc:
        raise OprimConnectionError(f"Docker error deleting image: {exc}") from exc


# ---------------------------------------------------------------------------
# 2.10 docker_volume_list
# ---------------------------------------------------------------------------


def docker_volume_list(
    *,
    docker_host: str = "unix:///var/run/docker.sock",
) -> list[dict[str, Any]]:
    """列出 docker 数据卷.

    Raises:
        OprimConnectionError
    """
    client = _make_client(docker_host)
    try:
        volumes = client.volumes.list()
        return [
            {
                "name": vol.name,
                "driver": vol.attrs.get("Driver", ""),
                "mountpoint": vol.attrs.get("Mountpoint", ""),
                "created_at": vol.attrs.get("CreatedAt", ""),
                "labels": vol.attrs.get("Labels") or {},
            }
            for vol in volumes
        ]
    except docker.errors.DockerException as exc:
        raise OprimConnectionError(f"Docker error listing volumes: {exc}") from exc


# ---------------------------------------------------------------------------
# 2.11 docker_volume_delete
# ---------------------------------------------------------------------------


def docker_volume_delete(
    *,
    name: str,
    force: bool = False,
    docker_host: str = "unix:///var/run/docker.sock",
) -> dict[str, Any]:
    """删除 docker 数据卷.

    Raises:
        OprimNotFoundError / OprimConnectionError
    """
    client = _make_client(docker_host)
    try:
        vol = client.volumes.get(name)
        vol.remove(force=force)
        return {"deleted": name}
    except docker.errors.NotFound as exc:
        raise OprimNotFoundError(f"Volume not found: {name}") from exc
    except docker.errors.DockerException as exc:
        raise OprimConnectionError(f"Docker error deleting volume: {exc}") from exc


# ---------------------------------------------------------------------------
# 2.12 docker_network_list
# ---------------------------------------------------------------------------


def docker_network_list(
    *,
    docker_host: str = "unix:///var/run/docker.sock",
) -> list[dict[str, Any]]:
    """列出 docker 网络.

    Raises:
        OprimConnectionError
    """
    client = _make_client(docker_host)
    try:
        networks = client.networks.list()
        return [
            {
                "id": net.id,
                "name": net.name,
                "driver": net.attrs.get("Driver", ""),
                "scope": net.attrs.get("Scope", ""),
                "internal": net.attrs.get("Internal", False),
            }
            for net in networks
        ]
    except docker.errors.DockerException as exc:
        raise OprimConnectionError(f"Docker error listing networks: {exc}") from exc


# ---------------------------------------------------------------------------
# 2.13 compose_up
# ---------------------------------------------------------------------------


def compose_up(
    *,
    compose_file: str,
    project_name: str | None = None,
    detach: bool = True,
    pull: Literal["always", "missing", "never"] = "missing",
    docker_host: str = "unix:///var/run/docker.sock",
) -> dict[str, Any]:
    """docker-compose up.

    Args:
        compose_file: path to docker-compose.yml
        project_name: project name (optional)
        detach: run in background
        pull: pull policy
        docker_host: docker host address

    Returns:
        {"started_services": list[str], "stdout": str, "stderr": str}

    Raises:
        OprimNotFoundError: compose file not found
        OprimConnectionError: docker compose command failed
    """
    import os
    import subprocess

    if not os.path.exists(compose_file):
        raise OprimNotFoundError(f"Compose file not found: {compose_file}")

    cmd = ["docker", "compose", "-f", compose_file]
    if project_name:
        cmd.extend(["-p", project_name])
    cmd.extend(["up", "--pull", pull])
    if detach:
        cmd.append("-d")

    env = os.environ.copy()
    env["DOCKER_HOST"] = docker_host

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        # Try to parse started services from stderr (docker compose output goes to stderr often)
        started_services: list[str] = []
        for line in proc.stderr.splitlines():
            if "Started" in line or "Created" in line:
                # Example: " Container project-service-1  Started"
                parts = line.split()
                if len(parts) >= 2:
                    started_services.append(parts[1])

        return {
            "started_services": sorted(list(set(started_services))),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except subprocess.CalledProcessError as exc:
        msg = f"Docker compose up failed (exit {exc.returncode}): {exc.stderr}"
        raise OprimConnectionError(msg) from exc
    except Exception as exc:
        raise OprimConnectionError(f"Failed to execute docker compose: {exc}") from exc


# ---------------------------------------------------------------------------
# 2.14 compose_down
# ---------------------------------------------------------------------------


def compose_down(
    *,
    compose_file: str,
    project_name: str | None = None,
    volumes: bool = False,
    remove_orphans: bool = True,
    docker_host: str = "unix:///var/run/docker.sock",
) -> dict[str, Any]:
    """docker-compose down.

    Args:
        compose_file: path to docker-compose.yml
        project_name: project name
        volumes: remove volumes
        remove_orphans: remove orphan containers
        docker_host: docker host address

    Returns:
        {"stdout": str, "stderr": str}

    Raises:
        OprimNotFoundError / OprimConnectionError
    """
    import os
    import subprocess

    if not os.path.exists(compose_file):
        raise OprimNotFoundError(f"Compose file not found: {compose_file}")

    cmd = ["docker", "compose", "-f", compose_file]
    if project_name:
        cmd.extend(["-p", project_name])
    cmd.append("down")
    if volumes:
        cmd.append("-v")
    if remove_orphans:
        cmd.append("--remove-orphans")

    env = os.environ.copy()
    env["DOCKER_HOST"] = docker_host

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except subprocess.CalledProcessError as exc:
        msg = f"Docker compose down failed (exit {exc.returncode}): {exc.stderr}"
        raise OprimConnectionError(msg) from exc
    except Exception as exc:
        raise OprimConnectionError(f"Failed to execute docker compose: {exc}") from exc


# ---------------------------------------------------------------------------
# 2.15 docker_container_list
# ---------------------------------------------------------------------------


def docker_container_list(
    *,
    all: bool = False,
    filters: dict[str, Any] | None = None,
    docker_host: str = "unix:///var/run/docker.sock",
) -> list[ContainerInfo]:
    """列出 docker 容器.

    Args:
        all: 是否列出所有容器 (默认仅运行中)
        filters: 过滤器 (e.g. {"label": ["foo=bar"]})
        docker_host: docker host

    Returns:
        ContainerInfo 列表

    Raises:
        OprimConnectionError
    """
    client = _make_client(docker_host)
    try:
        containers = client.containers.list(all=all, filters=filters)
        return [
            ContainerInfo(
                container_id=c.id,
                name=c.name,
                image=c.image.tags[0] if c.image.tags else c.image.id,
                state=_parse_state(c.attrs),
                status=c.status,
                started_at=c.attrs.get("State", {}).get("StartedAt"),
                finished_at=c.attrs.get("State", {}).get("FinishedAt"),
                exit_code=c.attrs.get("State", {}).get("ExitCode"),
                health=_parse_health(c.attrs),
                restart_count=c.attrs.get("RestartCount", 0),
                labels=c.attrs.get("Config", {}).get("Labels") or {},
                ports=_parse_ports(c.attrs),
                mounts=_parse_mounts(c.attrs),
            )
            for c in containers
        ]
    except docker.errors.DockerException as exc:
        raise OprimConnectionError(f"Docker error listing containers: {exc}") from exc


# ---------------------------------------------------------------------------
# 2.16 docker_container_create
# ---------------------------------------------------------------------------


def docker_container_create(
    *,
    image: str,
    name: str,
    command: list[str] | None = None,
    env: dict[str, str] | None = None,
    ports: dict[str, int | list[int] | None] | None = None,
    volumes: dict[str, dict[str, str]] | None = None,
    labels: dict[str, str] | None = None,
    restart_policy: Literal["no", "always", "on-failure", "unless-stopped"] = "unless-stopped",
    network: str | None = None,
    docker_host: str = "unix:///var/run/docker.sock",
) -> ContainerCreateResult:
    """创建容器 (不启动).

    Args:
        image: 镜像名:tag
        name: 容器名
        command: 覆盖 CMD (可选)
        env: 环境变量 dict
        ports: 端口映射 {"80/tcp": 8080} 或 {"80/tcp": [8080, 8081]}
        volumes: volume 挂载 {"/host/path": {"bind": "/container/path", "mode": "rw"}}
        labels: 容器标签
        restart_policy: 重启策略
        network: 加入的 docker 网络名 (可选)
        docker_host: docker daemon 地址

    Returns:
        ContainerCreateResult

    Raises:
        OprimNotFoundError: image 不存在
        OprimConnectionError: daemon 不可达
        OprimValidationError: 参数无效 (端口冲突等)
    """
    from oprim._exceptions import OprimValidationError

    client = _make_client(docker_host)
    kwargs: dict[str, Any] = {
        "image": image,
        "name": name,
        "detach": True,
        "restart_policy": {"Name": restart_policy},
    }
    if command:
        kwargs["command"] = command
    if env:
        kwargs["environment"] = env
    if ports:
        kwargs["ports"] = ports
    if volumes:
        kwargs["volumes"] = volumes
    if labels:
        kwargs["labels"] = labels
    if network:
        kwargs["network"] = network

    try:
        container = client.containers.create(**kwargs)
        return ContainerCreateResult(
            container_id=container.id,
            name=container.name,
            warnings=container.attrs.get("Warnings") or [],
        )
    except docker.errors.ImageNotFound as exc:
        raise OprimNotFoundError(f"Image not found: {image}") from exc
    except docker.errors.APIError as exc:
        msg = str(exc)
        if "port is already allocated" in msg or "bind" in msg.lower():
            raise OprimValidationError(
                f"Container create failed (port conflict?): {msg[:300]}"
            ) from exc
        raise OprimConnectionError(f"Docker API error: {msg[:300]}") from exc


# ---------------------------------------------------------------------------
# 2.17 docker_system_prune
# ---------------------------------------------------------------------------


def docker_system_prune(
    *,
    volumes: bool = False,
    docker_host: str = "unix:///var/run/docker.sock",
) -> PruneResult:
    """清理停止的容器、悬空镜像、未使用网络 (可选: volumes).

    Args:
        volumes: 是否同时清理未使用 volume (危险! 默认 False)
        docker_host: docker daemon 地址

    Returns:
        PruneResult

    Raises:
        OprimConnectionError
    """
    client = _make_client(docker_host)
    try:
        c_result = client.containers.prune()
        i_result = client.images.prune(filters={"dangling": True})
        v_removed = 0
        if volumes:
            v_result = client.volumes.prune()
            v_removed = len(v_result.get("VolumesDeleted") or [])

        containers_removed = len(c_result.get("ContainersDeleted") or [])
        images_removed = len(i_result.get("ImagesDeleted") or [])
        space = (c_result.get("SpaceReclaimed", 0) or 0) + (i_result.get("SpaceReclaimed", 0) or 0)

        return PruneResult(
            containers_removed=containers_removed,
            images_removed=images_removed,
            volumes_removed=v_removed,
            space_reclaimed_bytes=space,
        )
    except docker.errors.DockerException as exc:
        raise OprimConnectionError(f"Docker prune failed: {exc}") from exc


# ---------------------------------------------------------------------------
# 2.18 docker_node_info
# ---------------------------------------------------------------------------


def docker_node_info(
    *,
    docker_host: str,
    timeout_sec: int = 5,
) -> NodeInfo:
    """探测远程 Docker 节点基本信息 (多主机管理核心).

    永不 raise 网络错误 (返 reachable=False).

    Args:
        docker_host: 远程 docker daemon 地址 (e.g. "tcp://192.168.1.10:2375")
        timeout_sec: 连接超时

    Returns:
        NodeInfo
    """
    import docker as docker_lib

    try:
        client = docker_lib.DockerClient(base_url=docker_host, timeout=timeout_sec)
        info = client.info()
        return NodeInfo(
            docker_host=docker_host,
            reachable=True,
            server_version=client.version().get("Version"),
            os=info.get("OperatingSystem"),
            arch=info.get("Architecture"),
            cpus=info.get("NCPU"),
            memory_bytes=info.get("MemTotal"),
            containers_running=info.get("ContainersRunning"),
            error=None,
        )
    except Exception as exc:
        return NodeInfo(
            docker_host=docker_host,
            reachable=False,
            server_version=None,
            os=None,
            arch=None,
            cpus=None,
            memory_bytes=None,
            containers_running=None,
            error=str(exc)[:200],
        )


# ---------------------------------------------------------------------------
# 2.19 docker_container_rename
# ---------------------------------------------------------------------------


def docker_container_rename(
    *,
    container_id: str,
    new_name: str,
    docker_host: str = "unix:///var/run/docker.sock",
) -> ContainerRenameResult:
    """重命名容器.

    Args:
        container_id: 容器 ID 或当前 name
        new_name: 新名称
        docker_host: docker daemon 地址

    Returns:
        ContainerRenameResult

    Raises:
        OprimNotFoundError: 容器不存在
        OprimValidationError: 名称冲突
        OprimConnectionError: API 错误
    """
    from oprim._exceptions import OprimValidationError

    client = _make_client(docker_host)
    container = _get_container(client, container_id)
    old_name = container.name.lstrip("/")

    try:
        container.rename(new_name)
        return ContainerRenameResult(
            container_id=container.id,
            old_name=old_name,
            new_name=new_name,
        )
    except docker.errors.APIError as exc:
        msg = str(exc)
        if "Conflict" in msg or "already in use" in msg.lower():
            raise OprimValidationError(f"Rename failed (conflict): {msg[:300]}") from exc
        raise OprimConnectionError(f"Docker API error renaming container: {msg[:300]}") from exc
    except docker.errors.DockerException as exc:
        raise OprimConnectionError(f"Docker error renaming container: {exc}") from exc


# ---------------------------------------------------------------------------
# Aegis IMPL SPEC v1.0 — short-name aliases + docker_compose_pull
# ---------------------------------------------------------------------------

# Short-name aliases (B2 API surface — oskill elements reference these names)
docker_logs = docker_container_logs
docker_ps = docker_container_list
docker_restart = docker_container_restart
docker_stats = docker_container_stats
docker_inspect = docker_container_inspect
docker_compose_up = compose_up
docker_compose_down = compose_down


def docker_compose_pull(
    *,
    compose_file: str,
    project_name: str | None = None,
    docker_host: str = "unix:///var/run/docker.sock",
) -> dict[str, Any]:
    """docker compose pull — 预拉 compose 文件中所有服务镜像.

    Args:
        compose_file: path to docker-compose.yml
        project_name: project name (optional)
        docker_host: docker host address

    Returns:
        {"stdout": str, "stderr": str}

    Raises:
        OprimNotFoundError: compose file not found
        OprimConnectionError: docker compose command failed
    """
    import os
    import subprocess

    if not os.path.exists(compose_file):
        raise OprimNotFoundError(f"Compose file not found: {compose_file}")

    cmd = ["docker", "compose", "-f", compose_file]
    if project_name:
        cmd.extend(["-p", project_name])
    cmd.append("pull")

    env = os.environ.copy()
    env["DOCKER_HOST"] = docker_host

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        return {"stdout": proc.stdout, "stderr": proc.stderr}
    except subprocess.CalledProcessError as exc:
        msg = f"Docker compose pull failed (exit {exc.returncode}): {exc.stderr}"
        raise OprimConnectionError(msg) from exc
    except Exception as exc:
        raise OprimConnectionError(f"Failed to execute docker compose pull: {exc}") from exc


# ---------------------------------------------------------------------------
# 2.20 docker_network_create
# ---------------------------------------------------------------------------


def docker_network_create(
    *,
    name: str,
    driver: Literal["bridge", "host", "overlay", "macvlan", "none"] = "bridge",
    internal: bool = False,
    labels: dict[str, str] | None = None,
    options: dict[str, str] | None = None,
    docker_host: str = "unix:///var/run/docker.sock",
) -> NetworkCreateResult:
    """创建 docker 网络.

    Args:
        name: 网络名称
        driver: 网络驱动
        internal: 是否内部网络
        labels: 网络标签
        options: 驱动选项
        docker_host: docker daemon 地址

    Returns:
        NetworkCreateResult

    Raises:
        OprimValidationError: 网络已存在
        OprimConnectionError: API 错误
    """
    from oprim._exceptions import OprimValidationError

    client = _make_client(docker_host)
    try:
        net = client.networks.create(
            name=name, driver=driver, internal=internal, labels=labels, options=options
        )
        return NetworkCreateResult(
            network_id=net.id,
            name=net.name,
            driver=net.attrs.get("Driver", ""),
        )
    except docker.errors.APIError as exc:
        if "already exists" in str(exc).lower():
            raise OprimValidationError(f"Network already exists: {name}") from exc
        raise OprimConnectionError(f"Docker API error creating network: {exc}") from exc


# ---------------------------------------------------------------------------
# 2.21 docker_network_delete
# ---------------------------------------------------------------------------


def docker_network_delete(
    *,
    network_id: str,
    docker_host: str = "unix:///var/run/docker.sock",
) -> NetworkDeleteResult:
    """删除 docker 网络.

    Args:
        network_id: 网络 ID 或 name
        docker_host: docker daemon 地址

    Returns:
        NetworkDeleteResult

    Raises:
        OprimNotFoundError: 网络不存在
        OprimValidationError: 网络有容器在使用, 不能删除
        OprimConnectionError: API 错误
    """
    from oprim._exceptions import OprimValidationError

    client = _make_client(docker_host)
    try:
        net = client.networks.get(network_id)
        name = net.name
        net.remove()
        return NetworkDeleteResult(
            network_id=network_id,
            name=name,
            deleted=True,
        )
    except docker.errors.NotFound as exc:
        raise OprimNotFoundError(f"Network not found: {network_id}") from exc
    except docker.errors.APIError as exc:
        if "active endpoints" in str(exc).lower():
            raise OprimValidationError(f"Network {network_id} has active endpoints") from exc
        raise OprimConnectionError(f"Docker API error deleting network: {exc}") from exc


# ---------------------------------------------------------------------------
# 2.22 docker_volume_create
# ---------------------------------------------------------------------------


def docker_volume_create(
    *,
    name: str,
    driver: str = "local",
    labels: dict[str, str] | None = None,
    driver_opts: dict[str, str] | None = None,
    docker_host: str = "unix:///var/run/docker.sock",
) -> VolumeCreateResult:
    """创建 docker 数据卷.

    Args:
        name: 卷名称
        driver: 卷驱动, 默认 local
        labels: 卷标签
        driver_opts: 驱动选项
        docker_host: docker daemon 地址

    Returns:
        VolumeCreateResult

    Raises:
        OprimValidationError: 数据卷已存在
        OprimConnectionError: API 错误
    """
    from oprim._exceptions import OprimValidationError

    client = _make_client(docker_host)
    try:
        vol = client.volumes.create(
            name=name, driver=driver, labels=labels, driver_opts=driver_opts
        )
        return VolumeCreateResult(
            name=vol.name,
            driver=vol.attrs.get("Driver", ""),
            mountpoint=vol.attrs.get("Mountpoint", ""),
            created_at=vol.attrs.get("CreatedAt", ""),
        )
    except docker.errors.APIError as exc:
        if "already exists" in str(exc).lower():
            raise OprimValidationError(f"Volume already exists: {name}") from exc
        raise OprimConnectionError(f"Docker API error creating volume: {exc}") from exc


# ---------------------------------------------------------------------------
# 2.23 docker_container_exec
# ---------------------------------------------------------------------------


def docker_container_exec(
    *,
    container_id: str,
    command: list[str],
    workdir: str | None = None,
    env: dict[str, str] | None = None,
    user: str | None = None,
    timeout_sec: int = 30,
    docker_host: str = "unix:///var/run/docker.sock",
) -> ContainerExecResult:
    """在容器内执行命令.

    Args:
        container_id: 容器 ID 或 name
        command: 命令列表
        workdir: 工作目录
        env: 环境变量
        user: 执行用户
        timeout_sec: 超时秒数
        docker_host: docker daemon 地址

    Returns:
        ContainerExecResult

    Raises:
        OprimNotFoundError: 容器不存在
        OprimConnectionError: API 错误或执行超时
    """
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FutureTimeoutError

    client = _make_client(docker_host)
    container = _get_container(client, container_id)

    t0 = time.perf_counter()

    def _do_exec() -> tuple[int, tuple[bytes | None, bytes | None]]:
        return container.exec_run(
            cmd=command,
            workdir=workdir,
            environment=env,
            user=user,
            demux=True,
        )

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_do_exec)
            exit_code, (stdout_bytes, stderr_bytes) = future.result(timeout=timeout_sec)

        stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
        stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
        elapsed = int((time.perf_counter() - t0) * 1000)

        return ContainerExecResult(
            container_id=container.id,
            command=command,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            elapsed_ms=elapsed,
        )
    except FutureTimeoutError as exc:
        raise OprimConnectionError(f"Exec timeout ({timeout_sec}s): {command}") from exc
    except docker.errors.APIError as exc:
        raise OprimConnectionError(f"Docker API error executing command: {exc}") from exc
    except Exception as exc:
        raise OprimConnectionError(f"Unexpected error executing command: {exc}") from exc
