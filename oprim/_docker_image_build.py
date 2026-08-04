"""oprim.docker_image_build — 单次应用镜像/沙箱镜像构建.

经 docker SDK（与 obase.docker 同源）执行单次镜像构建，返回镜像 ID 与标签。
构建日志可选流式回调。

Example:
    >>> r = await docker_image_build("/repo", tag="myapp:1.0")
    >>> r["status"]
    'ok'
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from oprim._exceptions import OprimError, OprimValidationError


class DockerImageBuildError(OprimError):
    """镜像构建失败。"""


async def docker_image_build(
    context: str | Path,
    *,
    tag: str,
    dockerfile: str | None = None,
    build_args: dict[str, str] | None = None,
    docker_host: str = "unix:///var/run/docker.sock",
    timeout: float = 600.0,
    on_log: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """构建单次镜像。

    Args:
        context: 构建上下文目录。
        tag: 镜像标签（如 "myapp:1.0"）。
        dockerfile: Dockerfile 相对路径/文件名；None 用上下文内默认。
        build_args: 构建参数 (--build-arg)。
        docker_host: docker daemon 地址。
        timeout: 构建超时秒数。
        on_log: 逐条构建日志回调（{stream, message}）。

    Returns:
        {"status": "ok", "image_id": str, "tags": [str], "logs": [dict]}

    Raises:
        DockerImageBuildError: 构建失败 / 上下文缺失 / docker 不可用。
        OprimValidationError: tag 为空。
    """
    ctx = Path(context).expanduser()
    if not tag:
        raise OprimValidationError("docker_image_build: tag must not be empty")
    if not ctx.is_dir():
        raise DockerImageBuildError(f"docker_image_build: context dir not found: {ctx}")

    try:
        import docker  # type: ignore[import-untyped]
        import docker.errors  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - 环境相关
        raise DockerImageBuildError(
            "docker_image_build: docker SDK not installed", cause=exc
        ) from exc

    def _build() -> tuple[str, list[str], list[dict[str, Any]]]:
        client = docker.DockerClient(base_url=docker_host)
        logs: list[dict[str, Any]] = []
        kwargs: dict[str, Any] = {"path": str(ctx), "tag": tag, "rm": True}
        if dockerfile:
            kwargs["dockerfile"] = dockerfile
        if build_args:
            kwargs["buildargs"] = build_args
        try:
            image, _ = client.images.build(**kwargs)
            return image.id, list(image.tags), logs
        except docker.errors.BuildError as exc:
            for chunk in exc.build_log:
                if isinstance(chunk, dict):
                    logs.append(chunk)
            raise
        except docker.errors.DockerException as exc:
            raise DockerImageBuildError(
                f"docker_image_build failed: {type(exc).__name__}: {exc}", cause=exc
            ) from exc

    try:
        image_id, tags, logs = await asyncio.wait_for(
            asyncio.to_thread(_build), timeout=timeout
        )
    except TimeoutError as exc:
        raise DockerImageBuildError(
            f"docker_image_build timed out after {timeout}s", cause=exc
        ) from exc
    except DockerImageBuildError:
        raise

    if on_log:
        for entry in logs:
            try:
                on_log(entry)
            except Exception:  # pragma: no cover - 回调不致命
                pass

    return {"status": "ok", "image_id": image_id, "tags": tags, "logs": logs}
