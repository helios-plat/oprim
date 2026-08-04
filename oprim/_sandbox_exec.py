"""oprim.sandbox_exec — 沙箱内单次命令原子执行.

在指定容器（obase.docker_pool / docker_container_create 产物）内执行单条
shell 命令并收集输出。命令退出码非 0 不抛异常（以 dict.exit_code 体现），
仅容器缺失 / 超时 / docker CLI 不可用才抛 SandboxExecError。

Example:
    >>> r = await sandbox_exec("python -c 'print(1+1)'", container_id="c123")
    >>> r["exit_code"]
    0
    >>> r["stdout"].strip()
    '2'
"""

from __future__ import annotations

import asyncio
import shlex
from typing import Any

from oprim._exceptions import OprimError, OprimNotFoundError, OprimTimeoutError


class SandboxExecError(OprimError):
    """沙箱命令执行基础设施失败（docker 不可用等）。"""


async def sandbox_exec(
    command: str,
    *,
    container_id: str,
    timeout: float = 60.0,
    workdir: str | None = None,
    user: str | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """在容器内执行单条命令，返回标准化结果 dict。

    Args:
        command: shell 命令字符串（经 sh -c 在容器内执行）。
        container_id: 目标容器 ID / 名称。
        timeout: 超时秒数（含 docker 层），默认 60。
        workdir: 容器内工作目录，可选。
        user: 容器内执行用户，可选。
        env: 额外环境变量，可选。

    Returns:
        {
            "status": "ok" | "timeout" | "error",
            "exit_code": int | None,
            "stdout": str,
            "stderr": str,
            "elapsed_ms": int,
            "container_id": str,
        }

    Raises:
        SandboxExecError: docker CLI 不可用或容器不存在。
        OprimTimeoutError: 执行超时。
    """
    if not command.strip():
        raise SandboxExecError("sandbox_exec: command must not be empty")
    if timeout <= 0:
        raise SandboxExecError("sandbox_exec: timeout must be > 0")

    # 优先复用 obase docker 原子（同步 SDK 调用经线程池包裹）
    try:
        from obase.docker.containers import docker_container_exec
    except ImportError as exc:  # pragma: no cover - 环境相关
        raise SandboxExecError("sandbox_exec: obase.docker unavailable", cause=exc) from exc

    shell_cmd = ["sh", "-c", command]
    if workdir:
        shell_cmd = ["sh", "-c", f"cd {shlex.quote(workdir)} && {command}"]

    try:
        result = await asyncio.to_thread(
            docker_container_exec,
            container_id=container_id,
            command=shell_cmd,
            env=env,
            user=user,
            timeout_sec=max(1, int(timeout)),
        )
    except Exception as exc:  # noqa: BLE001 - 统一包装
        msg = str(exc)
        if "timeout" in msg.lower():
            raise OprimTimeoutError(
                f"sandbox_exec timed out after {timeout}s: {command[:60]}"
            ) from exc
        if "not found" in msg.lower() or "no such container" in msg.lower():
            raise OprimNotFoundError(f"sandbox container not found: {container_id}") from exc
        raise SandboxExecError(
            f"sandbox_exec failed: {type(exc).__name__}: {msg[:200]}", cause=exc
        ) from exc

    status = "ok" if result.exit_code == 0 else "error"
    return {
        "status": status,
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "elapsed_ms": result.elapsed_ms,
        "container_id": result.container_id,
    }
