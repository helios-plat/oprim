"""oprim._sandbox_execute_probe — 单次在隔离沙箱中执行探针命令并收集结果.

铁律: ≤1 个位置参数，其余 keyword-only.

组合: LocalSandboxPool (Protocol 注入).

例:
    >>> result = _sandbox_execute_probe("echo hello", sandbox_op=my_pool)
    >>> result["exit_code"]
    0
"""

from __future__ import annotations

from typing import Any, Protocol


class SandboxPoolProtocol(Protocol):
    """沙箱池 Protocol — 不 import obase，由调用方注入."""

    def execute(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
        stdin_data: str | None = None,
    ) -> Any: ...


def _sandbox_execute_probe(
    command: str,
    *,
    sandbox_op: SandboxPoolProtocol | None = None,
    cwd: str | None = None,
    timeout: int = 30,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """单次在隔离沙箱中执行探针命令并收集结构化结果.

    签名遵循 oprim 铁律：最多 1 个位置参数，其余 kw-only.

    Args:
        command: 要执行的 shell 命令
        sandbox_op: 沙箱池适配器（注入 SandboxPoolProtocol）
        cwd: 工作目录
        timeout: 超时秒数
        env: 环境变量

    Returns:
        {
            "status": "success" | "timeout" | "error",
            "stdout": str,
            "stderr": str,
            "exit_code": int,
        }
    """
    try:
        if sandbox_op is not None:
            result = sandbox_op.execute(
                command,
                cwd=cwd,
                timeout=timeout,
                env=env,
            )
            return {
                "status": "timeout" if getattr(result, "timed_out", False) else "success",
                "stdout": getattr(result, "stdout", ""),
                "stderr": getattr(result, "stderr", ""),
                "exit_code": getattr(result, "exit_code", -1),
            }

        # 回退：直接 subprocess
        import subprocess

        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            return {
                "status": "success",
                "stdout": proc.stdout[-5000:],
                "stderr": proc.stderr[-5000:],
                "exit_code": proc.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "stdout": "",
                "stderr": f"Timeout after {timeout}s",
                "exit_code": -1,
            }
    except Exception as e:
        return {
            "status": "error",
            "stdout": "",
            "stderr": str(e),
            "exit_code": -1,
        }
