"""oprim.tdd_test_run — 单次在沙箱环境中运行单元测试并解析输出.

铁律: ≤1 个位置参数，其余 keyword-only.

组合: subprocess (stdlib only, 无外部依赖).

例:
    >>> result = tdd_test_run("pytest tests/ -x", cwd=".", timeout_sec=15)
    >>> result["passed"]
    True
"""

from __future__ import annotations

import subprocess
from typing import Any


def tdd_test_run(
    test_command: str = "pytest",
    *,
    cwd: str = ".",
    timeout_sec: float = 30.0,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """单次在沙箱环境中运行单元测试并解析输出.

    签名遵循 oprim 铁律：最多 1 个位置参数，其余 kw-only.

    Args:
        test_command: 测试命令字符串（如 "pytest tests/ -x -v"）
        cwd: 工作目录
        timeout_sec: 超时秒数
        env: 环境变量覆盖

    Returns:
        {
            "passed": bool,
            "exit_code": int,
            "stdout": str (截断至 2000 字符),
            "stderr": str (截断至 2000 字符),
        }
    """
    try:
        proc = subprocess.run(
            test_command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=env,
        )
        passed = proc.returncode == 0
        return {
            "passed": passed,
            "exit_code": proc.returncode,
            "stdout": proc.stdout[-2000:],
            "stderr": proc.stderr[-2000:],
        }
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": f"Test command timed out after {timeout_sec}s",
        }
    except Exception as e:
        return {
            "passed": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": str(e)[:2000],
        }
