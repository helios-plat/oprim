"""oprim: 单次代码形式化验证 (≤1 位置参数).

铁律: ≤1 个位置参数，其余 keyword-only.

组合: SymbolicExecutor (Protocol 注入，不硬 import obase).

例:
    >>> result = _crosshair_verify(
    ...     "import deal\\n@deal.pre(lambda x: x >= 0)\\ndef f(x): return x\\n",
    ...     executor=my_executor,
    ... )
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Protocol


class SymbolicExecutorProtocol(Protocol):
    """符号执行器 Protocol — 不 import obase，由调用方注入."""

    def verify_file(self, filepath: str) -> dict[str, Any]: ...
    def verify_source(self, code: str, *, filename: str = "") -> dict[str, Any]: ...


def _crosshair_verify(
    code_content: str,
    *,
    executor: SymbolicExecutorProtocol,
) -> dict[str, Any]:
    """将代码写入临时文件并执行有界模型检查.

    签名遵循 oprim 铁律：最多 1 个位置参数，其余 kw-only.

    流程:
    1. 将 code_content 写入临时 .py 文件
    2. 调用 executor.verify_file() 符号执行
    3. 返回结果并清理临时文件

    Args:
        code_content: 含 deal 契约的 Python 源代码
        executor: 符号执行器（注入 SymbolicExecutorProtocol）

    Returns:
        {
            "ok": bool,
            "stdout": str,
            "stderr": str,
            "exit_code": int,
        }
    """
    fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="verify_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(code_content)

        return executor.verify_file(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
