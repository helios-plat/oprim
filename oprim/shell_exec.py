"""终端命令执行原子操作 (3O 范式 Phase 2 — oprim 元实现层).

单次原子 IO: 在隔离环境中执行命令。
纯物理动作, 不判断 exit_code 所代表的业务含义

绝不处理"报错就自动重试"之类的业务逻辑 (由上层 omodul 编排引擎负责)。
"""

from __future__ import annotations

from typing import Any

from obase import VFSProvider


async def execute_sandbox_cmd(
    command: str,  # ≤1个核心位置参数
    *,  # 其余强制 keyword-only
    vfs: VFSProvider,
    timeout: int = 30,
) -> dict[str, Any]:
    """单次原子 IO: 在隔离环境中执行命令。

    Args:
        command: 要执行的终端命令。
        vfs: 虚拟文件系统契约实现 (依赖注入)。
        timeout: 执行超时秒数, 默认 30。

    Returns:
        执行结果字典 (exit_code / stdout / stderr)。

    Raises:
        RuntimeError: 底层沙盒物理崩溃 (如宿主进程被杀)。
            常规命令失败 (exit_code != 0) 不在此拦截, 由上层评估。
    """
    result = await vfs.execute_isolated(command, timeout=timeout)

    # 仅拦截底层的、非业务的物理沙盒崩溃 (如宿主进程被杀)
    # 不拦截常规的命令执行失败 (exit_code != 0)，那是由上层评估的
    if result["exit_code"] == -1:
        raise RuntimeError(f"VFS Sandbox failed to execute: {result['stderr']}")

    return result
