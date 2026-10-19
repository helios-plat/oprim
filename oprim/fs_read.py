"""文件读取原子操作 (3O 范式 Phase 2 — oprim 元实现层).

单次原子 IO: 从沙盒 VFS 中读取文件。
仅认知 path 与 VFS 抽象, 不感知任何业务概念。
"""

from __future__ import annotations

from obase import VFSProvider


async def read_isolated_file(
    path: str,  # ≤1个核心位置参数
    *,  # 其余强制 keyword-only
    vfs: VFSProvider,  # 依赖注入
) -> str:
    """单次原子 IO: 从沙盒 VFS 中读取文件。

    Args:
        path: 沙盒内文件路径。
        vfs: 虚拟文件系统契约实现 (依赖注入)。

    Returns:
        文件内容字符串 (utf-8, 非法字节以替换符处理)。
    """
    data_bytes = await vfs.read_file(path)
    return data_bytes.decode("utf-8", errors="replace")
