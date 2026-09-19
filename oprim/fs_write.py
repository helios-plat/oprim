"""文件写入原子操作 (3O 范式 Phase 2 — oprim 元实现层).

单次原子 IO: 向沙盒 VFS 中写入文件。
content 是"写"这个动作最核心的语义, 故作为唯一位置参数
path 降级为 keyword-only。
"""

from __future__ import annotations

from obase import VFSProvider


async def write_isolated_file(
    content: str,  # ≤1个核心位置参数 (要写什么)
    *,  # 其余强制 keyword-only
    path: str,  # 写到哪里
    vfs: VFSProvider,  # 依赖注入
) -> None:
    """单次原子 IO: 向沙盒 VFS 中写入文件。

    Args:
        content: 要写入的内容。
        path: 沙盒内目标文件路径。
        vfs: 虚拟文件系统契约实现 (依赖注入)。
    """
    await vfs.write_file(path, content.encode("utf-8"))
