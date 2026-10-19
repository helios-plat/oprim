"""oprim: 共享内存原子的读写操作 (≤1 位置参数).

铁律: ≤1 个位置参数，其余 keyword-only。

基于 memoryview 切片实现零拷贝操作。
"""

from __future__ import annotations

import contextlib
from multiprocessing import shared_memory
from typing import Any


def _mmap_zero_copy_rw(
    shm_name: str,
    *,
    action: str,
    data: bytes | None = None,
    offset: int = 0,
    size: int = 0,
) -> dict[str, Any]:
    """单次共享内存 R/W 操作，不抛出异常.

    签名遵循 oprim 铁律：最多 1 个位置参数，其余 kw-only.

    通过 memoryview 原地写入，避免 Python 层拷贝。

    Args:
        shm_name: 共享内存名称
        action: "read" | "write"
        data: 写入数据（action="write" 时必需）
        offset: 偏移量（字节）
        size: 读取大小（action="read" 时，0=全部）

    Returns:
        {
            "status": "success" | "error",
            "bytes_written": int (write),
            "data": bytes (read),
            "bytes_read": int (read),
        }
    """
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
    except FileNotFoundError:
        return {"status": "error", "error": f"内存块 {shm_name} 不存在"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

    try:
        if action == "write":
            if data is None:
                shm.close()
                return {"status": "error", "error": "写入模式下缺失 data"}
            data_len = len(data)
            if offset + data_len > shm.size:
                shm.close()
                return {
                    "status": "error",
                    "error": f"写入越界: offset={offset} + len={data_len} > size={shm.size}",
                }
            # memoryview 原地写入，零拷贝
            shm.buf[offset : offset + data_len] = data
            shm.close()
            return {"status": "success", "bytes_written": data_len}

        elif action == "read":
            read_len = size if size > 0 else (shm.size - offset)
            if offset + read_len > shm.size:
                read_len = shm.size - offset
            if read_len <= 0:
                shm.close()
                return {
                    "status": "error",
                    "error": f"读取范围无效: offset={offset}, size={shm.size}",
                }
            result = bytes(shm.buf[offset : offset + read_len])
            shm.close()
            return {"status": "success", "data": result, "bytes_read": len(result)}

        else:
            shm.close()
            return {"status": "error", "error": f"未知动作: {action}"}

    except Exception as e:
        with contextlib.suppress(Exception):
            shm.close()
        return {"status": "error", "error": str(e)}
