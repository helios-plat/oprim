"""oprim.file_write_atomic — 带安全校验的单次文件原子落盘操作.

先写临时文件再 os.replace 原子替换；sandbox_root 给定时先做路径安全校验
（复用 oprim.fs.path_resolve 沙箱语义，防路径穿越）。

Example:
    >>> r = await file_write_atomic("/tmp/out.txt", "hello")
    >>> r["bytes_written"]
    5
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from oprim._exceptions import FileOprimError, PathSecurityError


async def file_write_atomic(
    path: str | Path,
    *,
    content: str,
    sandbox_root: str | Path | None = None,
    encoding: str = "utf-8",
    overwrite: bool = True,
) -> dict[str, Any]:
    """原子写入文件。

    Args:
        path: 目标文件路径。
        content: 文本内容。
        sandbox_root: 沙箱根目录；给定时目标必须位于其内。
        encoding: 文本编码。
        overwrite: 目标已存在时是否覆盖；False 且存在则抛 FileOprimError。

    Returns:
        {"status": "ok", "path": str, "bytes_written": int}

    Raises:
        FileOprimError: 写入失败 / overwrite=False 且目标存在。
        PathSecurityError: 路径越出 sandbox_root。
    """
    target = Path(path).expanduser()

    if sandbox_root is not None:
        root = Path(sandbox_root).resolve()
        try:
            resolved = target.resolve()
        except OSError as exc:
            raise FileOprimError(f"file_write_atomic: cannot resolve {target}", cause=exc) from exc
        if not str(resolved).startswith(str(root)):
            raise PathSecurityError(
                f"file_write_atomic: {target} escapes sandbox_root {root}"
            )

    if target.exists() and not overwrite:
        raise FileOprimError(f"file_write_atomic: target exists and overwrite=False: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    data = content.encode(encoding)

    try:
        fd, tmp_name = tempfile.mkstemp(
            dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, target)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:  # pragma: no cover
                pass
            raise
    except FileOprimError:
        raise
    except OSError as exc:
        raise FileOprimError(
            f"file_write_atomic failed: {type(exc).__name__}: {exc}", cause=exc
        ) from exc

    return {"status": "ok", "path": str(target), "bytes_written": len(data)}
