"""oprim.soul_config_rewrite — SOUL 配置原子重写.

带版本快照的 SOUL/配置文件原子重写：临时文件 + os.replace 落盘，
可选 version store（Protocol 注入）记录改前/改后版本。

Example:
    >>> r = await soul_config_rewrite("/cfg/soul.md", content="新人格", sandbox_root="/cfg")
    >>> r["written"]
    True
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from oprim._exceptions import FileOprimError, OprimValidationError, PathSecurityError


@runtime_checkable
class VersionStoreHandle(Protocol):
    """版本库协议（对齐 obase.versionstore.VersionStore 的 snapshot 面）。"""

    def snapshot(self, paths: list[str], *, message: str = "") -> str: ...


async def soul_config_rewrite(
    path: str | Path,
    *,
    content: str,
    sandbox_root: str | Path | None = None,
    version_store: VersionStoreHandle | None = None,
    encoding: str = "utf-8",
) -> dict[str, Any]:
    """原子重写 SOUL 配置。

    Args:
        path: 目标配置文件路径。
        content: 新内容。
        sandbox_root: 沙箱根（路径越界拒绝）。
        version_store: 版本库（可选，写前快照 + 写后版本）。
        encoding: 编码。

    Returns:
        {"status": "ok", "path": str, "bytes_written": int,
         "rev_before": str|None, "rev_after": str|None}

    Raises:
        FileOprimError: 写入失败。
        PathSecurityError: 路径越出 sandbox_root。
        OprimValidationError: path / content 缺失。
    """
    target = Path(path).expanduser()
    if not str(target):
        raise OprimValidationError("soul_config_rewrite: path must not be empty")
    if content is None:
        raise OprimValidationError("soul_config_rewrite: content must not be empty")

    if sandbox_root is not None:
        root = Path(sandbox_root).resolve()
        try:
            resolved = target.resolve()
        except OSError as exc:
            raise FileOprimError(f"soul_config_rewrite: cannot resolve {target}", cause=exc) from exc
        if not str(resolved).startswith(str(root)):
            raise PathSecurityError(
                f"soul_config_rewrite: {target} escapes sandbox_root {root}"
            )

    rev_before: str | None = None
    rev_after: str | None = None
    if version_store is not None and target.exists():
        try:
            rev_before = version_store.snapshot([str(target)], message="soul:before")
        except Exception:  # noqa: BLE001 - 快照失败不阻断写入
            rev_before = None

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
    except OSError as exc:
        raise FileOprimError(
            f"soul_config_rewrite failed: {type(exc).__name__}: {exc}", cause=exc
        ) from exc

    if version_store is not None:
        try:
            rev_after = version_store.snapshot([str(target)], message="soul:after")
        except Exception:  # noqa: BLE001
            rev_after = None

    return {
        "status": "ok",
        "path": str(target),
        "bytes_written": len(data),
        "rev_before": rev_before,
        "rev_after": rev_after,
    }
