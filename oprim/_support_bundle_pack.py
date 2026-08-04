"""oprim.support_bundle_pack — 支持包打包.

把指定文件/目录打包为 zip 支持包（大小上限 + 条目统计）。

Example:
    >>> r = await support_bundle_pack("bundle-1", paths=["/var/log", "cfg.json"])
    >>> r["entries"] >= 1
    True
"""

from __future__ import annotations

import time
import zipfile
from pathlib import Path
from typing import Any

from oprim._exceptions import OprimError, OprimValidationError


class SupportBundleError(OprimError):
    """支持包打包失败。"""


async def support_bundle_pack(
    bundle_name: str,
    *,
    paths: list[str],
    output_dir: str | None = None,
    max_bytes: int = 100 * 1024 * 1024,
    include_hidden: bool = False,
) -> dict[str, Any]:
    """打包支持包（zip，流式写入内存受 max_bytes 约束）。

    Args:
        bundle_name: 包名（不含扩展名）。
        paths: 待打包的文件/目录列表。
        output_dir: 输出目录；None 用当前目录。
        max_bytes: 总大小上限（超出抛 SupportBundleError）。
        include_hidden: 目录递归时是否包含隐藏文件。

    Returns:
        {"status": "ok", "archive": str, "entries": int, "bytes": int}

    Raises:
        SupportBundleError: 无有效路径 / 超限 / 打包失败。
        OprimValidationError: bundle_name 为空。
    """
    if not bundle_name or not bundle_name.strip():
        raise OprimValidationError("support_bundle_pack: bundle_name must not be empty")
    if not paths:
        raise SupportBundleError("support_bundle_pack: paths must not be empty")

    out_dir = Path(output_dir) if output_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / f"{bundle_name}-{int(time.time())}.zip"

    total = 0
    entries = 0
    try:
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for raw in paths:
                src = Path(raw).expanduser()
                if not src.exists():
                    raise SupportBundleError(f"support_bundle_pack: path not found: {src}")
                if src.is_dir():
                    for p in sorted(src.rglob("*")):
                        if p.is_dir():
                            continue
                        if not include_hidden and any(
                            part.startswith(".") for part in p.relative_to(src).parts
                        ):
                            continue
                        size = p.stat().st_size
                        if total + size > max_bytes:
                            raise SupportBundleError(
                                f"support_bundle_pack exceeds max_bytes={max_bytes}"
                            )
                        zf.write(p, arcname=f"{src.name}/{p.relative_to(src)}")
                        total += size
                        entries += 1
                else:
                    size = src.stat().st_size
                    if total + size > max_bytes:
                        raise SupportBundleError(
                            f"support_bundle_pack exceeds max_bytes={max_bytes}"
                        )
                    zf.write(src, arcname=src.name)
                    total += size
                    entries += 1
    except SupportBundleError:
        import contextlib

        with contextlib.suppress(OSError):
            archive_path.unlink()
        raise
    except Exception as exc:
        raise SupportBundleError(
            f"support_bundle_pack failed: {type(exc).__name__}: {exc}", cause=exc
        ) from exc

    return {
        "status": "ok",
        "archive": str(archive_path),
        "entries": entries,
        "bytes": total,
    }
