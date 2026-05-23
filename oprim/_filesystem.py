"""Filesystem oprim — 3 atomic filesystem operations."""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from oprim._exceptions import (
    OprimError,
    OprimNotFoundError,
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class DiskUsage(BaseModel):
    path: str
    total_bytes: int
    used_bytes: int
    free_bytes: int
    used_percent: float


class ArchiveResult(BaseModel):
    src_dir: str
    dst_path: str
    archive_bytes: int
    file_count: int
    elapsed_ms: int
    checksum_sha256: str


# ---------------------------------------------------------------------------
# 7.1 disk_usage
# ---------------------------------------------------------------------------

def disk_usage(
    *,
    path: str,
) -> DiskUsage:
    """查 path 所在文件系统的使用情况.

    Args:
        path: 文件系统路径

    Returns:
        DiskUsage 含 total / used / free bytes 和使用率

    Raises:
        OprimNotFoundError: path 不存在
    """
    p = Path(path)
    if not p.exists():
        raise OprimNotFoundError(f"Path not found: {path}")

    usage = shutil.disk_usage(path)
    used_percent = (usage.used / usage.total * 100.0) if usage.total > 0 else 0.0

    return DiskUsage(
        path=str(p.resolve()),
        total_bytes=usage.total,
        used_bytes=usage.used,
        free_bytes=usage.free,
        used_percent=round(used_percent, 2),
    )


# ---------------------------------------------------------------------------
# 7.2 dir_archive_to_targz
# ---------------------------------------------------------------------------

def _matches_any(name: str, patterns: list[str]) -> bool:
    import fnmatch
    return any(fnmatch.fnmatch(name, pat) for pat in patterns)


def dir_archive_to_targz(
    *,
    src_dir: str,
    dst_path: str,
    exclude_patterns: list[str] | None = None,
    follow_symlinks: bool = False,
) -> ArchiveResult:
    """把目录打包为 tar.gz.

    Compresses src_dir into dst_path and computes SHA-256 of the archive
    in a single streaming pass.

    Args:
        src_dir: 源目录
        dst_path: 目标 tar.gz 路径
        exclude_patterns: glob 排除模式列表 (e.g. ["*.tmp", ".git"])
        follow_symlinks: 是否跟随符号链接

    Returns:
        ArchiveResult 含文件数 / 大小 / checksum

    Raises:
        OprimNotFoundError: src_dir 不存在
        OprimError: 写入 dst_path 失败
    """
    src = Path(src_dir)
    if not src.exists() or not src.is_dir():
        raise OprimNotFoundError(f"Source directory not found: {src_dir}")

    excludes = exclude_patterns or []
    t0 = time.monotonic()
    file_count = 0

    try:
        with tarfile.open(dst_path, "w:gz") as tar:
            for file_path in sorted(src.rglob("*")):
                # Check exclude patterns against relative path parts
                rel = file_path.relative_to(src)
                parts = rel.parts
                if any(_matches_any(part, excludes) for part in parts):
                    continue
                if not follow_symlinks and file_path.is_symlink():
                    continue
                arcname = str(rel)
                tar.add(str(file_path), arcname=arcname, recursive=False)
                if file_path.is_file():
                    file_count += 1
    except (OSError, tarfile.TarError) as exc:
        raise OprimError(f"Failed to create archive at {dst_path}: {exc}") from exc

    elapsed = int((time.monotonic() - t0) * 1000)

    # Compute SHA-256 of the archive
    h = hashlib.sha256()
    try:
        with open(dst_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    except OSError as exc:
        raise OprimError(f"Failed to read archive for checksum: {exc}") from exc

    archive_bytes = Path(dst_path).stat().st_size

    return ArchiveResult(
        src_dir=src_dir,
        dst_path=dst_path,
        archive_bytes=archive_bytes,
        file_count=file_count,
        elapsed_ms=elapsed,
        checksum_sha256=h.hexdigest(),
    )


# ---------------------------------------------------------------------------
# 7.3 file_checksum
# ---------------------------------------------------------------------------

def file_checksum(
    *,
    file_path: str,
    algorithm: Literal["sha256", "md5", "sha1"] = "sha256",
    chunk_size: int = 65536,
) -> str:
    """计算文件 checksum.

    Args:
        file_path: 文件路径
        algorithm: 哈希算法 ("sha256", "md5", "sha1")
        chunk_size: 流式读取块大小 (bytes)

    Returns:
        十六进制 checksum 字符串

    Raises:
        OprimNotFoundError: 文件不存在
    """
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        raise OprimNotFoundError(f"File not found: {file_path}")

    h = hashlib.new(algorithm)
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()
