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
    OprimValidationError,
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
    threshold_percent: float | None = None
    over_threshold: bool | None = None


class ArchiveResult(BaseModel):
    sources: list[str]
    dst_path: str
    archive_bytes: int
    file_count: int
    elapsed_ms: int
    checksum_sha256: str

    @property
    def src_dir(self) -> str:
        import warnings

        msg = "ArchiveResult.src_dir is deprecated, use .sources"
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return self.sources[0] if self.sources else ""


# ---------------------------------------------------------------------------
# 7.1 disk_usage
# ---------------------------------------------------------------------------


def disk_usage(
    *,
    path: str,
    threshold_percent: float | None = None,
) -> DiskUsage:
    """查 path 所在文件系统的使用情况.

    Args:
        path: 文件系统路径
        threshold_percent: 可选使用率告警阈值(0-100). 给定时结果的 over_threshold 置为
            used_percent >= threshold_percent;不给则 over_threshold 为 None(未评估).

    Returns:
        DiskUsage 含 total / used / free bytes 和使用率;给定阈值时含 over_threshold.

    Raises:
        OprimNotFoundError: path 不存在
    """
    p = Path(path)
    if not p.exists():
        raise OprimNotFoundError(f"Path not found: {path}")

    usage = shutil.disk_usage(path)
    used_percent = (usage.used / usage.total * 100.0) if usage.total > 0 else 0.0
    over_threshold = used_percent >= threshold_percent if threshold_percent is not None else None

    return DiskUsage(
        path=str(p.resolve()),
        total_bytes=usage.total,
        used_bytes=usage.used,
        free_bytes=usage.free,
        used_percent=round(used_percent, 2),
        threshold_percent=threshold_percent,
        over_threshold=over_threshold,
    )


# ---------------------------------------------------------------------------
# 7.2 archive_to_targz
# ---------------------------------------------------------------------------


def _matches_any(name: str, patterns: list[str]) -> bool:
    import fnmatch

    return any(fnmatch.fnmatch(name, pat) for pat in patterns)


def archive_to_targz(
    *,
    sources: list[str],
    dst_path: str,
    exclude_patterns: list[str] | None = None,
    follow_symlinks: bool = False,
) -> ArchiveResult:
    """把多个目录或文件打包为 tar.gz.

    Args:
        sources: 源路径列表
        dst_path: 目标 tar.gz 路径
        exclude_patterns: glob 排除模式列表
        follow_symlinks: 是否跟随符号链接

    Returns:
        ArchiveResult 含文件数 / 大小 / checksum

    Raises:
        OprimNotFoundError: 某个源不存在
        OprimError: 写入失败
    """
    if not sources:
        raise OprimError("No sources provided for archiving")

    excludes = exclude_patterns or []
    t0 = time.monotonic()
    file_count = 0

    try:
        with tarfile.open(dst_path, "w:gz") as tar:
            for s_path in sources:
                src = Path(s_path)
                if not src.exists():
                    raise OprimNotFoundError(f"Source not found: {s_path}")

                if src.is_dir():
                    # Walk directory
                    for file_path in sorted(src.rglob("*")):
                        # Use relative path from src's parent to keep the src dir name in archive
                        # or relative to src to put contents in root.
                        # dir_archive_to_targz used relative_to(src), so let's stick to that for dir contents.
                        rel = file_path.relative_to(src)
                        if any(_matches_any(part, excludes) for part in rel.parts):
                            continue
                        if not follow_symlinks and file_path.is_symlink():
                            continue

                        # If we want multiple sources to coexist, we should probably keep their names
                        # But dir_archive_to_targz logic was rel = file_path.relative_to(src)
                        # and arcname = str(rel). This means if sources=[dir1, dir2], their contents
                        # will be mixed in the root of the archive.
                        arcname = str(rel)
                        if not arcname or arcname == ".":
                            continue
                        tar.add(str(file_path), arcname=arcname, recursive=False)
                        if file_path.is_file():
                            file_count += 1
                else:
                    if any(_matches_any(part, excludes) for part in src.parts):
                        continue
                    tar.add(str(src), arcname=src.name, recursive=False)
                    file_count += 1
    except (OSError, tarfile.TarError) as exc:
        if isinstance(exc, (OprimNotFoundError, OprimError)):
            raise
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
        sources=sources,
        dst_path=dst_path,
        archive_bytes=archive_bytes,
        file_count=file_count,
        elapsed_ms=elapsed,
        checksum_sha256=h.hexdigest(),
    )


def dir_archive_to_targz(
    *,
    src_dir: str,
    dst_path: str,
    exclude_patterns: list[str] | None = None,
    follow_symlinks: bool = False,
) -> ArchiveResult:
    """(Deprecated) use archive_to_targz."""
    import warnings

    msg = "dir_archive_to_targz is deprecated, use archive_to_targz"
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    return archive_to_targz(
        sources=[src_dir],
        dst_path=dst_path,
        exclude_patterns=exclude_patterns,
        follow_symlinks=follow_symlinks,
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


# ---------------------------------------------------------------------------
# Aegis IMPL SPEC v1.0 — short-name alias + fs_inode_check (B2)
# ---------------------------------------------------------------------------

fs_disk_usage = disk_usage


def fs_inode_check(
    *,
    path: str,
) -> dict[str, int | float | str]:
    """检查文件系统 inode 使用情况.

    Args:
        path: 目标路径 (任意挂载点内的路径即可)

    Returns:
        {
          "path": str,
          "inodes_total": int,
          "inodes_used": int,
          "inodes_free": int,
          "inodes_used_percent": float,
        }

    Raises:
        OprimNotFoundError: path 不存在
        OprimError: 平台不支持 inode 统计
    """
    p = Path(path)
    if not p.exists():
        raise OprimNotFoundError(f"Path not found: {path}")

    try:
        st = shutil.disk_usage(str(p))  # total/used/free bytes
    except Exception as exc:
        raise OprimError(f"Failed to stat path: {exc}") from exc

    try:
        import os

        stat_vfs = os.statvfs(str(p))
        total = stat_vfs.f_files
        free = stat_vfs.f_ffree
        used = total - free
        pct = round(used / total * 100, 2) if total > 0 else 0.0
    except AttributeError:
        raise OprimError("fs_inode_check is not supported on this platform (no os.statvfs)")

    return {
        "path": str(p),
        "inodes_total": total,
        "inodes_used": used,
        "inodes_free": free,
        "inodes_used_percent": pct,
    }


# ---------------------------------------------------------------------------
# disk_cleanup — allowlist 硬约束的磁盘清理 (aegis DESIGN §5.2 R2 / §9 S2 护栏)
# ---------------------------------------------------------------------------


class CleanupResult(BaseModel):
    freed_bytes: int
    touched_paths: list[str]  # 实删(或 dry_run 命中)的已解析路径
    dry_run: bool


def _path_size(p: Path) -> int:
    """文件返回自身大小;目录返回其下所有普通文件大小之和。"""
    if p.is_file() or p.is_symlink():
        try:
            return p.stat(follow_symlinks=False).st_size
        except OSError:
            return 0
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                continue
    return total


def disk_cleanup(
    *,
    targets: list[str],
    allowlist: list[str],
    dry_run: bool = True,
) -> CleanupResult:
    """删除 targets 中的文件/目录,但每个 target 必须落在 allowlist 前缀内。

    **护栏硬约束**:每个 target 解析(resolve,消解符号链接与 ..)后必须在某个 allowlist
    条目内(含相等),否则抛 OprimValidationError(**拒绝而非跳过**)——验证"不越界",
    供 aegis-verify S2 断言 cleanup 只触碰 allowlist 路径。dry_run=True 只统计不删。

    Args:
        targets: 待清理路径列表。
        allowlist: 允许清理的路径前缀列表(为空 → 任何 target 都越界报错)。
        dry_run: True 只统计不实删。

    Returns:
        CleanupResult(freed_bytes / touched_paths / dry_run)。

    Raises:
        OprimValidationError: 某 target 越出 allowlist。
    """
    allow_resolved = [Path(a).resolve() for a in allowlist]

    # 先全量校验:任一 target 越界即整批拒绝,不删任何东西(护栏优先于功能)。
    resolved: list[Path] = []
    for t in targets:
        p = Path(t).resolve()
        if not any(p == a or p.is_relative_to(a) for a in allow_resolved):
            raise OprimValidationError(f"target {t!r} (resolved {p}) escapes allowlist {allowlist}")
        resolved.append(p)

    freed = 0
    touched: list[str] = []
    for p in resolved:
        if not p.exists():
            continue  # 已不存在,幂等跳过
        freed += _path_size(p)
        touched.append(str(p))
        if not dry_run:
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p)
            else:
                p.unlink()

    return CleanupResult(freed_bytes=freed, touched_paths=touched, dry_run=dry_run)
