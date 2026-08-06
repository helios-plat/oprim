"""oprim._snapshot — 快照存储: 状态树分支的底座。

生产上该用什么: overlayfs 每节点一个 upper-dir / btrfs subvolume / ZFS clone /
Firecracker microVM 快照。毫秒级分支, 存储按写入量增长。
**不要用 `docker commit`** —— 秒级 + 层膨胀, 树搜索里每个节点提交一次就把磁盘吃穿。

这里实现可移植的兜底版本 + CowBackend 协议, 方便替换成上面任何一种。
hardlink 模式要求写入方永远"写临时文件 + rename"(atomic_write) ——
绝不能用 open(path,"w") 原地截断, 那会顺着硬链接改坏 store 里的原件。
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Protocol


def atomic_write(path: str, data: str) -> None:
    """写临时文件再 rename。硬链接 CoW 模式下的强制写法。"""
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, path)          # 原子替换, 不碰原 inode
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def tree_digest(root: str, skip: tuple = ()) -> str:
    """目录树的内容指纹。相同内容 → 相同 digest, 与时间戳/inode 无关。

    skip: 绝对路径元组 —— 当被快照的目录**包含快照库自身**时排除, 防自递归。
    """
    root_abs = os.path.abspath(root)
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            p = os.path.join(dirpath, name)
            rel = os.path.relpath(p, root)
            if rel.startswith(".tmp-") or "__pycache__" in rel:
                continue
            h.update(rel.encode())
            h.update(b"\x00")
            h.update(b"x" if os.access(p, os.X_OK) else b"-")
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            h.update(b"\xff")
        # 防自递归: 快照库/池目录在被快照目录内部时, 整棵子树剔除
        for d in list(dirnames):
            if os.path.abspath(os.path.join(dirpath, d)) in skip:
                dirnames.remove(d)
    return h.hexdigest()[:24]


class CowBackend(Protocol):
    """替换点: 实现成 overlayfs / btrfs / zfs / firecracker 即可。"""
    def materialize(self, src: str, dst: str) -> None: ...


class CopyBackend:
    def materialize(self, src: str, dst: str, skip: tuple = ()) -> None:
        src_abs = os.path.abspath(src)
        for dirpath, dirnames, filenames in os.walk(src):
            rel = os.path.relpath(dirpath, src)
            target = os.path.join(dst, rel) if rel != "." else dst
            os.makedirs(target, exist_ok=True)
            for name in filenames:
                shutil.copy2(os.path.join(dirpath, name), os.path.join(target, name))
            for d in list(dirnames):
                if os.path.abspath(os.path.join(dirpath, d)) in skip:
                    dirnames.remove(d)


class HardlinkBackend:
    """要求写入方使用 atomic_write。见模块 docstring 的警告。"""
    def materialize(self, src: str, dst: str, skip: tuple = ()) -> None:
        for dirpath, dirnames, filenames in os.walk(src):
            rel = os.path.relpath(dirpath, src)
            target = os.path.join(dst, rel) if rel != "." else dst
            os.makedirs(target, exist_ok=True)
            for name in filenames:
                s, d = os.path.join(dirpath, name), os.path.join(target, name)
                if os.path.exists(d):
                    os.remove(d)
                os.link(s, d)
            for d in list(dirnames):
                if os.path.abspath(os.path.join(dirpath, d)) in skip:
                    dirnames.remove(d)


@dataclass
class SnapshotStore:
    """内容寻址的目录树快照库。"""
    root: str
    backend: CowBackend = None                   # type: ignore[assignment]

    def __post_init__(self):
        if self.backend is None:
            self.backend = CopyBackend()
        self.root = os.path.abspath(self.root)
        os.makedirs(self.root, exist_ok=True)

    def _skip_dirs(self, src: str) -> tuple:
        """当快照库自身位于被快照目录内部时, 返回需要排除的绝对路径。"""
        src_abs = os.path.abspath(src)
        if self.root != src_abs and os.path.commonpath([src_abs, self.root]) == src_abs:
            return (self.root,)
        return ()

    def _path(self, digest: str) -> str:
        return os.path.join(self.root, digest)

    def commit(self, src: str) -> str:
        """把一个目录存进快照库, 返回内容 digest。同内容重复提交是空操作。"""
        skip = self._skip_dirs(src)
        digest = tree_digest(src, skip=skip)
        dst = self._path(digest)
        if not os.path.exists(dst):
            tmp = dst + ".partial"
            shutil.rmtree(tmp, ignore_errors=True)
            os.makedirs(tmp, exist_ok=True)
            self.backend.materialize(src, tmp, skip=skip)
            os.replace(tmp, dst)                 # 原子发布, 避免半截快照被读到
        return digest

    def checkout(self, digest: str, dst: str) -> str:
        """把快照展开到 dst(会先清空 dst)。"""
        src = self._path(digest)
        if not os.path.isdir(src):
            raise KeyError(f"快照 {digest} 不存在")
        shutil.rmtree(dst, ignore_errors=True)
        os.makedirs(dst, exist_ok=True)
        self.backend.materialize(src, dst)
        return dst

    def exists(self, digest: str) -> bool:
        return os.path.isdir(self._path(digest))

    def list(self) -> List[str]:
        return sorted(d for d in os.listdir(self.root) if not d.endswith(".partial"))

    def gc(self, keep: Optional[set] = None) -> int:
        """推演结束后销毁临时快照。keep 里的保留(通常是 base 和 chosen)。"""
        keep = keep or set()
        n = 0
        for d in self.list():
            if d not in keep:
                shutil.rmtree(self._path(d), ignore_errors=True)
                n += 1
        return n


__all__ = ["CopyBackend", "CowBackend", "HardlinkBackend", "SnapshotStore",
           "atomic_write", "tree_digest"]
