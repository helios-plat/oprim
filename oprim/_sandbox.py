"""oprim._sandbox — 沙箱与预热池。

隔离等级(按强度递增):
  none      : 裸 subprocess。**不是沙箱**, 只适合完全可信的代码。
  netns     : `unshare -Urn` —— user + network namespace。本模块默认。
              切断网络是 rollout 可复现的硬前提: 一旦能联网, 同一候选跑两次
              结果就可能不同, MCTS 的奖励分布不再平稳。
  container : Docker / Podman + gVisor。跨内核边界。见 DockerSandbox。
  microvm   : Firecracker。真正的强隔离 + 毫秒级快照恢复, 树搜索的终局形态。

**WSL 不在这个列表里。** 它共享内核、默认能读宿主 /mnt/c、网络直通,
不构成隔离边界, 别把它和 Docker 并列。

预热池: 容器/命名空间的启动开销在树搜索里会被放大 N 倍。
SandboxPool 预先创建并复用实例, acquire 只做 workspace 重置。
"""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

_HAS_UNSHARE = shutil.which("unshare") is not None


@dataclass
class ExecResult:
    argv: List[str]
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


_UNSHARE_OK: Optional[bool] = None


def unshare_available() -> bool:
    """功能探测: unshare -Urn 真能跑起来吗?

    有些容器/CI 环境 (seccomp / 禁 user namespace) 里 unshare 存在但调用即 EPERM ——
    只看 which 不够。探测结果模块级缓存, 全进程共享。
    """
    global _UNSHARE_OK
    if _UNSHARE_OK is None:
        if not _HAS_UNSHARE:
            _UNSHARE_OK = False
        else:
            try:
                p = subprocess.run(["unshare", "-Urn", "true"], capture_output=True, timeout=10)
                _UNSHARE_OK = p.returncode == 0
            except (OSError, subprocess.TimeoutExpired):
                _UNSHARE_OK = False
    return _UNSHARE_OK


class LocalSandbox:
    """workspace 目录 + 受限子进程。

    isolation="netns" 时用 `unshare -Urn` 起进程(新 user + network namespace)。
    实测启动开销约 2ms, 比容器低两个数量级, 适合做高频 rollout 的第一道隔离。
    """

    def __init__(self, base_dir: str, isolation: str = "netns",
                 env: Optional[dict] = None):
        self.workspace = tempfile.mkdtemp(prefix="ws-", dir=base_dir)
        # 功能探测: unshare 存在但被 seccomp 禁掉时回落 none, 不假装隔离
        self.isolation = isolation if (isolation == "netns" and unshare_available()) else "none"
        # 环境变量全部清空再按需注入 —— 防止宿主 API key / 代理配置泄进沙箱
        self.env = env or {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": self.workspace,
            "LC_ALL": "C.UTF-8",
            "PYTHONHASHSEED": "0",        # 确定性: 关掉 dict/set 的随机哈希
            "PYTHONDONTWRITEBYTECODE": "1",
            "TZ": "UTC",                  # 冻结时区, 避免跨机器行为漂移
            "SOURCE_DATE_EPOCH": "0",
        }

    def _wrap(self, argv: Sequence[str]) -> List[str]:
        if self.isolation == "netns":
            return ["unshare", "-Urn", "--"] + list(argv)
        return list(argv)

    def run(self, argv: Sequence[str], timeout_s: float = 30.0) -> ExecResult:
        t0 = time.time()
        full = self._wrap(argv)
        try:
            p = subprocess.run(full, cwd=self.workspace, env=self.env,
                               capture_output=True, text=True, timeout=timeout_s)
            return ExecResult(list(argv), p.returncode, p.stdout, p.stderr,
                              int((time.time() - t0) * 1000))
        except subprocess.TimeoutExpired as e:
            return ExecResult(list(argv), -9, e.stdout or "", e.stderr or "",
                              int((time.time() - t0) * 1000), timed_out=True)
        except FileNotFoundError as e:
            return ExecResult(list(argv), -2, "", str(e),
                              int((time.time() - t0) * 1000))

    def reset(self) -> None:
        shutil.rmtree(self.workspace, ignore_errors=True)
        os.makedirs(self.workspace, exist_ok=True)

    def destroy(self) -> None:
        shutil.rmtree(self.workspace, ignore_errors=True)


class DockerSandbox(LocalSandbox):
    """容器骨架 —— CI 环境可能没有 docker, 按需自行接。

    要点: --network=none --read-only --tmpfs /tmp --pids-limit --memory --cpus;
    --user $(id -u):$(id -g) 别用 root; 镜像 tag 钉死 digest(sha256:...);
    树搜索分支用 overlayfs upper-dir, 不要 docker commit。
    """

    def __init__(self, base_dir: str, image: str, **kw):
        super().__init__(base_dir, isolation="none", **kw)
        self.image = image

    def _wrap(self, argv):
        return ["docker", "run", "--rm", "--network=none", "--read-only",
                "--tmpfs", "/tmp", "--pids-limit", "256", "--memory", "512m",
                "--cpus", "1", "-v", f"{self.workspace}:/w", "-w", "/w",
                self.image] + list(argv)


@dataclass
class PoolStats:
    acquires: int = 0
    total_wait_ms: int = 0
    max_wait_ms: int = 0
    created: int = 0

    @property
    def avg_wait_ms(self) -> float:
        return self.total_wait_ms / self.acquires if self.acquires else 0.0


class SandboxPool:
    """预热池。size 决定并发上限, 也决定你愿意为搜索付出的峰值资源。"""

    def __init__(self, size: int = 4, base_dir: Optional[str] = None,
                 isolation: str = "netns", factory=None):
        self._owns_base = base_dir is None      # 调用方给的目录不归池子处置
        self.base_dir = base_dir or tempfile.mkdtemp(prefix="o3-pool-")
        os.makedirs(self.base_dir, exist_ok=True)
        self.size = size
        self.isolation = isolation
        self._factory = factory or (lambda: LocalSandbox(self.base_dir, isolation))
        self._q: "queue.Queue" = queue.Queue()
        self._all: List[LocalSandbox] = []
        self.stats = PoolStats()

    def prewarm(self) -> "SandboxPool":
        for _ in range(self.size):
            sb = self._factory()
            self._all.append(sb)
            self._q.put(sb)
            self.stats.created += 1
        return self

    def acquire(self, timeout: float = 60.0) -> LocalSandbox:
        t0 = time.time()
        try:
            sb = self._q.get(timeout=timeout)
        except queue.Empty:                      # 没预热就临时造一个(并记账)
            sb = self._factory()
            self._all.append(sb)
            self.stats.created += 1
        wait = int((time.time() - t0) * 1000)
        self.stats.acquires += 1
        self.stats.total_wait_ms += wait
        self.stats.max_wait_ms = max(self.stats.max_wait_ms, wait)
        return sb

    def release(self, sb: LocalSandbox) -> None:
        sb.reset()
        self._q.put(sb)

    def shutdown(self) -> None:
        for sb in self._all:
            sb.destroy()
        self._all.clear()
        if self._owns_base:                      # 只删自己创建的临时根目录
            shutil.rmtree(self.base_dir, ignore_errors=True)

    def __enter__(self):
        return self.prewarm()

    def __exit__(self, *exc):
        self.shutdown()
        return False


__all__ = ["DockerSandbox", "ExecResult", "LocalSandbox", "PoolStats",
           "SandboxPool", "unshare_available"]
