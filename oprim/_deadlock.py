"""oprim._deadlock — 死锁防线: 这是资源序问题, 不是博弈论问题。

拍卖解决不了循环等待。A 持 db 等 cache、B 持 cache 等 db, 报价再合理也解不开 ——
问题出在**获取顺序**上, 不在分配价格上。三层防线按成本递增:
  ① 预防 (ResourceOrder): 全局资源全序, 环意味着存在 r_i < r_j 且 r_j < r_i,
     与全序矛盾 —— 数学上直接消灭循环等待。
  ② 检测 (WaitForGraph): 真出现环时找出来并挑受害者回滚。
  ③ 兜底 (LeaseManager): 所有持有都是有期限的租约, 到期自动释放。
     这层是必须的 —— 你不可能保证前两层没有洞。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


class ResourceOrder:
    """资源全序。按 (rank, id) 排序, rank 相同按 id 字典序 —— 保证确定性。"""

    def __init__(self, ranked: Sequence[str]):
        self.rank = {r: i for i, r in enumerate(ranked)}

    def key(self, resource: str) -> Tuple[int, str]:
        return (self.rank.get(resource, len(self.rank)), resource)

    def sort(self, resources: Iterable[str]) -> List[str]:
        return sorted(set(resources), key=self.key)

    def violates(self, sequence: Sequence[str]) -> Optional[Tuple[str, str]]:
        """检查申请序列是否违反全序。返回第一处逆序对, 没有则 None。"""
        for a, b in zip(sequence, sequence[1:]):
            if self.key(b) < self.key(a):
                return (a, b)
        return None

    def plan(self, resources: Iterable[str]) -> List[str]:
        """把一组资源需求变成安全的申请顺序。按此顺序申请, 永不死锁。"""
        return self.sort(resources)


class WaitForGraph:
    """等待图。边 waiter → holder 表示"waiter 在等 holder 手上的资源"。"""

    def __init__(self):
        self.edges: Dict[str, Set[str]] = {}
        self.labels: Dict[Tuple[str, str], str] = {}

    def add_wait(self, waiter: str, holder: str, resource: str = "") -> None:
        if waiter == holder:
            return
        self.edges.setdefault(waiter, set()).add(holder)
        self.labels[(waiter, holder)] = resource

    def remove(self, waiter: str, holder: str) -> None:
        self.edges.get(waiter, set()).discard(holder)

    def cycles(self) -> List[List[str]]:
        """找出所有环(DFS + 递归栈)。节点按字典序遍历, 结果可复现。"""
        found: List[List[str]] = []
        seen: Set[str] = set()
        stack: List[str] = []
        on_stack: Set[str] = set()

        def dfs(u: str) -> None:
            seen.add(u)
            stack.append(u)
            on_stack.add(u)
            for v in sorted(self.edges.get(u, ())):
                if v in on_stack:
                    cyc = _canonical_cycle(stack[stack.index(v):])
                    if cyc not in found:
                        found.append(cyc)
                elif v not in seen:
                    dfs(v)
            stack.pop()
            on_stack.discard(u)

        for node in sorted(set(self.edges) | {v for s in self.edges.values() for v in s}):
            if node not in seen:
                dfs(node)
        return found

    def would_deadlock(self, waiter: str, holder: str) -> bool:
        """预检: 加这条边会不会成环。不真的加。"""
        if waiter == holder:
            return False
        target, seen, stack = waiter, set(), [holder]
        while stack:
            u = stack.pop()
            if u == target:
                return True
            if u in seen:
                continue
            seen.add(u)
            stack.extend(sorted(self.edges.get(u, ())))
        return False

    def victim(self, cycle: Sequence[str], cost: Optional[Dict[str, float]] = None) -> str:
        """挑回滚受害者: 代价最小者; 相同按 id 字典序 —— 保证确定性。"""
        cost = cost or {}
        return min(cycle, key=lambda n: (cost.get(n, 0.0), n))


def _canonical_cycle(cyc: Sequence[str]) -> List[str]:
    """把环旋转到字典序最小的起点, 便于去重。"""
    if not cyc:
        return []
    i = min(range(len(cyc)), key=lambda k: cyc[k])
    return list(cyc[i:]) + list(cyc[:i])


@dataclass
class Lease:
    holder: str
    resource: str
    granted_at: float
    ttl_s: float
    priority: float = 0.0

    def expires_at(self) -> float:
        return self.granted_at + self.ttl_s

    def expired(self, now: float) -> bool:
        return now >= self.expires_at()


@dataclass
class LeaseEvent:
    kind: str            # grant | deny | expire | preempt | release
    holder: str
    resource: str
    detail: str = ""


class LeaseManager:
    """有期限的资源持有 —— 最后一道防线: 前两层都漏了, 租约到期也会自动松手。"""

    def __init__(self, order: Optional[ResourceOrder] = None,
                 default_ttl_s: float = 30.0):
        self.order = order
        self.default_ttl_s = default_ttl_s
        self.held: Dict[str, Lease] = {}
        self.wfg = WaitForGraph()
        self.events: List[LeaseEvent] = []

    def _log(self, kind, holder, resource, detail="") -> LeaseEvent:
        e = LeaseEvent(kind, holder, resource, detail)
        self.events.append(e)
        return e

    def acquire(self, holder: str, resource: str, now: float,
                ttl_s: Optional[float] = None, priority: float = 0.0,
                allow_preempt: bool = True) -> LeaseEvent:
        cur = self.held.get(resource)
        if cur and cur.expired(now):
            self._log("expire", cur.holder, resource, "TTL 到期自动释放")
            self.wfg.remove(holder, cur.holder)
            cur = None
            self.held.pop(resource, None)

        if cur is None:
            self.held[resource] = Lease(holder, resource, now,
                                        ttl_s if ttl_s is not None else self.default_ttl_s,
                                        priority)
            return self._log("grant", holder, resource)
        if cur.holder == holder:
            return self._log("grant", holder, resource, "重入")
        if allow_preempt and priority > cur.priority:
            self._log("preempt", cur.holder, resource,
                      f"被 {holder} 抢占(优先级 {priority} > {cur.priority})")
            self.held[resource] = Lease(holder, resource, now,
                                        ttl_s if ttl_s is not None else self.default_ttl_s,
                                        priority)
            return self._log("grant", holder, resource, "抢占后授予")
        if self.wfg.would_deadlock(holder, cur.holder):
            return self._log("deny", holder, resource,
                             f"授予会与 {cur.holder} 形成循环等待, 拒绝以防死锁")
        self.wfg.add_wait(holder, cur.holder, resource)
        return self._log("deny", holder, resource, f"资源被 {cur.holder} 持有, 进入等待")

    def release(self, holder: str, resource: str) -> Optional[LeaseEvent]:
        cur = self.held.get(resource)
        if cur and cur.holder == holder:
            del self.held[resource]
            for waiter in list(self.wfg.edges):
                self.wfg.remove(waiter, holder)
            return self._log("release", holder, resource)
        return None

    def sweep(self, now: float) -> List[LeaseEvent]:
        out = []
        for res, lease in sorted(self.held.items()):
            if lease.expired(now):
                del self.held[res]
                out.append(self._log("expire", lease.holder, res, "sweep 回收"))
        return out

    def acquire_all(self, holder: str, resources: Sequence[str], now: float,
                    **kw) -> List[LeaseEvent]:
        """按全序批量申请 —— 这是预防层的实际用法。"""
        seq = self.order.plan(resources) if self.order else sorted(set(resources))
        return [self.acquire(holder, r, now, **kw) for r in seq]


__all__ = ["Lease", "LeaseEvent", "LeaseManager", "ResourceOrder", "WaitForGraph"]
