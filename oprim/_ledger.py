"""oprim._ledger — 分配问题数据模型与决策账本。

关于"报价"的定位: 这里保留 bid(报价), 但它的角色和一价拍卖里的报价不是一回事:
  * 拍卖里 bid 是策略性出价, 会被自利参与者压价(shade), 你拿到的不是真实成本;
  * 这里 bid 是**类型化信号**(成本估计 + ETA + 置信度), 由 O3 遥测拟合或账本校准,
    真正的分配由 _allocate 的组合最优化决定, 不是"谁报价低谁先拿"。

确定性: Problem.digest() 与 Ledger.replay_key() 都是内容寻址 ——
同输入重跑必须得到同样结果, 这是"确定性 = 可重放"在 O2 侧的落点。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class Task:
    id: str
    demand: Dict[str, float] = field(default_factory=dict)   # resource -> 需求量
    priority: float = 1.0                                     # 未被分配时的机会成本权重
    requires_skills: Sequence[str] = ()
    resources: Sequence[str] = ()                             # 需要独占持有的资源 id


@dataclass(frozen=True)
class Worker:
    id: str
    capacity: Dict[str, float] = field(default_factory=dict)
    skills: Sequence[str] = ()
    max_tasks: int = 1                                        # 一轮最多接几个任务


@dataclass(frozen=True)
class Bid:
    worker_id: str
    task_id: str
    cost: float
    eta_ms: int = 0
    confidence: float = 1.0           # ∈ (0,1], 用来对 cost 做风险调整

    @property
    def risk_adjusted(self) -> float:
        """置信度越低, 等效成本越高。confidence=1 时等于原值。"""
        c = min(1.0, max(1e-6, self.confidence))
        return self.cost / c


@dataclass
class Problem:
    tasks: List[Task]
    workers: List[Worker]
    bids: List[Bid]
    unassigned_penalty: float = 1e6   # 任务没人接的代价。必须显著大于任何真实成本,
                                      # 否则求解器会"选择性罢工"。

    def bid_map(self) -> Dict[tuple, Bid]:
        return {(b.worker_id, b.task_id): b for b in self.bids}

    def without_worker(self, worker_id: str) -> "Problem":
        """VCG 需要的反事实世界: 把某个 Worker 整个拿掉重算一遍。"""
        return Problem(list(self.tasks),
                       [w for w in self.workers if w.id != worker_id],
                       [b for b in self.bids if b.worker_id != worker_id],
                       self.unassigned_penalty)

    def with_bid_override(self, worker_id: str, task_id: str, cost: float) -> "Problem":
        """检验策略操纵时用: 把某人对某任务的报价换掉, 其他不变。"""
        out: List[Bid] = []
        hit = False
        for b in self.bids:
            if b.worker_id == worker_id and b.task_id == task_id:
                out.append(Bid(b.worker_id, b.task_id, cost, b.eta_ms, b.confidence))
                hit = True
            else:
                out.append(b)
        if not hit:
            out.append(Bid(worker_id, task_id, cost))
        return Problem(list(self.tasks), list(self.workers), out, self.unassigned_penalty)

    def digest(self) -> str:
        def canon(objs) -> list:
            # dict 之间不可比, 先各自序列化再按字符串排序 —— digest 与输入顺序无关
            return sorted(json.dumps(asdict(o), sort_keys=True, default=str,
                                     ensure_ascii=False, separators=(",", ":"))
                          for o in objs)

        blob = json.dumps({
            "tasks": canon(self.tasks),
            "workers": canon(self.workers),
            "bids": canon(self.bids),
            "penalty": self.unassigned_penalty,
        }, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:20]


@dataclass
class LedgerEntry:
    ts: float
    kind: str                          # allocate | payment | lease | preempt | release | escalate
    payload: Dict[str, Any]


class Ledger:
    """决策账本。每一次分配、付款、租约变更都在这里留痕。
    给定 problem.digest() + 策略参数, 重跑必须得到同样的 replay_key。"""

    def __init__(self):
        self.entries: List[LedgerEntry] = []

    def record(self, kind: str, **payload) -> LedgerEntry:
        e = LedgerEntry(time.time(), kind, payload)
        self.entries.append(e)
        return e

    def by_kind(self, kind: str) -> List[LedgerEntry]:
        return [e for e in self.entries if e.kind == kind]

    def replay_key(self) -> str:
        """账本内容指纹 —— 不含时间戳, 两次相同决策得到相同的 key。"""
        blob = json.dumps([{"kind": e.kind, "payload": e.payload} for e in self.entries],
                          sort_keys=True, default=str, ensure_ascii=False)
        return "led_" + hashlib.sha256(blob.encode()).hexdigest()[:20]


__all__ = ["Bid", "Ledger", "LedgerEntry", "Problem", "Task", "Worker"]
