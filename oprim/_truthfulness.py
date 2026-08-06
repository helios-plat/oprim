"""oprim._truthfulness — 激励相容的经验检验: 暴力最优反应替代"算纳什均衡"。

你真正想知道的是: **"如实报价"会不会被某个谎报打败?** 这是最优反应问题,
不是均衡计算问题。nashpy 只支持两人有限博弈, N 人纳什计算是 PPAD-complete,
不该出现在请求路径上。

这里固定其他人报价, 对某个 Worker 的报价扫一个网格, 看有没有严格获利的谎报。
**离线跑, 不在请求路径上。** 这是证伪工具: 发现偏离就证明可被操纵;
没发现只说明网格上没找到, 不构成激励相容的证明(那要看机制的理论性质)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from ._allocate import Allocation, assign_one_to_one
from ._ledger import Problem

Allocator = Callable[[Problem], Allocation]
PaymentRule = Callable[..., object]


@dataclass
class Deviation:
    worker_id: str
    task_id: str
    truthful_bid: float
    misreport: float
    truthful_utility: float
    misreport_utility: float

    @property
    def gain(self) -> float:
        return round(self.misreport_utility - self.truthful_utility, 6)


@dataclass
class TruthfulnessReport:
    rule: str
    manipulable: bool
    deviations: List[Deviation] = field(default_factory=list)
    probes: int = 0

    @property
    def best_deviation(self) -> Optional[Deviation]:
        return max(self.deviations, key=lambda d: d.gain) if self.deviations else None

    def summary(self) -> str:
        if not self.manipulable:
            return f"{self.rule}: 在检验网格上未发现有利可图的谎报({self.probes} 次探测)"
        d = self.best_deviation
        return (f"{self.rule}: **可被操纵** —— {d.worker_id} 把对 {d.task_id} 的报价"
                f"从真实成本 {d.truthful_bid:g} 抬到 {d.misreport:g}, "
                f"效用 {d.truthful_utility:g} → {d.misreport_utility:g}"
                f"(多赚 {d.gain:g}, {self.probes} 次探测)")


def _utility(p: Problem, worker_id: str, true_costs: Dict[str, float],
             allocator: Allocator, payment_rule: PaymentRule) -> float:
    alloc = allocator(p)
    pr = payment_rule(p, alloc, allocator=allocator) \
        if payment_rule.__name__ == "vcg" else payment_rule(p, alloc)
    got = pr.payments.get(worker_id, 0.0)
    cost = sum(true_costs[t] for w, t in alloc.pairs
               if w == worker_id and t in true_costs)
    return round(got - cost, 6)


def check_strategyproof(p: Problem,
                        payment_rule: PaymentRule,
                        *,
                        allocator: Optional[Allocator] = None,
                        multipliers: Sequence[float] = (0.5, 0.8, 0.9, 1.1, 1.25,
                                                        1.5, 2.0, 3.0),
                        workers: Optional[Sequence[str]] = None) -> TruthfulnessReport:
    """暴力检验: 每个 Worker 对每个任务的报价乘一组系数, 看有没有严格获利的谎报。"""
    allocator = allocator or assign_one_to_one
    rule_name = getattr(payment_rule, "__name__", "rule")

    true_costs_by_worker: Dict[str, Dict[str, float]] = {}
    for b in p.bids:
        true_costs_by_worker.setdefault(b.worker_id, {})[b.task_id] = b.cost

    targets = list(workers) if workers else sorted(true_costs_by_worker)
    report = TruthfulnessReport(rule_name, False)

    for wid in targets:
        truths = true_costs_by_worker[wid]
        base_u = _utility(p, wid, truths, allocator, payment_rule)
        for tid, true_cost in sorted(truths.items()):
            for k in multipliers:
                mis = round(true_cost * k, 6)
                if abs(mis - true_cost) < 1e-12:
                    continue
                p2 = p.with_bid_override(wid, tid, mis)
                u = _utility(p2, wid, truths, allocator, payment_rule)
                report.probes += 1
                if u > base_u + 1e-9:
                    report.manipulable = True
                    report.deviations.append(Deviation(wid, tid, true_cost, mis, base_u, u))

    report.deviations.sort(key=lambda d: (-d.gain, d.worker_id, d.task_id))
    return report


__all__ = ["Deviation", "TruthfulnessReport", "check_strategyproof"]
