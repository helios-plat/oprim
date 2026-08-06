"""oprim._payments — 支付规则。

一价密封拍卖不是激励相容的: 理性投标者会 shade bid(抬价), 赚取报价与真实成本之差。
你账本里拿到的因此**不是真实成本**。VCG 让如实报价成为占优策略:

    payment_i = welfare(没有 i 的世界) − welfare(有 i 的世界里除 i 之外的部分)

代价是 n+1 次分配求解。**只有当 Worker 是你控制不了效用函数的第三方/多租户 Agent 时
才需要它** —— 自家 Worker 没有隐藏的私有效用, 给它们套机制设计只是仪式。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from ._allocate import Allocation, assign_one_to_one, welfare
from ._ledger import Problem

Allocator = Callable[[Problem], Allocation]


@dataclass
class PaymentResult:
    rule: str
    payments: Dict[str, float] = field(default_factory=dict)      # worker_id -> 收到的钱
    solves: int = 1
    detail: Dict[str, str] = field(default_factory=dict)

    def total(self) -> float:
        return round(sum(self.payments.values()), 6)


def first_price(p: Problem, alloc: Allocation, **_) -> PaymentResult:
    """按自己的报价付。简单、直觉、**不激励相容** —— 这里作为对照组存在。"""
    bm = p.bid_map()
    pay: Dict[str, float] = {}
    for w, t in alloc.pairs:
        b = bm.get((w, t))
        if b:
            pay[w] = round(pay.get(w, 0.0) + b.cost, 6)
    return PaymentResult("first_price", pay, 1,
                         {"warning": "非激励相容: 理性 Worker 会抬价"})


def vcg(p: Problem, alloc: Allocation,
        allocator: Optional[Allocator] = None) -> PaymentResult:
    """VCG 支付。n+1 次求解, n = 中标 Worker 数。"""
    allocator = allocator or assign_one_to_one
    base_w = welfare(p, alloc)
    bm = p.bid_map()

    winners = sorted({w for w, _ in alloc.pairs})
    pay: Dict[str, float] = {}
    detail: Dict[str, str] = {}
    solves = 1

    for wid in winners:
        own_cost = sum(bm[(w, t)].cost for w, t in alloc.pairs
                       if w == wid and (w, t) in bm)
        others_in_base = base_w - own_cost

        counterfactual = allocator(p.without_worker(wid))
        solves += 1
        without_w = welfare(p.without_worker(wid), counterfactual)

        payment = round(without_w - others_in_base, 6)
        pay[wid] = payment
        detail[wid] = (f"w(-{wid})={without_w:.4f} − 其他人在基准解中的福利"
                       f"={others_in_base:.4f}")

    # VCG 不保证预算平衡。某个 Worker 不可替代时, 其外部性会吃进 unassigned_penalty。
    base_scale = max(1.0, abs(alloc.total_cost))
    dominated = [w for w, v in pay.items() if abs(v) > 10 * base_scale]
    if dominated:
        detail["_warning"] = (
            f"{dominated} 的支付被 unassigned_penalty({p.unassigned_penalty:g}) 主导: "
            f"拿掉它们会导致任务无人可接。VCG 不保证预算平衡, "
            f"要么调整 penalty 到真实机会成本, 要么加预算上限/保留价。")
    return PaymentResult("vcg", pay, solves, detail)


def second_price_per_task(p: Problem, alloc: Allocation, **_) -> PaymentResult:
    """逐任务二价: 按该任务的次低报价付。

    **注意**: 只有在任务完全独立(无容量耦合、一人一任务)时它才等价于 VCG。
    一旦有容量耦合就不再激励相容 —— _truthfulness 可验证你的具体设定。
    """
    by_task: Dict[str, list] = {}
    for b in p.bids:
        by_task.setdefault(b.task_id, []).append((b.cost, b.worker_id))

    pay: Dict[str, float] = {}
    for w, t in alloc.pairs:
        others = sorted(c for c, wid in by_task.get(t, []) if wid != w)
        price = others[0] if others else p.unassigned_penalty
        pay[w] = round(pay.get(w, 0.0) + price, 6)
    return PaymentResult("second_price", pay, 1,
                         {"caveat": "仅在任务独立时等价于 VCG"})


RULES: Dict[str, Callable] = {
    "first_price": first_price,
    "second_price": second_price_per_task,
    "vcg": vcg,
}


__all__ = ["PaymentResult", "RULES", "first_price", "second_price_per_task", "vcg"]
