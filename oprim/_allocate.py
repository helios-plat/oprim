"""oprim._allocate — 分配器: 用组合最优化替代拍卖排队。

为什么不是纳什均衡: 需求是"全局最优 + 消除分配不均 + 无死锁", 分别对应:
  * 全局最优   → assignment problem 的最优解(匈牙利 / MILP), **不是**均衡解。
                纳什均衡不保证帕累托最优 —— 囚徒困境的唯一均衡就是严格帕累托劣的。
  * 分配不均   → 目标函数里的均衡项 / 容量约束, 不是博弈。
  * 死锁       → 资源全序 + wait-for 图(见 _deadlock), 和博弈论无关。
而且混合策略纳什均衡输出的是**概率分布**, 执行要掷骰子, 与确定性目标冲突。

确定性: 两种解法都可能存在多个等价最优解, scipy 一升级分配就可能变。
所以两边都做字典序打桩(先求最优值, 再在"不劣于最优值"的约束下按固定次序打破平局)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ._ledger import Bid, Problem, Task, Worker

UNASSIGNED = "__unassigned__"


def _np():
    import numpy  # noqa: PLC0415 - 懒加载
    return numpy


@dataclass
class Allocation:
    pairs: List[Tuple[str, str]] = field(default_factory=list)   # (worker_id, task_id)
    unassigned: List[str] = field(default_factory=list)
    total_cost: float = 0.0
    method: str = ""
    solver_status: str = "ok"
    note: str = ""

    def by_worker(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for w, t in self.pairs:
            out.setdefault(w, []).append(t)
        return {k: sorted(v) for k, v in sorted(out.items())}

    def worker_of(self, task_id: str) -> Optional[str]:
        for w, t in self.pairs:
            if t == task_id:
                return w
        return None


def cost_matrix(p: Problem, risk_adjusted: bool = True) -> Tuple["object", List[str], List[str]]:
    """构造 |workers| × |tasks| 成本矩阵。没有报价 / 技能不匹配的格子填 inf 代理值。"""
    np = _np()
    ws = [w.id for w in sorted(p.workers, key=lambda x: x.id)]
    ts = [t.id for t in sorted(p.tasks, key=lambda x: x.id)]
    wmap = {w.id: w for w in p.workers}
    tmap = {t.id: t for t in p.tasks}
    bm = p.bid_map()

    C = np.full((len(ws), len(ts)), p.unassigned_penalty, dtype=float)
    for i, wid in enumerate(ws):
        for j, tid in enumerate(ts):
            need = set(tmap[tid].requires_skills)
            if need and not need.issubset(set(wmap[wid].skills)):
                continue                                   # 技能不匹配 → 保持惩罚值
            b = bm.get((wid, tid))
            if b is None:
                continue
            C[i, j] = b.risk_adjusted if risk_adjusted else b.cost
    return C, ws, ts


def _tiebreak(C) -> "object":
    """加一个确定性的极小扰动打破平局: 量级远小于任何真实成本差,
    不改变最优解集合, 只在多个最优解之间做出可复现的选择。"""
    np = _np()
    n, m = C.shape
    scale = 1e-9 * (1.0 + float(np.max(np.abs(C[np.isfinite(C)])) if C.size else 1.0))
    idx = np.arange(n * m, dtype=float).reshape(n, m) / max(1, n * m)
    return C + scale * idx


def assign_one_to_one(p: Problem, risk_adjusted: bool = True) -> Allocation:
    """匈牙利算法。O(n³), 精确最优, 确定性。"""
    from scipy.optimize import linear_sum_assignment  # noqa: PLC0415 - 懒加载

    C, ws, ts = cost_matrix(p, risk_adjusted)
    if not ws or not ts:
        return Allocation(unassigned=sorted(t.id for t in p.tasks),
                          method="hungarian", note="没有 worker 或没有任务")

    n, m = C.shape
    if n < m:                                    # 补虚拟 worker, 保证每个任务都有列可配
        C = _np().vstack([C, _np().full((m - n, m), p.unassigned_penalty)])
    elif m < n:
        C = _np().hstack([C, _np().zeros((n, n - m))])   # 虚拟任务代价 0 = worker 闲置

    r, c = linear_sum_assignment(_tiebreak(C))
    pairs, unassigned, total = [], [], 0.0
    for i, j in zip(r.tolist(), c.tolist()):
        if i >= len(ws) or j >= len(ts):
            continue                             # 虚拟行/列
        if C[i, j] >= p.unassigned_penalty:      # 分到"不可行"格子 = 实际没人能接
            unassigned.append(ts[j])
            continue
        pairs.append((ws[i], ts[j]))
        total += float(C[i, j])

    assigned = {t for _, t in pairs}
    unassigned += [t for t in ts if t not in assigned and t not in unassigned]
    return Allocation(sorted(pairs), sorted(unassigned), round(total, 6), "hungarian")


def assign_with_capacity(p: Problem, risk_adjusted: bool = True,
                         balance_weight: float = 0.0) -> Allocation:
    """MILP: 一人可接多任务, 受多维资源容量与 max_tasks 约束。

    balance_weight > 0 时给"单人负载上限"加惩罚项把负载压平 ——
    这才是"消除分配不均"的正确做法: 写进目标函数, 不是靠均衡去凑。
    """
    from scipy.optimize import Bounds, LinearConstraint, milp  # noqa: PLC0415 - 懒加载

    np = _np()
    C, ws, ts = cost_matrix(p, risk_adjusted)
    if not ws or not ts:
        return Allocation(unassigned=sorted(t.id for t in p.tasks), method="milp")

    n, m = C.shape
    wmap = {w.id: w for w in p.workers}
    tmap = {t.id: t for t in p.tasks}
    resources = sorted({r for w in p.workers for r in w.capacity}
                       | {r for t in p.tasks for r in t.demand})

    NX, NU = n * m, m
    NV = NX + NU + 1
    LI = NX + NU

    def xi(i, j):
        return i * m + j

    c = np.zeros(NV)
    c[:NX] = C.flatten()
    c[NX:NX + NU] = [p.unassigned_penalty * tmap[t].priority for t in ts]
    c[LI] = balance_weight

    cons = []
    # ① 每个任务恰好被分配一次, 或被标记为未分配
    A = np.zeros((m, NV))
    for j in range(m):
        for i in range(n):
            A[j, xi(i, j)] = 1.0
        A[j, NX + j] = 1.0
    cons.append(LinearConstraint(A, 1, 1))
    # ② 每个 worker 的任务数上限
    A2 = np.zeros((n, NV))
    ub = np.zeros(n)
    for i, wid in enumerate(ws):
        for j in range(m):
            A2[i, xi(i, j)] = 1.0
        ub[i] = wmap[wid].max_tasks
    cons.append(LinearConstraint(A2, 0, ub))
    # ③ 多维资源容量
    if resources:
        A3 = np.zeros((n * len(resources), NV))
        ub3 = np.zeros(n * len(resources))
        for i, wid in enumerate(ws):
            for k, res in enumerate(resources):
                row = i * len(resources) + k
                for j, tid in enumerate(ts):
                    A3[row, xi(i, j)] = tmap[tid].demand.get(res, 0.0)
                ub3[row] = wmap[wid].capacity.get(res, 0.0)
        cons.append(LinearConstraint(A3, 0, ub3))
    # ④ 负载均衡: L >= 每个 worker 的任务数
    if balance_weight > 0:
        A4 = np.zeros((n, NV))
        for i in range(n):
            for j in range(m):
                A4[i, xi(i, j)] = 1.0
            A4[i, LI] = -1.0
        cons.append(LinearConstraint(A4, -np.inf, 0))

    integrality = np.ones(NV)
    integrality[LI] = 0
    lo = np.zeros(NV)
    hi = np.ones(NV)
    hi[LI] = float(m)

    res = milp(c=c, constraints=cons, integrality=integrality, bounds=Bounds(lo, hi))
    if res.status != 0 or res.x is None:
        return Allocation(unassigned=sorted(ts), method="milp",
                          solver_status=str(res.status), note=res.message)

    # 字典序打桩: 目标值锁死在最优, 再按固定次序最小化"选中格子序号和"
    z_star = float(res.fun)
    cons2 = list(cons) + [LinearConstraint(c.reshape(1, -1), -np.inf, z_star + 1e-7)]
    c2 = np.zeros(NV)
    c2[:NX] = np.arange(NX, dtype=float) + 1.0
    res2 = milp(c=c2, constraints=cons2, integrality=integrality, bounds=Bounds(lo, hi))
    x = (res2.x if res2.status == 0 and res2.x is not None else res.x)

    pairs, total = [], 0.0
    for i, wid in enumerate(ws):
        for j, tid in enumerate(ts):
            if x[xi(i, j)] > 0.5:
                pairs.append((wid, tid))
                total += float(C[i, j])
    unassigned = sorted(ts[j] for j in range(m) if x[NX + j] > 0.5)
    return Allocation(sorted(pairs), unassigned, round(total, 6), "milp")


def welfare(p: Problem, alloc: Allocation) -> float:
    """社会福利的负值口径: 总成本 + 未分配惩罚。越小越好。VCG 支付要用。"""
    tmap = {t.id: t for t in p.tasks}
    return alloc.total_cost + sum(
        p.unassigned_penalty * tmap[t].priority for t in alloc.unassigned if t in tmap)


__all__ = ["Allocation", "UNASSIGNED", "assign_one_to_one", "assign_with_capacity",
           "cost_matrix", "welfare"]
