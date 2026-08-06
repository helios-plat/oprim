"""oprim._ir_solve — 求解层: 阶段 A 可行性 + MUS, 阶段 B MaxSMT 最优化。

为什么分两个阶段: "能不能做到" 和 "怎样做最好" 是两个问题, 反馈形态不同:
  - 阶段 A 不可行 → 需要**矛盾核心**回灌 LLM 反思
  - 阶段 B 有软约束落空 → 需要**被放弃的偏好清单**回灌用户确认

三态是一等公民: check() 返回 sat / unsat / **unknown**。
unknown 必须单独处理并升级, 绝不能塞进 unsat 分支 ——
那等于把"我不知道"谎报成"我证明了不可能"。

确定性: sat 给的是"一个"模型不是"那个"模型。字典序打桩
(按变量名序追加 minimize) 在 lex 优先级下把等价最优解压成唯一一个 ——
这是让分配结果跨 z3 版本可复现的关键。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ._ir_compile import Compiled
from ._mus import MusResult, shrink_to_mus


def _safe_set_param(z3, name: str, value: Any) -> None:
    """Z3 各版本参数名有出入, 未知参数不该让整个流程崩掉。"""
    try:
        z3.set_param(name, value)
    except Exception:
        pass


def install_determinism(z3, seed: int = 0) -> None:
    """固定随机种子。注意: 这**不足以**保证解唯一, 只是必要条件之一。
    真正锁死解的是 optimize() 里的字典序打桩。"""
    _safe_set_param(z3, "smt.random_seed", seed)
    _safe_set_param(z3, "sat.random_seed", seed)
    _safe_set_param(z3, "smt.arith.random_initial_value", False)


@dataclass
class Feasibility:
    status: str                                  # sat | unsat | unknown
    mus: Optional[MusResult] = None
    reason_unknown: str = ""
    elapsed_ms: int = 0
    checks: int = 0


def check_feasible(c: Compiled,
                   timeout_ms: int = 5000,
                   protected: Optional[Set[str]] = None,
                   mus_max_checks: int = 200) -> Feasibility:
    z3 = c.z3
    t0 = time.time()
    s = z3.Solver()
    s.set("timeout", timeout_ms)

    for cid, expr in c.hard.items():
        s.add(z3.Implies(c.selectors[cid], expr))

    all_ids = sorted(c.hard)
    r = s.check(*[c.selectors[i] for i in all_ids])
    checks = 1

    if r == z3.sat:
        return Feasibility("sat", elapsed_ms=int((time.time() - t0) * 1000), checks=checks)
    if r == z3.unknown:
        return Feasibility("unknown", reason_unknown=str(s.reason_unknown()),
                           elapsed_ms=int((time.time() - t0) * 1000), checks=checks)

    # unsat: 取核并收缩
    sel_to_cid = {str(c.selectors[i]): i for i in all_ids}
    raw_core = [sel_to_cid[str(b)] for b in s.unsat_core() if str(b) in sel_to_cid]
    if not raw_core:
        raw_core = all_ids            # 求解器没给核(罕见), 退化为全集起点

    def oracle(subset):
        nonlocal checks
        checks += 1
        v = s.check(*[c.selectors[i] for i in subset])
        return "sat" if v == z3.sat else ("unsat" if v == z3.unsat else "unknown")

    res = shrink_to_mus(oracle, raw_core, protected=protected or frozenset(),
                        max_checks=mus_max_checks)
    return Feasibility("unsat", mus=res,
                       elapsed_ms=int((time.time() - t0) * 1000), checks=checks + res.checks)


@dataclass
class Solution:
    status: str                                        # sat | unsat | unknown
    assignment: Dict[str, Any] = field(default_factory=dict)
    objective_value: Optional[str] = None
    satisfied_soft: List[str] = field(default_factory=list)
    relaxed_soft: List[str] = field(default_factory=list)   # 被牺牲的偏好 —— 必须回报给用户
    reason_unknown: str = ""
    elapsed_ms: int = 0


def _py_value(z3, val) -> Any:
    if z3.is_true(val):
        return True
    if z3.is_false(val):
        return False
    if z3.is_int_value(val):
        return val.as_long()
    if z3.is_rational_value(val):
        f = val.as_fraction()
        return float(f) if f.denominator != 1 else int(f)
    return str(val)


def optimize(c: Compiled,
             timeout_ms: int = 10000,
             deterministic_tiebreak: bool = True,
             objective_first: bool = True) -> Solution:
    """MaxSMT + 字典序打桩。

    deterministic_tiebreak: 对每个变量按名字序追加 minimize 目标。
        在 lex 优先级下, 这会把"多个同样最优的解"压成唯一一个。
        变量上千时改为: 只对关键决策变量打桩, 或求出最优值后加等式约束再枚举。
    objective_first: True = 显式目标压过软约束; False = 软约束压过显式目标。
    """
    z3 = c.z3
    t0 = time.time()
    opt = z3.Optimize()
    try:
        opt.set(priority="lex")
    except Exception:
        pass
    try:
        opt.set("timeout", timeout_ms)
    except Exception:
        pass

    for expr in c.hard.values():
        opt.add(expr)

    def _add_objective():
        if c.objective is None:
            return
        if c.objective_sense == "min":
            opt.minimize(c.objective)
        else:
            opt.maximize(c.objective)

    def _add_softs():
        for cid, expr in c.soft.items():
            w = c.weights.get(cid, 1.0)
            w = int(w) if float(w).is_integer() else str(w)
            opt.add_soft(expr, w, "prefs")

    if objective_first:
        _add_objective()
        _add_softs()
    else:
        _add_softs()
        _add_objective()

    if deterministic_tiebreak:
        for name in sorted(c.vars):
            zv = c.vars[name]
            opt.minimize(z3.If(zv, 1, 0) if c.var_types[name] == "bool" else zv)

    r = opt.check()
    ms = int((time.time() - t0) * 1000)

    if r == z3.unsat:
        return Solution("unsat", elapsed_ms=ms)
    if r == z3.unknown:
        return Solution("unknown", reason_unknown=str(opt.reason_unknown()), elapsed_ms=ms)

    m = opt.model()
    sol = Solution("sat", elapsed_ms=ms)
    for name, zv in c.vars.items():
        sol.assignment[name] = _py_value(z3, m.eval(zv, model_completion=True))
    for cid, expr in c.soft.items():
        (sol.satisfied_soft if z3.is_true(m.eval(expr, model_completion=True))
         else sol.relaxed_soft).append(cid)
    sol.satisfied_soft.sort()
    sol.relaxed_soft.sort()
    if c.objective is not None:
        sol.objective_value = str(m.eval(c.objective, model_completion=True))
    return sol


__all__ = ["Feasibility", "Solution", "check_feasible", "install_determinism", "optimize"]
