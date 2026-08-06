"""oprim._ir_compile — Plan IR → Z3 编译器。

关键设计: **assumption literal(选择子) 而非 assert_and_track** ——
MUS 收缩需要反复检查任意约束子集, 用选择子后:
    s.add(Implies(sel_i, expr_i))          # 一次性装载, assertion stack 永不动
    s.check(*[sel_i for i in subset])      # 任意子集, 增量求解器全程复用
没有 push/pop、没有重建, 几十次收缩迭代的成本才压得住。

z3 懒加载: 只有真正编译/求解时才 import z3 (oprim 元数据扫描不触发)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ._plan_ir import Constraint, PlanIR

SEL_PREFIX = "__sel__"


def _z3() -> Any:
    import z3  # noqa: PLC0415 - 懒加载: 无 z3 环境也能 import oprim
    return z3


@dataclass
class Compiled:
    z3: Any                                   # z3 模块句柄
    vars: Dict[str, Any] = field(default_factory=dict)
    var_types: Dict[str, str] = field(default_factory=dict)
    hard: Dict[str, Any] = field(default_factory=dict)          # cid -> z3 表达式
    soft: Dict[str, Any] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    selectors: Dict[str, Any] = field(default_factory=dict)     # cid -> z3 Bool 选择子
    domain_ids: List[str] = field(default_factory=list)
    objective: Optional[Any] = None
    objective_sense: str = "min"


def _num(z3, value: Any, prefer_real: bool = False):
    if isinstance(value, bool):
        return z3.BoolVal(value)
    if isinstance(value, int) and not prefer_real:
        return z3.IntVal(value)
    # 用字符串构造 RealVal, 避开二进制浮点误差: RealVal("0.1") 是精确的 1/10
    return z3.RealVal(str(value))


def compile_expr(z3, node: Any, vars_: Dict[str, Any]):
    """把已通过校验的 IR 表达式编译成 z3 AST。

    假定 _plan_ir.validate() 已过关 —— 校验和编译职责分离,
    编译期报错说明校验器有洞, 应去补校验器而不是在这里 try/except。
    """
    if "var" in node:
        return vars_[node["var"]]
    if "lit" in node:
        return _num(z3, node["lit"])

    op = node["op"]
    args = [compile_expr(z3, a, vars_) for a in node.get("args", [])]

    if op in ("+", "sum"):
        out = args[0]
        for a in args[1:]:
            out = out + a
        return out
    if op == "-":
        if len(args) == 1:
            return -args[0]
        out = args[0]
        for a in args[1:]:
            out = out - a
        return out
    if op == "*":
        out = args[0]
        for a in args[1:]:
            out = out * a
        return out
    if op == "/":
        return args[0] / args[1]
    if op == "abs":
        return z3.If(args[0] >= 0, args[0], -args[0])
    if op == "ite":
        return z3.If(args[0], args[1], args[2])

    if op == "<=":
        return args[0] <= args[1]
    if op == "<":
        return args[0] < args[1]
    if op == ">=":
        return args[0] >= args[1]
    if op == ">":
        return args[0] > args[1]
    if op == "==":
        return args[0] == args[1]
    if op == "!=":
        # Not(a == b) 而非 a != b: 语义等价, 不依赖 __ne__ 重载行为
        return z3.Not(args[0] == args[1])

    if op == "and":
        return z3.And(*args) if len(args) > 1 else args[0]
    if op == "or":
        return z3.Or(*args) if len(args) > 1 else args[0]
    if op == "not":
        return z3.Not(args[0])
    if op == "implies":
        return z3.Implies(args[0], args[1])
    if op == "iff":
        return args[0] == args[1]
    if op == "distinct":
        return z3.Distinct(*args)

    raise ValueError(f"编译器未覆盖算子 {op!r}(校验器有洞, 去补 _plan_ir.ALL_OPS 与本函数)")


def compile_ir(ir: PlanIR, z3=None) -> Compiled:
    """IR → Compiled。定义域被展开成显式约束 —— 这样它们也能出现在 MUS 里。

    真实场景里大量 unsat 的根因就是定义域; 藏在变量声明里不参与归因的话,
    LLM 会盯着业务约束反复瞎改。
    """
    z3 = z3 or _z3()
    out = Compiled(z3=z3)

    for v in ir.vars:
        if v.type == "int":
            zv = z3.Int(v.name)
        elif v.type == "real":
            zv = z3.Real(v.name)
        else:
            zv = z3.Bool(v.name)
        out.vars[v.name] = zv
        out.var_types[v.name] = v.type

    def _add_hard(cid: str, expr) -> None:
        out.hard[cid] = expr
        out.selectors[cid] = z3.Bool(SEL_PREFIX + cid)

    for v in ir.vars:
        if v.type == "bool":
            continue
        if v.lo is not None:
            cid = f"dom_{v.name}_lo"
            _add_hard(cid, out.vars[v.name] >= _num(z3, v.lo, v.type == "real"))
            out.domain_ids.append(cid)
        if v.hi is not None:
            cid = f"dom_{v.name}_hi"
            _add_hard(cid, out.vars[v.name] <= _num(z3, v.hi, v.type == "real"))
            out.domain_ids.append(cid)

    for c in ir.constraints:
        expr = compile_expr(z3, c.expr, out.vars)
        if c.kind == "soft":
            out.soft[c.id] = expr
            out.weights[c.id] = c.weight
        else:
            _add_hard(c.id, expr)

    if ir.objective is not None:
        out.objective = compile_expr(z3, ir.objective.expr, out.vars)
        out.objective_sense = ir.objective.sense
    return out


def domain_constraint_meta(ir: PlanIR, cid: str) -> Optional[Constraint]:
    """给自动生成的定义域约束补一条伪 Constraint, 让 MUS 报告显示自然语言。"""
    if not cid.startswith("dom_"):
        return None
    body = cid[4:]
    if body.endswith("_lo"):
        name, bound = body[:-3], "lo"
    elif body.endswith("_hi"):
        name, bound = body[:-3], "hi"
    else:
        return None
    v = ir.var_map.get(name)
    if v is None:
        return None
    val = v.lo if bound == "lo" else v.hi
    label = "不小于" if bound == "lo" else "不大于"
    unit = f" {v.unit}" if v.unit else ""
    desc = v.desc or v.name
    return Constraint(id=cid, kind="hard",
                      intent=f"{desc} {label} {val}{unit}(变量定义域)",
                      expr={}, origin="domain")


__all__ = ["Compiled", "compile_expr", "compile_ir", "domain_constraint_meta", "SEL_PREFIX"]
