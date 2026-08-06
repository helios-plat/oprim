"""oprim._plan_ir — 神经符号 Plan IR (schema / 解析 / 校验)。

设计原则:
1. LLM 只被允许产出这个 schema 的 JSON, 绝不产出 Python / SMT-LIB 源码。
2. 表达式是封闭算子集, 未知算子/非线性/整数除法陷阱全部在这里被拒绝,
   并生成结构化 RepairHint 回灌 LLM。
3. 每条约束自带 intent(原始自然语言) + origin(出处) —— MUS 反思与回译 diff 的前提。
4. 线性守卫把问题锁在 QF_LIA / QF_LRA 可判定片段, unknown 只可能来自超时。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

IR_VERSION = "o1.ir/v1"

ARITH_OPS = {
    "+": (1, None), "-": (1, None), "*": (2, None), "/": (2, 2),
    "sum": (1, None), "abs": (1, 1), "ite": (3, 3),
}
COMPARE_OPS = {"<=": (2, 2), "<": (2, 2), ">=": (2, 2), ">": (2, 2),
               "==": (2, 2), "!=": (2, 2)}
BOOL_OPS = {"and": (1, None), "or": (1, None), "not": (1, 1),
            "implies": (2, 2), "iff": (2, 2), "distinct": (2, None)}
ALL_OPS = {**ARITH_OPS, **COMPARE_OPS, **BOOL_OPS}
VAR_TYPES = ("int", "real", "bool")


@dataclass
class IRError:
    """结构化校验错误。hint 是直接喂回 LLM 的修复指令。"""
    code: str
    path: str
    message: str
    hint: str = ""

    def as_dict(self) -> Dict[str, str]:
        return {"code": self.code, "path": self.path,
                "message": self.message, "hint": self.hint}

    def __str__(self) -> str:
        return f"[{self.code}] {self.path}: {self.message}" + (
            f"\n    修复建议: {self.hint}" if self.hint else "")


class IRValidationError(Exception):
    def __init__(self, errors: Sequence[IRError]):
        self.errors = list(errors)
        super().__init__(f"{len(self.errors)} 个 IR 校验错误")


@dataclass
class VarDecl:
    name: str
    type: str
    desc: str = ""
    unit: str = ""
    lo: Optional[float] = None
    hi: Optional[float] = None


@dataclass
class Constraint:
    id: str
    kind: str                                   # hard | soft
    intent: str
    expr: Dict[str, Any]
    origin: str = ""
    weight: float = 1.0
    protected: bool = False                     # 不参与 MUS 归因 (安全不变量)


@dataclass
class Objective:
    sense: str                                  # min | max
    expr: Dict[str, Any]
    intent: str = ""


@dataclass
class PlanIR:
    intent: str
    vars: List[VarDecl] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)
    objective: Optional[Objective] = None
    version: str = IR_VERSION

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "version": self.version,
            "intent": self.intent,
            "vars": [
                {k: v for k, v in {
                    "name": x.name, "type": x.type, "desc": x.desc,
                    "unit": x.unit, "lo": x.lo, "hi": x.hi,
                }.items() if v not in ("", None)}
                for x in self.vars
            ],
            "constraints": [
                {k: v for k, v in {
                    "id": c.id, "kind": c.kind, "intent": c.intent,
                    "expr": c.expr, "origin": c.origin,
                    "weight": c.weight if c.kind == "soft" else None,
                    "protected": True if c.protected else None,
                }.items() if v not in ("", None)}
                for c in self.constraints
            ],
        }
        if self.objective:
            d["objective"] = {"sense": self.objective.sense,
                              "expr": self.objective.expr,
                              "intent": self.objective.intent}
        return d

    def canonical_json(self) -> str:
        """稳定序列化 —— plan_id 的内容寻址输入。"""
        return json.dumps(self.to_dict(), sort_keys=True,
                          ensure_ascii=False, separators=(",", ":"))

    @property
    def var_map(self) -> Dict[str, VarDecl]:
        return {v.name: v for v in self.vars}


def parse_ir(raw: Any) -> PlanIR:
    """dict / JSON 字符串 → PlanIR。只做形状检查, 语义交给 validate。"""
    if isinstance(raw, (str, bytes)):
        raw = json.loads(raw)
    if not isinstance(raw, dict):
        raise IRValidationError([IRError("E_SHAPE", "$", "顶层必须是 JSON 对象",
                                         "输出形如 {\"intent\":...,\"vars\":[...],\"constraints\":[...]}")])

    errs: List[IRError] = []
    if raw.get("version", IR_VERSION) != IR_VERSION:
        errs.append(IRError("E_VERSION", "$.version", "IR 版本不受支持",
                            f"使用 version={IR_VERSION!r}"))

    vars_: List[VarDecl] = []
    for i, v in enumerate(raw.get("vars", []) or []):
        if not isinstance(v, dict) or "name" not in v or "type" not in v:
            errs.append(IRError("E_VAR_SHAPE", f"$.vars[{i}]", "变量必须含 name 与 type",
                                "如 {\"name\":\"cpu_a\",\"type\":\"int\",\"lo\":0,\"hi\":64}"))
            continue
        vars_.append(VarDecl(str(v["name"]), str(v["type"]),
                             str(v.get("desc", "")), str(v.get("unit", "")),
                             v.get("lo"), v.get("hi")))

    cons: List[Constraint] = []
    for i, c in enumerate(raw.get("constraints", []) or []):
        if not isinstance(c, dict) or "expr" not in c:
            errs.append(IRError("E_CON_SHAPE", f"$.constraints[{i}]", "约束必须含 expr",
                                "如 {\"id\":\"c1\",\"kind\":\"hard\",\"intent\":\"...\",\"expr\":{...}}"))
            continue
        cons.append(Constraint(str(c.get("id") or f"c_{i}"), str(c.get("kind", "hard")),
                               str(c.get("intent", "")), c["expr"],
                               str(c.get("origin", "")), float(c.get("weight", 1.0)),
                               bool(c.get("protected", False))))

    obj = None
    if raw.get("objective"):
        o = raw["objective"]
        obj = Objective(str(o.get("sense", "min")), o.get("expr"), str(o.get("intent", "")))

    if errs:
        raise IRValidationError(errs)
    return PlanIR(intent=str(raw.get("intent", "")), vars=vars_,
                  constraints=cons, objective=obj)


def _is_const(node: Any) -> bool:
    if isinstance(node, dict):
        if "lit" in node:
            return True
        if "var" in node:
            return False
        if "op" in node:
            return all(_is_const(a) for a in node.get("args", []))
    return False


def _lit_type(v: Any) -> Optional[str]:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "real"
    return None


class _TypeChecker:
    """最小类型推断 + 线性守卫。副作用式收集错误, 一次性返回全部问题。"""

    def __init__(self, var_map: Dict[str, VarDecl]):
        self.vars = var_map
        self.errors: List[IRError] = []

    def err(self, code: str, path: str, msg: str, hint: str = "") -> None:
        self.errors.append(IRError(code, path, msg, hint))

    def infer(self, node: Any, path: str) -> Optional[str]:
        if not isinstance(node, dict):
            self.err("E_NODE", path, f"表达式节点必须是对象, 收到 {type(node).__name__}",
                     "只允许三种节点: {\"var\":名}、{\"lit\":值}、{\"op\":算子,\"args\":[...]}")
            return None
        if "var" in node:
            name = node["var"]
            if name not in self.vars:
                self.err("E_UNDEF_VAR", path, f"未声明的变量 {name!r}",
                         f"先在 $.vars 里声明 {name!r}, 或改用已声明变量: {sorted(self.vars)}")
                return None
            return self.vars[name].type
        if "lit" in node:
            t = _lit_type(node["lit"])
            if t is None:
                self.err("E_LIT", path, f"字面量类型不支持: {node['lit']!r}",
                         "只允许 int / float / bool 字面量; 字符串请建模为布尔变量")
            return t
        if "op" not in node:
            self.err("E_NODE", path, "节点既无 var/lit 也无 op",
                     "只允许 {\"var\":...} / {\"lit\":...} / {\"op\":...,\"args\":[...]}")
            return None

        op = node["op"]
        if op not in ALL_OPS:
            self.err("E_UNKNOWN_OP", path, f"未知算子 {op!r}", f"可用算子仅限: {sorted(ALL_OPS)}")
            return None
        args = node.get("args", [])
        if not isinstance(args, list):
            self.err("E_ARGS", path, "args 必须是数组")
            return None
        lo, hi = ALL_OPS[op]
        if len(args) < lo or (hi is not None and len(args) > hi):
            want = str(lo) if hi == lo else "%s~%s" % (lo, hi if hi else "N")
            self.err("E_ARITY", path, f"算子 {op!r} 需要 {want} 个参数, 收到 {len(args)}",
                     f"检查 {op!r} 的参数个数")
            return None
        kids = [self.infer(a, f"{path}.args[{i}]") for i, a in enumerate(args)]

        if op in ("and", "or", "not", "implies", "iff"):
            for i, t in enumerate(kids):
                if t is not None and t != "bool":
                    self.err("E_TYPE", f"{path}.args[{i}]",
                             f"{op!r} 的参数应为 bool, 推断为 {t}",
                             "把算术表达式包在比较算子里, 如 {\"op\":\"<=\",\"args\":[...]}")
            return "bool"
        if op == "distinct":
            return "bool"
        if op in COMPARE_OPS:
            for i, t in enumerate(kids):
                if t == "bool" and op not in ("==", "!="):
                    self.err("E_TYPE", f"{path}.args[{i}]",
                             f"{op!r} 不能比较 bool", "布尔相等请用 iff")
            return "bool"
        if op == "ite":
            if kids[0] is not None and kids[0] != "bool":
                self.err("E_TYPE", f"{path}.args[0]", "ite 的条件必须是 bool")
            return self._join(kids[1], kids[2])
        if op == "*":
            non_const = [i for i, a in enumerate(args) if not _is_const(a)]
            if len(non_const) > 1:
                self.err("E_NONLINEAR", path, "乘法中出现两个及以上非常量因子(变量×变量)",
                         "非线性会让问题跳出可判定片段、返回 unknown。请改写为线性形式")
                return None
        if op == "/":
            divisor = args[1] if len(args) > 1 else None
            if not _is_const(divisor):
                self.err("E_NONLINEAR", path, "除数必须是常量",
                         "变量除法非线性。请两边同乘除数改写为线性不等式")
                return None
            if isinstance(divisor, dict) and divisor.get("lit") in (0, 0.0):
                self.err("E_DIV_ZERO", path, "除数为 0")
                return None
            t = self._join(*kids)
            if t == "int":
                self.err("E_INT_DIV", path, "整数除法语义有陷阱(Z3 的 div 是欧几里得除法)",
                         "把不等式两边同乘除数改写。例如 a/3 <= 10 写成 a <= 30")
                return None
            return "real"
        if op == "abs":
            return kids[0]
        return self._join(*kids)

    @staticmethod
    def _join(*types: Optional[str]) -> Optional[str]:
        ts = [t for t in types if t]
        if not ts:
            return None
        if "real" in ts:
            return "real"
        if "bool" in ts and len(set(ts)) > 1:
            return None
        return ts[0]


def validate(ir: PlanIR) -> List[IRError]:
    """全量校验。返回空列表 = 通过。"""
    errs: List[IRError] = []
    seen_v = set()
    for v in ir.vars:
        if v.type not in VAR_TYPES:
            errs.append(IRError("E_VAR_TYPE", f"$.vars[{v.name}]", f"类型 {v.type!r} 不支持",
                                f"只允许 {VAR_TYPES}"))
        if v.name in seen_v:
            errs.append(IRError("E_DUP_VAR", f"$.vars[{v.name}]", "变量重名"))
        seen_v.add(v.name)
        if v.lo is not None and v.hi is not None and v.lo > v.hi:
            errs.append(IRError("E_DOMAIN", f"$.vars[{v.name}]",
                                f"定义域为空: lo={v.lo} > hi={v.hi}",
                                "这是必然 unsat 的根因, 先修定义域再谈约束"))

    seen_c = set()
    tc = _TypeChecker(ir.var_map)
    for c in ir.constraints:
        if c.id in seen_c:
            errs.append(IRError("E_DUP_ID", f"$.constraints[{c.id}]", "约束 id 重复",
                                "id 是 MUS 归因的锚点, 必须全局唯一"))
        seen_c.add(c.id)
        if c.kind not in ("hard", "soft"):
            errs.append(IRError("E_KIND", f"$.constraints[{c.id}]",
                                f"kind 必须是 hard/soft, 收到 {c.kind!r}"))
        if not c.intent.strip():
            errs.append(IRError("E_NO_INTENT", f"$.constraints[{c.id}]",
                                "缺少 intent(原始自然语言意图)",
                                "没有 intent, UNSAT 反思和回译 diff 都会失效。必须补齐"))
        t = tc.infer(c.expr, f"$.constraints[{c.id}].expr")
        if t is not None and t != "bool":
            errs.append(IRError("E_TYPE", f"$.constraints[{c.id}].expr",
                                f"约束顶层必须是 bool, 推断为 {t}",
                                "外面套一层比较算子, 如 {\"op\":\"<=\",\"args\":[<表达式>,{\"lit\":N}]}"))

    if ir.objective:
        if ir.objective.sense not in ("min", "max"):
            errs.append(IRError("E_OBJ", "$.objective.sense", "sense 必须是 min/max"))
        t = tc.infer(ir.objective.expr, "$.objective.expr")
        if t == "bool":
            errs.append(IRError("E_OBJ_TYPE", "$.objective.expr", "目标函数不能是 bool",
                                "目标应为数值表达式"))
    return errs + tc.errors


__all__ = ["ALL_OPS", "COMPARE_OPS", "BOOL_OPS", "ARITH_OPS", "IR_VERSION",
           "IRError", "IRValidationError", "VarDecl", "Constraint", "Objective",
           "PlanIR", "parse_ir", "validate"]
