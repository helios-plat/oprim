"""oprim._backtranslate — 回译 Diff, 抓 LLM 的翻译幻觉。

核心认知: **Z3 只验证"你给它的约束", 不验证"你给的是对的约束"。**
翻译层是新的幻觉集中地 —— 因为输出带着数学证明的外观。

本模块把 IR 表达式渲染回自然语言, 再和约束自带的 intent 做**确定性**比对:
  1. NUM_DRIFT  常量漂移: 表达式里的数字在意图里没出现   (命中率最高)
  2. DIR_FLIP   方向翻转: 意图说"至少", 表达式写成了 <=   (危害最大)
  3. VAR_MISS   变量遗漏: 意图提到的量没进表达式
  4. LOW_SIM    文本相似度过低(仅提示, 不阻断)
前三条 FAIL(阻断), 第四条 WARN。想上语义审查用 Reviewer 协议挂独立模型,
但它只能追加 finding, 不能推翻 FAIL。
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

from ._plan_ir import PlanIR, VarDecl

_CMP = {"<=": "{0} 不超过 {1}", "<": "{0} 严格小于 {1}",
        ">=": "{0} 至少为 {1}", ">": "{0} 严格大于 {1}",
        "==": "{0} 等于 {1}", "!=": "{0} 不等于 {1}"}


def _fmt_num(v: Any) -> str:
    if isinstance(v, bool):
        return "真" if v else "假"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def render(node: Any, vm: Dict[str, VarDecl], top: bool = True) -> str:
    """IR 表达式 → 中文。刻意做得啰嗦, 方便和意图逐词比对。"""
    if "var" in node:
        v = vm.get(node["var"])
        if v is None:
            return f"<未声明:{node['var']}>"
        unit = f"（{v.unit}）" if v.unit else ""
        return (v.desc or v.name) + unit
    if "lit" in node:
        return _fmt_num(node["lit"])

    op, args = node["op"], node.get("args", [])
    r = [render(a, vm, top=False) for a in args]

    if op in _CMP:
        return _CMP[op].format(r[0], r[1])
    if op in ("+", "sum"):
        return ("、".join(r) + " 之和") if len(r) > 2 else " 加 ".join(r)
    if op == "-":
        return f"负 {r[0]}" if len(r) == 1 else " 减 ".join(r)
    if op == "*":
        return " 乘以 ".join(r)
    if op == "/":
        return f"{r[0]} 除以 {r[1]}"
    if op == "abs":
        return f"{r[0]} 的绝对值"
    if op == "ite":
        return f"若 {r[0]} 则取 {r[1]} 否则取 {r[2]}"
    if op == "and":
        return "同时满足: " + "；".join(r) if top else "（" + " 且 ".join(r) + "）"
    if op == "or":
        return "至少满足其一: " + "；".join(r) if top else "（" + " 或 ".join(r) + "）"
    if op == "not":
        return f"并非（{r[0]}）"
    if op == "implies":
        return f"若 {r[0]}, 则 {r[1]}"
    if op == "iff":
        return f"{r[0]} 当且仅当 {r[1]}"
    if op == "distinct":
        return "、".join(r) + " 两两不同"
    return f"{op}({', '.join(r)})"


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_CN_MULT = {"万": 10000, "千": 1000, "百": 100, "k": 1000, "K": 1000, "m": 1000000, "M": 1000000}

_GE_WORDS = ("至少", "不少于", "不低于", "不小于", "大于", "超过", "起步", "下限", ">=", "≥", ">")
_LE_WORDS = ("至多", "不超过", "不多于", "不高于", "不大于", "小于", "低于", "上限", "封顶", "<=", "≤", "<")


@dataclass
class DiffFinding:
    code: str
    severity: str          # FAIL | WARN | INFO
    message: str
    hint: str = ""


@dataclass
class DiffReport:
    cid: str
    intent: str
    rendered: str
    similarity: float
    findings: List[DiffFinding] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return any(f.severity == "FAIL" for f in self.findings)


class Reviewer(Protocol):
    """可选的第二模型审查钩子。只能追加 finding, 不能推翻既有 FAIL。"""
    def review(self, intent: str, rendered: str) -> Sequence[DiffFinding]: ...


def _numbers_in(text: str) -> set:
    out = set()
    for m in _NUM_RE.finditer(text):
        val = float(m.group())
        tail = text[m.end():m.end() + 1]
        if tail in _CN_MULT:
            val *= _CN_MULT[tail]
        out.add(val)
    return out


def _expr_literals(node: Any) -> set:
    out = set()
    if isinstance(node, dict):
        if "lit" in node and isinstance(node["lit"], (int, float)) and not isinstance(node["lit"], bool):
            out.add(float(node["lit"]))
        for a in node.get("args", []) or []:
            out |= _expr_literals(a)
    return out


def _vars_in(node: Any) -> set:
    out = set()
    if isinstance(node, dict):
        if "var" in node:
            out.add(node["var"])
        for a in node.get("args", []) or []:
            out |= _vars_in(a)
    return out


def _top_direction(node: Any) -> Optional[str]:
    if isinstance(node, dict) and node.get("op") in ("<=", "<"):
        return "le"
    if isinstance(node, dict) and node.get("op") in (">=", ">"):
        return "ge"
    return None


def diff_one(cid: str, intent: str, expr: Any, vm: Dict[str, VarDecl],
             reviewer: Optional[Reviewer] = None,
             sim_threshold: float = 0.25) -> DiffReport:
    rendered = render(expr, vm)
    sim = difflib.SequenceMatcher(None, intent, rendered).ratio()
    rep = DiffReport(cid=cid, intent=intent, rendered=rendered, similarity=round(sim, 3))

    # 1. 常量漂移
    in_intent, in_expr = _numbers_in(intent), _expr_literals(expr)
    drifted = {x for x in in_expr if x not in in_intent}
    if drifted:
        rep.findings.append(DiffFinding(
            "NUM_DRIFT", "FAIL",
            f"表达式含常量 {sorted(drifted)}, 但意图文本 {intent!r} 里没有这些数字",
            "逐字核对阈值。若意图里的数字带单位换算(如 1 万 → 10000), "
            "请在 intent 中写出换算后的数值, 或在 vars 的 unit 字段声明单位"))

    # 2. 方向翻转
    d = _top_direction(expr)
    if d:
        wants_ge = any(w in intent for w in _GE_WORDS)
        wants_le = any(w in intent for w in _LE_WORDS)
        if d == "le" and wants_ge and not wants_le:
            rep.findings.append(DiffFinding(
                "DIR_FLIP", "FAIL", "意图表达的是下界(至少/不少于), 表达式却是上界(<=/<)",
                "把算子改成 >= 或 >, 或交换两侧操作数"))
        if d == "ge" and wants_le and not wants_ge:
            rep.findings.append(DiffFinding(
                "DIR_FLIP", "FAIL", "意图表达的是上界(不超过/至多), 表达式却是下界(>=/>)",
                "把算子改成 <= 或 <, 或交换两侧操作数"))

    # 3. 变量遗漏
    used = _vars_in(expr)
    missed = [n for n, v in vm.items()
              if n not in used and ((v.desc and v.desc in intent) or
                                    (len(n) > 2 and n in intent))]
    if missed:
        rep.findings.append(DiffFinding(
            "VAR_MISS", "FAIL",
            f"意图提到了 {missed}, 但表达式里没用到",
            "补进表达式, 或从 intent 里删掉不属于本条约束的描述"))

    # 4. 相似度(仅提示)
    if sim < sim_threshold:
        rep.findings.append(DiffFinding(
            "LOW_SIM", "WARN",
            f"回译文本与意图相似度仅 {sim:.2f}, 人工确认一下",
            "可能是措辞差异, 也可能是整条约束翻错了"))

    if reviewer is not None:
        rep.findings.extend(reviewer.review(intent, rendered))
    return rep


def diff_all(ir: PlanIR, reviewer: Optional[Reviewer] = None) -> List[DiffReport]:
    vm = ir.var_map
    return [diff_one(c.id, c.intent, c.expr, vm, reviewer) for c in ir.constraints]


__all__ = ["DiffFinding", "DiffReport", "Reviewer", "diff_all", "diff_one", "render"]
