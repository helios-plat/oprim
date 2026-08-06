"""oprim._lookahead_scan — 前瞻性与静态不变量校验引擎 (AST 硬扫描).

数学级法律: 不依赖 LLM "猜", 用 ``ast`` 语法树做硬编码契约扫描.
在代码交给协处理器回测**之前**挂载, 任何可能污染回测/实盘决策的行都会被标红.

契约 (rule_id):
    L1 lookahead_shift   未来函数: ``shift(-k)`` / ``shift(-1)`` 硬拦截.
                         例外: 赋值给监督标签列 (label/target/y/future/...)
                         降级为 WARNING (仅允许用于构造标签, 不得进决策向量).
    L2 future_index      未来行索引: ``df.iloc[i+1]`` / ``df.loc[i+1:]`` /
                         ``df.iloc[idx + 1]`` 硬拦截; 字面量 ``iloc[1:]``/``iloc[:-1]``
                         为对齐惯用法 → WARNING (需人工复核).
    L3 roll_negative     ``np.roll(x, -k)`` 负偏移 = 未来数据前移 → 硬拦截.
    L4 rolling_leakage   滚动统计量泄漏: rolling/expanding/ewm 统计量未 ``.shift(1)``
                         即参与决策 → WARNING (当前 bar 收盘已知数据进入同 bar 开盘
                         决策即泄漏; 必须滞后一拍).
    L5 div_zero_volume   除零风险: 分母为 volume 列的除法无守卫 (集合竞价量可为 0,
                         经典 VWAP 崩溃点) → WARNING.

verdict:
    block   存在 VIOLATION (硬拦截, 不许进入回测/实盘)
    review  仅 WARNING (放行但必须人工/审判复核)
    pass    干净
    error   源码无法解析 (按 block 处理)
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any

# 监督标签列 (未来函数仅允许出现在这些名字的赋值里)
_LABEL_NAME_RE = re.compile(r"(label|target|y$|y_|future|forward|ret_future|next_|horizon)", re.IGNORECASE)
# 决策/信号列 (泄漏规则关注的消费者)
_SIGNAL_NAME_RE = re.compile(r"(signal|score|weight|position|allocation|exposure|decision|side|order)", re.IGNORECASE)
# 滚动窗口统计方法 (泄漏规则关注的终结方法)
_ROLLING_TERMINAL = {"mean", "std", "sum", "min", "max", "median", "var", "quantile"}
# 时序算子 (泄漏规则关注的起点)
_ROLLING_SOURCES = {"rolling", "expanding", "ewm"}
_VOLUME_RE = re.compile(r"volume|vol$|vol_|turnover", re.IGNORECASE)


@dataclass
class LookaheadFinding:
    """单条静态校验发现.

    Attributes:
        rule_id:   契约编号 (L1..L5 / syntax).
        severity:  VIOLATION (硬拦截) | WARNING (复核) .
        line:      源码行号 (1 基).
        col:       列号.
        message:   人话描述.
        snippet:   该行源码摘录.
    """

    rule_id: str
    severity: str
    line: int
    col: int
    message: str
    snippet: str = ""


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _is_negative_const(node: ast.AST) -> bool:
    """shift(-1) / shift(-2) 之类的负常量参数."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and node.value < 0:
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = node.operand
        return isinstance(inner, ast.Constant) and isinstance(inner.value, (int, float))
    return False


def _negative_value(node: ast.AST) -> int | None:
    """取负常量数值 (用于消息描述)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and node.value < 0:
        return int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = node.operand
        if isinstance(inner, ast.Constant) and isinstance(inner.value, (int, float)):
            return -int(inner.value)
    return None


def _is_positive_offset(node: ast.AST) -> bool:
    """``i + 1`` / ``idx + 1`` / 字面量正数 这类"未来偏移"下界."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and node.value > 0:
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        right = node.right
        if isinstance(right, ast.Constant) and isinstance(right.value, (int, float)) and right.value > 0:
            return True
    return False


def _attr_name(node: ast.AST) -> str | None:
    """``df.iloc`` → 'iloc'; 非属性访问返回 None."""
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_volume_expr(node: ast.AST) -> bool:
    """判断表达式是否指向 volume 类列 (df['volume'] / df.volume / df['vol'].cumsum())."""
    # 穿透调用链: df["volume"].cumsum() → 根是 Subscript
    while isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        node = node.value
    if isinstance(node, ast.Subscript):
        idx = node.slice
        if isinstance(idx, ast.Constant) and isinstance(idx.value, str):
            return bool(_VOLUME_RE.search(idx.value))
        return False
    if isinstance(node, ast.Attribute):
        return bool(_VOLUME_RE.search(node.attr))
    return False


def _extract_snippet(source_lines: list[str], lineno: int) -> str:
    try:
        return source_lines[lineno - 1].strip()[:160]
    except IndexError:
        return ""


# ---------------------------------------------------------------------------
# 扫描器
# ---------------------------------------------------------------------------

def _scan_shift_calls(tree: ast.Module, source_lines: list[str], findings: list[LookaheadFinding]) -> None:
    """L1: 未来函数 shift(-k)."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _attr_name(node.func) != "shift":
            continue
        if not node.args:
            continue
        if not _is_negative_const(node.args[0]):
            continue
        k = _negative_value(node.args[0]) or 1
        message = (
            f"未来函数: shift({k}) 使用未来 {abs(k)} 根 K 线的数据, "
            f"当前决策向量会在 tick i 偷看 tick i+{abs(k)} 的信息 (偷价/look-ahead bias)."
        )
        findings.append(
            LookaheadFinding(
                rule_id="L1",
                severity="VIOLATION",
                line=node.lineno,
                col=node.col_offset,
                message=message,
                snippet=_extract_snippet(source_lines, node.lineno),
            )
        )


def _scan_future_indexing(tree: ast.Module, source_lines: list[str], findings: list[LookaheadFinding]) -> None:
    """L2: 未来行索引 iloc/loc."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        value = node.value
        if not isinstance(value, ast.Attribute):
            continue
        method = value.attr
        if method not in ("iloc", "loc", "ix"):
            continue
        sl = node.slice
        # --- 变量偏移: df.iloc[i+1] / df.loc[i+1:] → VIOLATION ---
        if isinstance(sl, ast.BinOp) and _is_positive_offset(sl):
            findings.append(
                LookaheadFinding(
                    rule_id="L2",
                    severity="VIOLATION",
                    line=node.lineno,
                    col=node.col_offset,
                    message=(
                        f"未来行索引: {method}[i + k] 以变量偏移访问未来行, "
                        "循环中会在 tick i 读取 tick i+k 的数据 (偷价)."
                    ),
                    snippet=_extract_snippet(source_lines, node.lineno),
                )
            )
            continue
        if isinstance(sl, ast.Slice):
            lower = sl.lower
            if isinstance(lower, ast.BinOp) and _is_positive_offset(lower):
                findings.append(
                    LookaheadFinding(
                        rule_id="L2",
                        severity="VIOLATION",
                        line=node.lineno,
                        col=node.col_offset,
                        message=(
                            "未来行索引: 切片下界含正偏移 (i+1 起始), 会把未来行并入当前窗口."
                        ),
                        snippet=_extract_snippet(source_lines, node.lineno),
                    )
                )
                continue
            if lower is not None and _is_positive_offset(lower):
                findings.append(
                    LookaheadFinding(
                        rule_id="L2",
                        severity="VIOLATION",
                        line=node.lineno,
                        col=node.col_offset,
                        message=f"未来行索引: {method}[n:] 从第 n 行起取数, 与当前行错位 (偷价风险).",
                        snippet=_extract_snippet(source_lines, node.lineno),
                    )
                )


def _scan_roll_negative(tree: ast.Module, source_lines: list[str], findings: list[LookaheadFinding]) -> None:
    """L3: np.roll(x, -k)."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _attr_name(node.func) != "roll":
            continue
        if len(node.args) < 2:
            continue
        shift_arg = node.args[1]
        if _is_negative_const(shift_arg):
            k = _negative_value(shift_arg) or -1
            findings.append(
                LookaheadFinding(
                    rule_id="L3",
                    severity="VIOLATION",
                    line=node.lineno,
                    col=node.col_offset,
                    message=(
                        f"np.roll 负偏移 ({k}): 将未来行数据前移进当前决策位 (偷价). "
                        "如需对齐标签请使用正偏移 + 截断."
                    ),
                    snippet=_extract_snippet(source_lines, node.lineno),
                )
            )


def _collect_rolling_stats(tree: ast.Module) -> dict[str, ast.Assign]:
    """收集滚动统计量赋值: name -> Assign 节点 (rolling/expanding/ewm ... terminal)."""
    out: dict[str, ast.Assign] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        if not isinstance(value, ast.Call):
            continue
        # 找到链条最外层终结方法: df.close.rolling(20).mean() → mean
        fn = value.func
        if not isinstance(fn, ast.Attribute):
            continue
        if fn.attr not in _ROLLING_TERMINAL:
            continue
        # 回溯链条确认源头是 rolling/expanding/ewm
        cur: ast.AST | None = fn.value
        has_rolling_source = False
        while cur is not None:
            if isinstance(cur, ast.Call):
                f = cur.func
                if isinstance(f, ast.Attribute) and f.attr in _ROLLING_SOURCES:
                    has_rolling_source = True
                    break
                cur = f.value if isinstance(f, ast.Attribute) else None
            elif isinstance(cur, ast.Attribute):
                cur = cur.value
            else:
                cur = None
        if not has_rolling_source:
            continue
        # 记录所有目标名 (单目标赋值)
        for t in node.targets:
            if isinstance(t, ast.Name):
                out[t.id] = node
            elif isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) and isinstance(t.slice.value, str):
                out[f'df["{t.slice.value}"]'] = node
    return out


def _scan_rolling_leakage(
    tree: ast.Module, source_lines: list[str], findings: list[LookaheadFinding], stats: dict[str, ast.Assign]
) -> None:
    """L4: 滚动统计量未滞后一拍即参与决策 → WARNING."""
    full_src = ast.unparse(tree) if hasattr(ast, "unparse") else ""
    for name, node in stats.items():
        # 标签列豁免 (监督标签允许用当前窗口)
        if _LABEL_NAME_RE.search(name):
            continue
        # 该统计量在别处被 shift(1) 滞后 → 合规
        if re.search(rf"{re.escape(name)}\s*\.\s*shift\s*\(", full_src):
            continue
        findings.append(
            LookaheadFinding(
                rule_id="L4",
                severity="WARNING",
                line=node.lineno,
                col=node.col_offset,
                message=(
                    f"滚动统计量 '{name}' 未滞后: 统计窗口包含当前 tick 自身, "
                    "若该列参与同 bar 决策向量即构成数据泄漏. 应在使用前 shift(1) 滞后一拍."
                ),
                snippet=_extract_snippet(source_lines, node.lineno),
            )
        )


def _scan_div_zero_volume(tree: ast.Module, source_lines: list[str], findings: list[LookaheadFinding]) -> None:
    """L5: 分母为 volume 的除法 (VWAP 类) 无守卫 → WARNING."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
            continue
        if _is_volume_expr(node.right):
            findings.append(
                LookaheadFinding(
                    rule_id="L5",
                    severity="WARNING",
                    line=node.lineno,
                    col=node.col_offset,
                    message=(
                        "除零风险: 分母为成交量列且未见零值守卫. "
                        "开盘集合竞价成交量可为 0, VWAP 类计算会崩溃/产生 NaN 信号."
                    ),
                    snippet=_extract_snippet(source_lines, node.lineno),
                )
            )


def scan_lookahead(source: str, filename: str = "<strategy>") -> dict[str, Any]:
    """对策略源码执行静态不变量扫描.

    Args:
        source:   策略 Python 源码 (需可被 ``ast.parse``).
        filename: 展示用文件名.

    Returns:
        dict::

            {
              "verdict": "pass" | "review" | "block" | "error",
              "filename": str,
              "findings": [LookaheadFinding-as-dict ...],
              "violations": [ ... ], "warnings": [ ... ],
              "summary": {"violations": int, "warnings": int, ...}
            }
    """
    source_lines = source.splitlines()
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        finding = LookaheadFinding(
            rule_id="syntax",
            severity="VIOLATION",
            line=exc.lineno or 1,
            col=exc.offset or 0,
            message=f"源码语法错误: {exc.msg}",
            snippet=_extract_snippet(source_lines, exc.lineno or 1),
        )
        return _assemble([finding], filename, source_lines)

    findings: list[LookaheadFinding] = []
    _scan_shift_calls(tree, source_lines, findings)
    _scan_future_indexing(tree, source_lines, findings)
    _scan_roll_negative(tree, source_lines, findings)
    stats = _collect_rolling_stats(tree)
    _scan_rolling_leakage(tree, source_lines, findings, stats)
    _scan_div_zero_volume(tree, source_lines, findings)

    # L1 例外: shift(-k) 赋值给标签列 → 降级 WARNING.
    # 兼容 df["target"] = ... 下标赋值 与 target = ... 裸名赋值
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            if isinstance(t, ast.Name):
                name = t.id
            elif isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) and isinstance(t.slice.value, str):
                name = t.slice.value
            else:
                name = ""
            if not name or not _LABEL_NAME_RE.search(name):
                continue
            for f in findings:
                if f.rule_id == "L1" and f.line == node.lineno:
                    f.severity = "WARNING"
                    f.message = (
                        f"未来函数 shift 仅用于构造监督标签列 '{name}' (允许, "
                        "但请确认该列绝不参与信号/决策计算)."
                    )

    return _assemble(findings, filename, source_lines)


def _assemble(findings: list[LookaheadFinding], filename: str, source_lines: list[str]) -> dict[str, Any]:
    findings.sort(key=lambda f: (f.severity != "VIOLATION", f.line, f.col))
    violations = [f for f in findings if f.severity == "VIOLATION"]
    warnings = [f for f in findings if f.severity == "WARNING"]

    if violations:
        verdict = "block"
    elif warnings:
        verdict = "review"
    else:
        verdict = "pass"

    summary = {
        "violations": len(violations),
        "warnings": len(warnings),
        "lookahead_shifts": sum(1 for f in findings if f.rule_id == "L1"),
        "future_indexes": sum(1 for f in findings if f.rule_id == "L2"),
        "roll_negative": sum(1 for f in findings if f.rule_id == "L3"),
        "leakage_risks": sum(1 for f in findings if f.rule_id == "L4"),
        "div_zero_risks": sum(1 for f in findings if f.rule_id == "L5"),
        "lines_scanned": len(source_lines),
    }

    def _to_dict(f: LookaheadFinding) -> dict[str, Any]:
        return {
            "rule_id": f.rule_id,
            "severity": f.severity,
            "line": f.line,
            "col": f.col,
            "message": f.message,
            "snippet": f.snippet,
        }

    return {
        "verdict": verdict,
        "filename": filename,
        "findings": [_to_dict(f) for f in findings],
        "violations": [_to_dict(f) for f in violations],
        "warnings": [_to_dict(f) for f in warnings],
        "summary": summary,
    }


__all__ = ["LookaheadFinding", "scan_lookahead"]
