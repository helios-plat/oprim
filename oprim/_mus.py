"""oprim._mus — MUS (Minimal Unsatisfiable Subset) 收缩。

刻意与 z3 解耦: 只依赖 ``oracle(subset) -> "sat"|"unsat"|"unknown"``,
收缩算法本身可用纯 Python mock 单测, 不必装求解器。

为什么必须收缩: Z3 的 ``unsat_core()`` 不保证最小, 实测常给 8 条里的 6 条
而真核只有 3 条。直接甩给 LLM = 让它在噪声里找信号。

三态处理: ``unknown``(超时) 时**不能**丢掉候选约束 —— 那会产生一个假的、
其实可满足的"矛盾核心"。此时保留该约束并把整个 MUS 标记 verified=False。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Set

Oracle = Callable[[Sequence[str]], str]   # 返回 "sat" / "unsat" / "unknown"


@dataclass
class MusResult:
    mus: List[str]
    verified: bool = True
    checks: int = 0
    dropped: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def shrink_to_mus(oracle: Oracle,
                  core: Sequence[str],
                  protected: Set[str] = frozenset(),
                  max_checks: int = 200) -> MusResult:
    """删除法收缩。

    protected: 不参与归因的约束 id(安全不变量/公理)。它们始终在集合内,
               但不会被列进 MUS —— 你不希望 LLM 反思"要不把安全约束删了吧"。
    """
    # 排序 = 确定性。同一份输入必须给出同一个 MUS, 否则 plan_id 复现不了。
    candidates = sorted(core)
    current = list(candidates)
    res = MusResult(mus=current, checks=0)

    for cid in candidates:
        if cid not in current:
            continue                       # 已被之前某轮返回的更小核带走
        if cid in protected:
            continue
        if res.checks >= max_checks:
            res.notes.append(f"达到 check 预算上限 {max_checks}, 核可能非最小")
            res.verified = False
            break

        trial = [x for x in current if x != cid]
        verdict = oracle(trial)
        res.checks += 1

        if verdict == "unsat":
            current = trial                # cid 无关, 剔除
            res.dropped.append(cid)
        elif verdict == "unknown":
            res.verified = False
            res.notes.append(f"检查 {cid!r} 时求解器返回 unknown(超时), 保守保留")
        # "sat" → cid 是必需的, 留下

    res.mus = sorted(current)
    return res


def explain(mus: Sequence[str],
            intents: Dict[str, str],
            origins: Dict[str, str]) -> List[Dict[str, str]]:
    """把 MUS 翻回自然语言 —— 喂给 LLM 的东西, 永远不要直接甩 `c_17`。"""
    return [{"id": cid,
             "intent": intents.get(cid, "(缺少 intent —— 校验器应该已经拦下)"),
             "origin": origins.get(cid, "")}
            for cid in mus]


__all__ = ["MusResult", "Oracle", "explain", "shrink_to_mus"]
