"""oprim._games — 离线博弈论工具, **不在请求路径上**。

目的有二:
1. 给"纳什均衡 ≠ 帕累托最优"提供可运行反例: 囚徒困境的唯一纯策略纳什均衡
   是严格帕累托劣的。所以"算个均衡来保证全局帕累托最优"在数学上不通,
   分配该走 _allocate 的组合最优化。
2. 真正用得上博弈论的场合: **离线验证机制**(如"如实报价是否占优策略")。

只实现两人有限博弈的纯策略部分 —— 混合策略均衡的输出是概率分布,
执行时要掷骰子, 与确定性目标冲突, 刻意不提供。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


def _np():
    import numpy  # noqa: PLC0415
    return numpy


@dataclass
class Game:
    """两人有限博弈。A[i,j] = 行玩家收益, B[i,j] = 列玩家收益。收益越大越好。"""
    A: "object"
    B: "object"
    row_labels: Sequence[str] = ()
    col_labels: Sequence[str] = ()

    def __post_init__(self):
        np = _np()
        self.A = np.asarray(self.A, dtype=float)
        self.B = np.asarray(self.B, dtype=float)
        if self.A.shape != self.B.shape:
            raise ValueError("A 与 B 形状必须一致")
        if not self.row_labels:
            self.row_labels = [f"r{i}" for i in range(self.A.shape[0])]
        if not self.col_labels:
            self.col_labels = [f"c{j}" for j in range(self.A.shape[1])]

    def label(self, i: int, j: int) -> str:
        return f"({self.row_labels[i]}, {self.col_labels[j]})"


def pure_nash(g: Game) -> List[Tuple[int, int]]:
    """所有纯策略纳什均衡: 双方都无法单方面偏离获益的格子。"""
    out = []
    n, m = g.A.shape
    for i in range(n):
        for j in range(m):
            if g.A[i, j] >= g.A[:, j].max() - 1e-12 and \
               g.B[i, j] >= g.B[i, :].max() - 1e-12:
                out.append((i, j))
    return out


def pareto_optimal(g: Game) -> List[Tuple[int, int]]:
    """帕累托前沿: 不存在另一格让一方更好且另一方不更差。"""
    n, m = g.A.shape
    cells = [(i, j) for i in range(n) for j in range(m)]
    out = []
    for (i, j) in cells:
        dominated = any(
            (g.A[k, l] >= g.A[i, j] and g.B[k, l] >= g.B[i, j]) and
            (g.A[k, l] > g.A[i, j] + 1e-12 or g.B[k, l] > g.B[i, j] + 1e-12)
            for (k, l) in cells)
        if not dominated:
            out.append((i, j))
    return out


def dominant_strategies(g: Game) -> Tuple[List[int], List[int]]:
    """严格占优策略(行玩家, 列玩家)。VCG 下"如实报价"就该出现在这里。"""
    n, m = g.A.shape
    rows = [i for i in range(n)
            if all(g.A[i, j] >= g.A[k, j] - 1e-12 for j in range(m) for k in range(n))]
    cols = [j for j in range(m)
            if all(g.B[i, j] >= g.B[i, l] - 1e-12 for i in range(n) for l in range(m))]
    return rows, cols


def prisoners_dilemma() -> Game:
    """经典囚徒困境(收益越大越好)。用来证伪"均衡即最优"。"""
    return Game(A=[[-1, -3], [0, -2]], B=[[-1, 0], [-3, -2]],
                row_labels=["合作", "背叛"], col_labels=["合作", "背叛"])


def nash_vs_pareto_report(g: Game) -> str:
    ne = pure_nash(g)
    po = pareto_optimal(g)
    lines = [f"纯策略纳什均衡: {[g.label(i, j) for i, j in ne] or '（无）'}",
             f"帕累托前沿:     {[g.label(i, j) for i, j in po]}"]
    bad = [c for c in ne if c not in po]
    if bad:
        lines.append(f"→ 均衡 {[g.label(i, j) for i, j in bad]} 不在帕累托前沿上。"
                     f"「算个纳什均衡来保证全局帕累托最优」在数学上不成立。")
    else:
        lines.append("→ 本例中均衡恰好落在前沿上(这是特例, 不是通例)。")
    return "\n".join(lines)


__all__ = ["Game", "dominant_strategies", "nash_vs_pareto_report",
           "pareto_optimal", "prisoners_dilemma", "pure_nash"]
