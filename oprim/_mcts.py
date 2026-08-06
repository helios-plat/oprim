"""oprim._mcts — 树搜索(深度 > 1 的接缝)。

先说清楚: **大多数场景不需要树搜索。** 深度 1 的 lookahead(_lookahead) 已经能解决
绝大部分"三个补丁选一个"的问题。只有当动作存在真实序贯依赖(先改配置才能跑迁移,
先扩容才能压测)时, 树才值那个钱。

PUCT 而非纯 UCT:
    UCT  = W/N + c·√(lnN / Nᵢ)
    PUCT = W/N + c·P(a)·√N / (1 + Nᵢ)
纯 UCT 假设你能负担把每个动作都试几次; Agent 场景里每次扩展是一次 LLM 调用,
试不起。PUCT 让 LLM 先验直接参与选择, 这是从"1000 次模拟"降到"16 次模拟"的关键。

另两个省钱手段: 渐进加宽(子节点上限 ⌈k·N^α⌉) + 置换表(状态指纹相同则复用节点)。
奖励必须归一到 [0,1], 否则 c 的取值失去意义(见 _reward)。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Optional, Protocol, Sequence, Tuple


class WorldModel(Protocol):
    """领域接缝。sandbox 版 step() = checkout+apply, reward() = run_probes。"""
    def key(self, state: Any) -> Hashable: ...
    def actions(self, state: Any) -> Sequence[Tuple[Any, float]]: ...   # (动作, 先验)
    def step(self, state: Any, action: Any) -> Any: ...
    def terminal(self, state: Any) -> bool: ...
    def reward(self, state: Any) -> float: ...                          # ∈ [0,1]


@dataclass
class Node:
    key: Hashable
    prior: float = 1.0
    N: int = 0
    W: float = 0.0
    children: Dict[Any, "Node"] = field(default_factory=dict)
    untried: List[Tuple[Any, float]] = field(default_factory=list)
    terminal: bool = False

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N else 0.0

    def allowed_children(self, k: float, alpha: float) -> int:
        return max(1, math.ceil(k * (self.N ** alpha)))


def puct(child: Node, parent_N: int, c: float) -> float:
    return child.Q + c * child.prior * math.sqrt(parent_N) / (1 + child.N)


@dataclass
class SearchStats:
    simulations: int = 0
    expansions: int = 0
    rollouts: int = 0
    transposition_hits: int = 0


class MCTS:
    def __init__(self, model: WorldModel, c_puct: float = 1.4,
                 widening_k: float = 2.0, widening_alpha: float = 0.5,
                 max_depth: int = 4, seed: int = 0):
        self.m = model
        self.c = c_puct
        self.k = widening_k
        self.alpha = widening_alpha
        self.max_depth = max_depth
        self.rng = random.Random(seed)
        self.table: Dict[Hashable, Node] = {}      # 置换表
        self.stats = SearchStats()

    def _node(self, state) -> Node:
        key = self.m.key(state)
        if key in self.table:
            self.stats.transposition_hits += 1
            return self.table[key]
        n = Node(key=key, terminal=self.m.terminal(state))
        n.untried = list(self.m.actions(state)) if not n.terminal else []
        # 先验降序 = 优先扩展 LLM 看好的分支; 同分按稳定顺序, 保证可复现
        n.untried.sort(key=lambda ap: (-ap[1], repr(ap[0])))
        self.table[key] = n
        return n

    def search(self, root_state, budget: int = 16) -> Optional[Any]:
        root = self._node(root_state)
        for _ in range(budget):
            self._simulate(root_state, root, 0)
            self.stats.simulations += 1
        if not root.children:
            return None
        # 按访问次数选(robust child), 不是按 Q —— 访问多的证据更足
        best = max(root.children.items(), key=lambda kv: (kv[1].N, kv[1].Q, repr(kv[0])))
        return best[0]

    def _simulate(self, state, node: Node, depth: int) -> float:
        if node.terminal or depth >= self.max_depth:
            r = self.m.reward(state)
            self.stats.rollouts += 1
            self._backprop(node, r)
            return r

        if node.untried and len(node.children) < node.allowed_children(self.k, self.alpha):
            action, prior = node.untried.pop(0)
            child_state = self.m.step(state, action)
            child = self._node(child_state)
            child.prior = prior
            node.children[action] = child
            self.stats.expansions += 1
            r = self.m.reward(child_state)
            self.stats.rollouts += 1
            self._backprop(child, r)
            self._backprop(node, r)
            return r

        if not node.children:
            r = self.m.reward(state)
            self.stats.rollouts += 1
            self._backprop(node, r)
            return r

        action, child = max(node.children.items(),
                            key=lambda kv: (puct(kv[1], max(1, node.N), self.c), repr(kv[0])))
        r = self._simulate(self.m.step(state, action), child, depth + 1)
        self._backprop(node, r)
        return r

    @staticmethod
    def _backprop(node: Node, r: float) -> None:
        node.N += 1
        node.W += r


def best_path(mcts: MCTS, root_state, max_len: int = 8) -> List[Any]:
    """搜索完毕后抽出主变着(principal variation)。"""
    path, state = [], root_state
    node = mcts.table.get(mcts.m.key(state))
    while node and node.children and len(path) < max_len:
        action, node = max(node.children.items(), key=lambda kv: (kv[1].N, repr(kv[0])))
        path.append(action)
        state = mcts.m.step(state, action)
    return path


__all__ = ["MCTS", "Node", "SearchStats", "WorldModel", "best_path", "puct"]
