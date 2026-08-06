"""oprim._structural_counterfactual — 显式外生噪声 SCM + 三步法 (L3 反事实)。

把二进制故障网络建模为带显式噪声节点的结构因果模型 (SCM):

    每个节点 X 的机制:  X_fault = (∨_{p∈parents} p_fault ∧ t_{X,p}) ∨ U_X
    U_X ~ Bernoulli(b_X)   — 外生噪声 (漏损/基础故障率)

参数从 noisy-OR CPD 表精确恢复:
    b_X = P(fault | 所有父节点 ok)
    t_{X,p} = 1 − (1 − P(fault | 仅 p fault)) / (1 − b_X)

三步法 (Pearl L3):
    1. Abduction   用事实证据 e 计算 P(U | e) — 均值场/边际 MAP 近似
                   (逐个 U 变量, 其余夹持在 MAP, 检查确定性传播与证据的相容性)
    2. Action      对候选组件 do(X=ok): 割裂入边, 机制改为确定性干预
    3. Prediction  在干预后模型上夹持溯因 U, 拓扑传播 (OR 网络 O(V+E)):
                   P(Y | do(X=ok), U ~ P(U|e))

同时给出三层对照:
    事实   P(Y | e)                    — 本次观测到的事实
    L2     P(Y | do(X=ok))             — 不锚定本次 U (平均情形, 对"下一次"的指导)
    L3     P(Y | do(X=ok), U~P(U|e))   — 锚定本次噪声 (对"这一次若当时…"的答案)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import networkx as nx


@dataclass
class SCMNode:
    """单个节点的噪声机制参数。"""

    name: str
    base: float                            # b_X = P(U_X = 1) = 漏损/基础故障率
    transmissions: dict[str, float] = field(default_factory=dict)  # parent -> t
    parents: list[str] = field(default_factory=list)

    def transmission_to(self, parent: str) -> float:
        return self.transmissions.get(parent, 0.0)


class StructuralSCM:
    """显式外生噪声 SCM (二进制故障网络)。"""

    def __init__(self, nodes: dict[str, SCMNode], graph: nx.DiGraph):
        self.nodes = nodes
        self.graph = graph

    # ── 构造 ────────────────────────────────────────────────────────
    @classmethod
    def from_graph(cls, dag: nx.DiGraph,
                   cpd_map: dict[str, Any] | None = None) -> StructuralSCM:
        """从因果 DAG (+ noisy-OR CPD 表) 构建显式噪声 SCM。

        cpd_map 缺省时用 p_fail/cond_fail 节点属性: 根节点 b=p_fail,
        非根节点 b=0.05 且 t 由 cond_fail 表恢复 (见 _fit_node_params)。
        """
        nodes: dict[str, SCMNode] = {}
        for name in dag.nodes:
            parents = list(dag.predecessors(name))
            cpd = (cpd_map or {}).get(name)
            if cpd is not None:
                base, trans = _recover_noisy_or(name, parents, cpd)
            else:
                attrs = dag.nodes[name]
                base = float(attrs.get("p_fail", 0.05))
                trans = _transmissions_from_cond_fail(parents, attrs.get("cond_fail"))
            nodes[name] = SCMNode(name=name, base=base, transmissions=trans,
                                  parents=parents)
        return cls(nodes, dag)

    # ── 确定性传播 (给定 U 赋值, 全网络状态唯一) ─────────────────────
    def fault_state(self, u_assignment: dict[str, bool]) -> dict[str, bool]:
        """按拓扑序确定性传播: X_fault = (∨ 父故障∧t) ∨ U_X。"""
        state: dict[str, bool] = {}
        for name in nx.topological_sort(self.graph):
            u = bool(u_assignment.get(name, False))
            parent_hit = any(
                state.get(p, False) and self.nodes[name].transmission_to(p) > 0
                for p in self.nodes[name].parents
            )
            state[name] = u or parent_hit
        return state

    def consistent(self, u_assignment: dict[str, bool],
                   evidence: dict[str, str]) -> bool:
        """给定 U 赋值, 传播出的状态是否与证据相容 (证据值: 'ok'/'fault')。"""
        state = self.fault_state(u_assignment)
        for node, value in evidence.items():
            want_fault = str(value).lower() in ("fault", "1", "true", "fail")
            if state.get(node, False) != want_fault:
                return False
        return True

    # ── 1. Abduction: P(U | e) 均值场/边际 MAP ───────────────────────
    def abduct(self, evidence: dict[str, str], *, sweeps: int = 5) -> dict[str, float]:
        """均值场近似: 逐个 U 变量计算 P(U_X=1 | e, U_其他=MAP)。

        其余 U 夹持在当前 MAP (后验 ≥ 0.5), 用确定性传播检查相容性:
          P(U_X=1|e) ∝ P(e|U_X=1, U_rest=MAP)·P(U_X=1) — 相容为 1, 否则 0。
        两者都相容 → 保持先验 (证据对该噪声无信息); 都不相容 → 保持先验并记 note。
        """
        u_posterior: dict[str, float] = {n: nd.base for n, nd in self.nodes.items()}
        order = list(nx.topological_sort(self.graph))
        for _ in range(sweeps):
            for name in order:
                clamped = {n: (u_posterior[n] >= 0.5) for n in self.nodes if n != name}
                ok1 = self.consistent({**clamped, name: True}, evidence)
                ok0 = self.consistent({**clamped, name: False}, evidence)
                prior = self.nodes[name].base
                if ok1 and not ok0:
                    u_posterior[name] = 1.0
                elif ok0 and not ok1:
                    u_posterior[name] = 0.0
                elif ok1 and ok0:
                    u_posterior[name] = prior      # 证据对该噪声无信息
                # else: 都不相容 (其余 U 夹持不合理) → 保持当前值
        return u_posterior

    # ── 概率传播 (OR 网络, O(V+E)) ───────────────────────────────────
    def propagate(self, u_priors: dict[str, float],
                  intervened: Sequence[str] = ()) -> dict[str, float]:
        """拓扑传播: P(X_fault) = 1 − (1−P(U_X))·Π_p(1 − t_{Xp}·P(p_fault))。

        intervened 中的节点被 do 为确定性 ok (P=0, 且其 U 不再生效)。
        """
        intervened = set(intervened)
        p_fault: dict[str, float] = {}
        for name in nx.topological_sort(self.graph):
            if name in intervened:
                p_fault[name] = 0.0
                continue
            u = float(u_priors.get(name, self.nodes[name].base))
            survive = 1.0 - u
            for p in self.nodes[name].parents:
                t = self.nodes[name].transmission_to(p)
                survive *= 1.0 - t * p_fault.get(p, 0.0)
            p_fault[name] = max(0.0, min(1.0, 1.0 - survive))
        return p_fault

    # ── L2 / L3 干预概率 ─────────────────────────────────────────────
    def l2_p_fault(self, intervened: Sequence[str], failure_node: str) -> float:
        """P(Y | do(X=ok)) — 不锚定本次 U (U 取先验 b)。"""
        priors = {n: nd.base for n, nd in self.nodes.items()}
        return self.propagate(priors, intervened=intervened).get(failure_node, 0.0)

    def l3_p_fault(self, intervened: Sequence[str], failure_node: str,
                   u_posterior: dict[str, float]) -> float:
        """P(Y | do(X=ok), U ~ P(U|e)) — 锚定本次溯因噪声。"""
        return self.propagate(u_posterior, intervened=intervened).get(failure_node, 0.0)


# ---------------------------------------------------------------------------
# noisy-OR 参数恢复
# ---------------------------------------------------------------------------

def _recover_noisy_or(name: str, parents: list[str], cpd: Any) -> tuple[float, dict[str, float]]:
    """从 pgmpy TabularCPD 表恢复 (b, t)。

    b = P(fault | 全部父 ok);  t_i = 1 − (1 − P(fault | 仅 i fault)) / (1 − b)。
    表若与 noisy-OR 不一致 (恢复出负值) → 截断到 [0,1]。
    支持 1-D 根表 (无父节点) 与多维父表 (last axis = 最后一个父)。
    """
    try:
        values = cpd.values
        flat = values.reshape(-1) if hasattr(values, "reshape") else list(values)
        states = getattr(cpd, "state_names", None) or {}
        variable = str(getattr(cpd, "variable", name))
        fault_idx = _fault_index(states.get(variable, []))
        if fault_idx is None:
            return 0.05, {}

        parent_states = [states.get(p, ["ok", "fault"]) for p in parents]
        parent_fault_idx = [0 if len(s) < 2 else 1 for s in parent_states]
        card = [max(2, len(s)) for s in parent_states]
        ncols = 1
        for c in card:
            ncols *= c

        def p_fault(parent_bits: Sequence[int]) -> float:
            col = 0
            stride = 1
            for j in reversed(range(len(parents))):
                col += parent_bits[j] * stride
                stride *= card[j]
            return float(flat[fault_idx * ncols + col])

        all_ok = [0] * len(parents)
        base = p_fault(all_ok)
        trans: dict[str, float] = {}
        denom = 1.0 - base
        for j, p in enumerate(parents):
            bits = [0] * len(parents)
            bits[j] = parent_fault_idx[j]
            t = 0.0 if denom <= 1e-12 else 1.0 - (1.0 - p_fault(bits)) / denom
            trans[p] = max(0.0, min(1.0, t))
        return max(0.0, min(1.0, base)), trans
    except Exception:
        return 0.05, {}


def _fault_index(states: Sequence[Any]) -> int | None:
    for i, s in enumerate(states):
        if str(s).lower() in ("fault", "fail", "1", "true"):
            return i
    return None


def _transmissions_from_cond_fail(parents: list[str],
                                  cond_fail: Any) -> dict[str, float]:
    """节点属性 cond_fail 兜底: {父状态组合: 概率} → 逐父 t (近似)。"""
    if not cond_fail or not parents:
        return {}
    trans: dict[str, float] = {}
    try:
        items = cond_fail.items() if isinstance(cond_fail, dict) else []
        for combo, prob in items:
            keys = list(combo) if isinstance(combo, (tuple, list)) else [combo]
            for j, p in enumerate(parents):
                if j < len(keys) and str(keys[j]).lower() in ("fault", "1", "true"):
                    trans[p] = max(trans.get(p, 0.0), float(prob))
    except Exception:
        return {}
    return trans


__all__ = ["SCMNode", "StructuralSCM"]
