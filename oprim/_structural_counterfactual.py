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
from itertools import product
from typing import Any

import networkx as nx


@dataclass
class SCMNode:
    """单个节点的噪声机制参数。"""

    name: str
    base: float  # b_X = P(U_X = 1) = 漏损/基础故障率
    transmissions: dict[str, float] = field(default_factory=dict)  # parent -> t
    parents: list[str] = field(default_factory=list)

    def transmission_to(self, parent: str) -> float:
        return self.transmissions.get(parent, 0.0)


class StructuralSCM:
    """显式外生噪声 SCM (二进制故障网络)。

    推断口径 (``INFERENCE``):
      - 默认**精确** twin-network 枚举: ``abduct`` / ``l2_p_fault`` /
        ``l3_p_fault(evidence=...)`` 联合枚举全部外生噪声 (leak + 边激活),
        无父独立性假设, reconvergent DAG 上亦精确, 正确捕捉 explaining-away。
      - 自由外生变量 > ``max_vars`` (默认 20, 即 2^20 世界) 时**回落**均值场:
        ``propagate`` 乘积式 (仅 polytree 精确) + ``abduct`` MAP sweeps。
      - ``l3_p_fault`` 不传 evidence (只给 u_posterior) 时走旧的均值场口径 (向后兼容)。
    """

    #: 推断口径标记 —— 精确枚举优先, 超 max_vars 回落均值场。
    INFERENCE: str = "exact_enum_with_meanfield_fallback"

    def __init__(self, nodes: dict[str, SCMNode], graph: nx.DiGraph):
        self.nodes = nodes
        self.graph = graph

    # ── 构造 ────────────────────────────────────────────────────────
    @classmethod
    def from_graph(cls, dag: nx.DiGraph, cpd_map: dict[str, Any] | None = None) -> StructuralSCM:
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
            nodes[name] = SCMNode(name=name, base=base, transmissions=trans, parents=parents)
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

    def consistent(self, u_assignment: dict[str, bool], evidence: dict[str, str]) -> bool:
        """给定 U 赋值, 传播出的状态是否与证据相容 (证据值: 'ok'/'fault')。"""
        state = self.fault_state(u_assignment)
        for node, value in evidence.items():
            want_fault = str(value).lower() in ("fault", "1", "true", "fail")
            if state.get(node, False) != want_fault:
                return False
        return True

    # ── 精确外生噪声枚举 (twin-network, 无父独立性假设) ────────────────
    def _exogenous(self):
        """自由外生噪声: leak L_X~Bern(b_X) (0<b<1) + 边激活 A_{X,p}~Bern(t) (0<t<1)。

        b/t ∈ {0,1} 退化为确定性, 不进枚举 (省 2 的幂次)。
        返回 (拓扑序, 自由leak[(node,b)], 固定leak{node:bool},
              自由边[((child,parent),t)], 固定边{(child,parent):bool})。
        """
        order = list(nx.topological_sort(self.graph))
        lf: list[tuple[str, float]] = []
        lfx: dict[str, bool] = {}
        for n in order:
            b = self.nodes[n].base
            if b <= 0.0:
                lfx[n] = False
            elif b >= 1.0:
                lfx[n] = True
            else:
                lf.append((n, b))
        ef: list[tuple[tuple[str, str], float]] = []
        efx: dict[tuple[str, str], bool] = {}
        for name in order:
            for p in self.nodes[name].parents:
                t = self.nodes[name].transmission_to(p)
                if t <= 0.0:
                    continue
                key = (name, p)
                if t >= 1.0:
                    efx[key] = True
                else:
                    ef.append((key, t))
        return order, lf, lfx, ef, efx

    def _state_from_noise(self, order, leak_bits, edge_bits, intervened) -> dict[str, bool]:
        """给定 leak/edge 噪声赋值确定性传播全网状态; intervened 节点 do 为 ok。"""
        iv = set(intervened)
        state: dict[str, bool] = {}
        for name in order:
            if name in iv:
                state[name] = False
                continue
            hit = any(
                state.get(p, False) and edge_bits.get((name, p), False)
                for p in self.nodes[name].parents
            )
            state[name] = bool(leak_bits.get(name, False)) or hit
        return state

    @staticmethod
    def _matches(state: dict[str, bool], evidence: dict[str, str]) -> bool:
        for node, value in evidence.items():
            want = str(value).lower() in ("fault", "1", "true", "fail")
            if state.get(node, False) != want:
                return False
        return True

    def _iter_worlds(self, order, lf, lfx, ef, efx):
        """枚举全部自由外生赋值, yield (联合先验权重, leak_bits, edge_bits)。"""
        free = [("L", k, pr) for k, pr in lf] + [("E", k, pr) for k, pr in ef]
        for bits in product((0, 1), repeat=len(free)):
            w = 1.0
            leak_bits = dict(lfx)
            edge_bits = dict(efx)
            for i, (kind, key, pr) in enumerate(free):
                on = bits[i]
                w *= pr if on else (1.0 - pr)
                (leak_bits if kind == "L" else edge_bits)[key] = bool(on)
            yield w, leak_bits, edge_bits

    def exact_intervene(
        self, evidence, intervened, failure_node, *, max_vars: int = 20
    ) -> float | None:
        """精确 twin-network 反事实: P(Y=fault | do(intervened=ok), evidence)。

        同一份外生噪声在事实世界 (无干预) 条件于 evidence, 反事实世界施加 do ——
        无父独立性假设, reconvergent DAG 上亦精确。
        自由外生变量 > max_vars → None (回落均值场); evidence 概率为 0 → None。
        """
        order, lf, lfx, ef, efx = self._exogenous()
        if len(lf) + len(ef) > max_vars:
            return None
        num = den = 0.0
        for w, lb, eb in self._iter_worlds(order, lf, lfx, ef, efx):
            if w == 0.0:
                continue
            if not self._matches(self._state_from_noise(order, lb, eb, ()), evidence):
                continue
            den += w
            if self._state_from_noise(order, lb, eb, intervened).get(failure_node, False):
                num += w
        return None if den <= 0.0 else num / den

    def _abduct_exact(self, evidence, *, max_vars: int = 20) -> dict[str, float] | None:
        """精确边际溯因 P(L_X=1 | evidence) (枚举); 过大或证据不可能 → None。"""
        order, lf, lfx, ef, efx = self._exogenous()
        if len(lf) + len(ef) > max_vars:
            return None
        node_num = {n: 0.0 for n in self.nodes}
        den = 0.0
        for w, lb, eb in self._iter_worlds(order, lf, lfx, ef, efx):
            if w == 0.0:
                continue
            if not self._matches(self._state_from_noise(order, lb, eb, ()), evidence):
                continue
            den += w
            for n in self.nodes:
                if lb.get(n, False):
                    node_num[n] += w
        return None if den <= 0.0 else {n: node_num[n] / den for n in self.nodes}

    # ── 1. Abduction: P(L | e) 精确枚举 (过大回落均值场 MAP) ───────────
    def abduct(
        self, evidence: dict[str, str], *, sweeps: int = 5, max_vars: int = 20
    ) -> dict[str, float]:
        """精确边际溯因 P(L_X=1 | e); 自由外生 > max_vars 时回落均值场 sweeps 近似。"""
        exact = self._abduct_exact(evidence, max_vars=max_vars)
        if exact is not None:
            return exact
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
                    u_posterior[name] = prior  # 证据对该噪声无信息
                # else: 都不相容 (其余 U 夹持不合理) → 保持当前值
        return u_posterior

    # ── 概率传播 (OR 网络, O(V+E)) ───────────────────────────────────
    def propagate(
        self, u_priors: dict[str, float], intervened: Sequence[str] = ()
    ) -> dict[str, float]:
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

    # ── L2 / L3 干预概率 (精确枚举优先, 过大回落均值场乘积式) ──────────
    def l2_p_fault(
        self, intervened: Sequence[str], failure_node: str, *, max_vars: int = 20
    ) -> float:
        """P(Y | do(X=ok)) — 平均情形 (不锚定本次噪声)。精确边际干预, 过大回落乘积式。"""
        exact = self.exact_intervene({}, intervened, failure_node, max_vars=max_vars)
        if exact is not None:
            return exact
        priors = {n: nd.base for n, nd in self.nodes.items()}
        return self.propagate(priors, intervened=intervened).get(failure_node, 0.0)

    def l3_p_fault(
        self,
        intervened: Sequence[str],
        failure_node: str,
        u_posterior: dict[str, float] | None = None,
        *,
        evidence: dict[str, str] | None = None,
        max_vars: int = 20,
    ) -> float:
        """P(Y | do(X=ok), 锚定本次噪声)。

        evidence 提供 → 精确 twin-network 反事实 (推荐, 过大回落);
        否则用 u_posterior 走均值场乘积式传播 (旧口径, 向后兼容)。
        """
        if evidence is not None:
            exact = self.exact_intervene(evidence, intervened, failure_node, max_vars=max_vars)
            if exact is not None:
                return exact
        if u_posterior is None:
            u_posterior = {n: nd.base for n, nd in self.nodes.items()}
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


def _transmissions_from_cond_fail(parents: list[str], cond_fail: Any) -> dict[str, float]:
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
