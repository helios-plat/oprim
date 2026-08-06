"""Finite-horizon counterfactual rollout planner (Phase 4).

Solves
    π* = argmax_{π ∈ Π_{1:H}} E[ Σ_{t=0}^{H-1} γ^t (ΔP_t − λ C(a_t)) ]
over intervention sequences of depth ≤ H.

Key insight: do-interventions *fix* a node's state, so the joint effect of a
set of interventions is order-independent — but the *discounted* utility is
order-sensitive (marginal ΔP is submodular, so fixing high-impact nodes first
earns their ΔP earlier and at higher discount weight). We therefore search the
ordered action space with:
  - exact enumeration when the action space is small,
  - beam pruning (top-K states per horizon level) otherwise,
  - min-effective-Δ filtering (skip actions whose marginal gain is negligible),
  - cost-sensitivity (per-node repair cost, λ trade-off),
  - value-of-information bonus (uncertainty-weighted explore bonus),
  - an optional first-step "observe" action (observe-first strategies).

Quantitative evaluation uses the Phase 2 machinery: each state's P(fault) is
computed exactly with pgmpy VariableElimination on a mutilated Bayesian
network (deterministic delta CPD for every intervened node), memoized per
intervened-set.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

try:
    from pgmpy.inference import VariableElimination
    from pgmpy.models import DiscreteBayesianNetwork

    _HAS_PGMPY = True
except ImportError:  # pragma: no cover - exercised only in minimal envs
    _HAS_PGMPY = False
    DiscreteBayesianNetwork = None  # type: ignore
    VariableElimination = None  # type: ignore

from oprim._do_calculus_intervention import (
    _extract_card_and_states,
    _extract_p_fault,
    _make_deterministic_cpd,
)

DEFAULT_ACTION_COST = 0.05
OBSERVE_ACTION = "observe"


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class RolloutAction:
    """One planned step of the counterfactual rollout."""

    node: str
    step: int
    action_type: str = "intervene"  # "intervene" | "observe"
    delta_p: float = 0.0
    cost: float = 0.0
    p_fault_before: float = 0.0
    p_fault_after: float = 0.0
    utility_step: float = 0.0


@dataclass
class RolloutPlan:
    """Finite-horizon intervention plan."""

    status: str  # "ok" | "structural_only" | "empty_graph" | "no_actions"
    planned_actions: list[RolloutAction] = field(default_factory=list)
    total_utility: float = 0.0
    p_fault_baseline: float = 0.0
    p_fault_after_plan: float = 0.0
    total_cost: float = 0.0
    explored_states: int = 0
    search_backend: str = "structural"
    horizon: int = 0
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# State evaluator (memoized P(fault | do(S = ok)))
# ---------------------------------------------------------------------------

class _StateEvaluator:
    """P(failure_node = fault | do(every node in S = ok)) with per-set memo."""

    def __init__(
        self,
        dag: nx.DiGraph,
        cpd_map: dict[str, Any] | None,
        failure_node: str,
        intervention_value: Any = "ok",
        use_cache: bool = True,
    ) -> None:
        self.dag = dag
        self.cpd_map = cpd_map
        self.failure_node = failure_node
        self.intervention_value = intervention_value
        self._memo: dict[frozenset, float | None] = {}
        # 跨调用进程级 LRU (深度 1 结果复用): 同图同版本 → 二次规划直接命中
        self._use_cache = use_cache
        self._fp = ""
        if use_cache:
            try:
                from oprim._inference_cache import (  # noqa: PLC0415
                    get_intervention_cache,
                    graph_fingerprint,
                )

                self._cache = get_intervention_cache()
                self._fp = graph_fingerprint(dag, cpd_map)
            except Exception:
                self._cache = None

    def p_fault(self, intervened: frozenset[str]) -> float | None:
        if intervened in self._memo:
            return self._memo[intervened]
        if self._use_cache and self._cache is not None and len(intervened) == 1:
            # 深度 1 结果跨调用复用 (键: 图指纹 + 单元素干预集)
            key = ("p_fault_d1", self._fp, tuple(sorted(intervened)))
            hit = self._cache.get(key)
            if hit is not None:
                self._memo[intervened] = hit
                return hit
        if not self.cpd_map or not _HAS_PGMPY:
            self._memo[intervened] = None
            return None
        model = self._build_model(intervened)
        try:
            infer = VariableElimination(model)
            q = infer.query(variables=[self.failure_node], show_progress=False)
            states = (
                list(q.state_names[self.failure_node])
                if self.failure_node in q.state_names
                else list(range(q.cardinality[0]))
            )
            probs = q.values.ravel()
            dist = {str(states[i]): float(probs[i]) for i in range(len(probs))}
            p = _extract_p_fault(dist)
        except Exception:
            p = None
        self._memo[intervened] = p
        if self._use_cache and self._cache is not None and len(intervened) == 1 and p is not None:
            self._cache.set(key, p)
        return p

    def _build_model(self, intervened: frozenset[str]):
        # Graph mutilation: 被干预节点的所有入边必须切断 (do-calculus 核心)
        edges = [(u, v) for u, v in self.dag.edges() if v not in intervened]
        model = DiscreteBayesianNetwork(edges)
        for n in self.dag.nodes:
            if n not in model.nodes():
                model.add_node(n)
        for node, cpd in self.cpd_map.items():
            if node in intervened:
                card, states = _extract_card_and_states(cpd)
                det = _make_deterministic_cpd(
                    node, self.intervention_value, cardinality=card, state_names=states
                )
                model.add_cpds(det)
            else:
                model.add_cpds(cpd)
        return model


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

def counterfactual_rollout(
    causal_dag: nx.DiGraph,
    *,
    failure_node: str,
    candidate_nodes: Sequence[str] | None = None,
    horizon: int = 3,
    gamma: float = 0.95,
    cost_lambda: float = 0.05,
    cpd_map: dict[str, Any] | None = None,
    action_cost: dict[str, float] | None = None,
    default_action_cost: float = DEFAULT_ACTION_COST,
    min_effective_delta: float = 0.02,
    beam_width: int = 8,
    intervention_value: Any = "ok",
    uncertainty: dict[str, float] | None = None,
    explore_bonus: float = 0.0,
    allow_observe: bool = False,
    observe_cost: float = 0.01,
    use_cache: bool = True,
    approx_second_step: bool = True,
) -> RolloutPlan:
    """
    Finite-horizon counterfactual rollout over intervention sequences.

    Parameters
    ----------
    causal_dag : nx.DiGraph — cause → effect DAG.
    failure_node : str — the outcome whose P(fault) we minimise.
    candidate_nodes : optional — nodes allowed as actions (default: all nodes
                      that can influence failure_node, excluding it).
    horizon : int — max sequence depth H (≤ 3 recommended).
    gamma : float — discount factor ∈ (0, 1].
    cost_lambda : float — cost sensitivity λ.
    cpd_map : optional — TabularCPD map; without it the planner degrades to
              structural (path-frequency) ordering.
    action_cost : optional — per-node repair cost; default_action_cost used
                  for nodes without an entry.
    min_effective_delta : float — actions whose marginal ΔP is below this are
                          pruned (minimum-effective-Δ filter).
    beam_width : int — number of states kept per horizon level.
    intervention_value : the healthy state forced by each action.
    uncertainty : optional {node: 0..1} CPD uncertainty map — used by the
                  value-of-information explore bonus.
    explore_bonus : η ≥ 0 — weight of the uncertainty bonus per step.
    allow_observe : bool — allow a first-step "observe" action (no state
                    change, earns explore bonus on failure_node's uncertainty).
    observe_cost : cost of the observe action.
    use_cache : bool — 深度 1 结果经进程级 LRU 跨调用复用 (默认 True):
                 同图同版本下, 二次规划直接命中, 不重建 BN。
    approx_second_step : bool — 次步 ΔP 用残差缩放近似 (默认 True):
                 p(S∪{a}) ≈ p(S)·(1 − Δ₁(a)/baseline), 跳过次步全量推理;
                 False = 精确枚举 (仍受 per-set memo 保护)。
    """
    if failure_node not in causal_dag:
        return RolloutPlan(
            status="empty_graph",
            notes=[f"failure_node '{failure_node}' not present in DAG"],
        )
    if not causal_dag.nodes:
        return RolloutPlan(status="empty_graph", notes=["empty causal graph"])

    if candidate_nodes is None:
        candidate_nodes = [
            n
            for n in causal_dag.nodes
            if n != failure_node
            and nx.has_path(causal_dag, n, failure_node)
        ]
    candidates = [n for n in candidate_nodes if n != failure_node]

    evaluator = _StateEvaluator(causal_dag, cpd_map, failure_node, intervention_value,
                                use_cache=use_cache)
    baseline = evaluator.p_fault(frozenset())
    if baseline is None:
        # Structural-only fallback: order candidates by path count to failure
        path_counts: dict[str, int] = {}
        for src in candidates:
            try:
                path_counts[src] = sum(
                    1
                    for _ in nx.all_simple_paths(causal_dag, src, failure_node, cutoff=8)
                )
            except nx.NetworkXError:
                path_counts[src] = 0
        ranked = sorted(
            [n for n in candidates if path_counts.get(n, 0) > 0],
            key=lambda n: path_counts[n],
            reverse=True,
        )
        actions = [
            RolloutAction(node=n, step=i, action_type="intervene")
            for i, n in enumerate(ranked[:horizon])
        ]
        return RolloutPlan(
            status="structural_only",
            planned_actions=actions,
            p_fault_baseline=0.0,
            search_backend="structural",
            horizon=horizon,
            notes=["no CPD map supplied — path-frequency ordering only"],
        )

    if baseline <= 0.0:
        return RolloutPlan(
            status="no_actions",
            p_fault_baseline=baseline,
            search_backend="structural",
            horizon=horizon,
            notes=["baseline P(fault) already zero"],
        )

    uncertainty = uncertainty or {}
    # 深度 1 精确 ΔP (跨调用 LRU 复用; 残差缩放的基础)
    depth1_delta: dict[str, float] = {}
    if approx_second_step and baseline:
        for a in candidates:
            p1 = evaluator.p_fault(frozenset({a}))
            if p1 is not None:
                depth1_delta[a] = max(0.0, baseline - p1)

    # state = (intervened_tuple, observed_flag, p_fault, utility, actions)
    start: tuple = ((), False, baseline, 0.0, ())
    level = [start]
    explored = 1
    best_state = start
    total_enum = 0

    for step in range(horizon):
        next_level: list[tuple] = []
        for intervened, observed, p_cur, util, actions in level:
            already = set(intervened)
            for a in candidates:
                if a in already:
                    continue
                if approx_second_step and step >= 1 and a in depth1_delta:
                    # 残差缩放近似 (noisy-OR 一致形式): 次步动作按深度 1 相对效果
                    # 收缩剩余故障质量 —— p(S∪{a}) ≈ p(S) − t_a·(1−p(S)),
                    # 其中 t_a = Δ₁(a)/baseline。无全量推理。
                    t_a = depth1_delta[a] / max(baseline, 1e-12)
                    p_new = max(0.0, min(p_cur, p_cur - t_a * (1.0 - p_cur)))
                else:
                    p_new = evaluator.p_fault(frozenset(already) | {a})
                if p_new is None:
                    continue
                delta = p_cur - p_new
                if delta < min_effective_delta:
                    continue  # 最小有效 Δ 过滤
                cost = (action_cost or {}).get(a, default_action_cost)
                bonus = explore_bonus * float(uncertainty.get(a, 0.0))
                util_step = (delta - cost_lambda * cost) + bonus
                util_new = util + (gamma ** step) * util_step
                act = RolloutAction(
                    node=a,
                    step=step,
                    action_type="intervene",
                    delta_p=delta,
                    cost=cost,
                    p_fault_before=p_cur,
                    p_fault_after=p_new,
                    utility_step=util_step,
                )
                next_level.append((intervened + (a,), observed, p_new, util_new, actions + (act,)))
                total_enum += 1

            # observe-first: a single non-state-changing action at the first step
            if allow_observe and not observed:
                bonus_obs = explore_bonus * float(uncertainty.get(failure_node, 0.0))
                util_step_obs = (-cost_lambda * observe_cost) + bonus_obs
                util_new = util + (gamma ** step) * util_step_obs
                act = RolloutAction(
                    node="observe",
                    step=step,
                    action_type="observe",
                    delta_p=0.0,
                    cost=observe_cost,
                    p_fault_before=p_cur,
                    p_fault_after=p_cur,
                    utility_step=util_step_obs,
                )
                next_level.append((intervened, True, p_cur, util_new, actions + (act,)))
                total_enum += 1

        if not next_level:
            break
        # Beam pruning: keep top-K by cumulative utility
        next_level.sort(key=lambda s: s[3], reverse=True)
        level = next_level[:beam_width]
        explored += len(level)
        if level[0][3] > best_state[3]:
            best_state = level[0]

    # best_state has the highest cumulative utility reached
    if best_state[3] <= 0.0 and not best_state[4]:
        best_state = start

    _, _, p_final, util_final, acts = best_state
    actions = list(acts)
    total_cost = sum(a.cost for a in actions)
    backend = (
        "exact_enumeration"
        if beam_width >= max(total_enum, 1) and total_enum <= beam_width * horizon
        else "beam_pruned"
    )
    if not cpd_map:
        backend = "structural"

    return RolloutPlan(
        status="ok" if actions else "no_actions",
        planned_actions=actions,
        total_utility=util_final,
        p_fault_baseline=baseline,
        p_fault_after_plan=p_final,
        total_cost=total_cost,
        explored_states=explored,
        search_backend=backend,
        horizon=horizon,
    )


def _sequence_utility(
    causal_dag: nx.DiGraph,
    *,
    sequence: Sequence[str],
    failure_node: str,
    cpd_map: dict[str, Any] | None,
    gamma: float = 0.95,
    cost_lambda: float = 0.05,
    action_cost: dict[str, float] | None = None,
    default_action_cost: float = DEFAULT_ACTION_COST,
    intervention_value: Any = "ok",
) -> float:
    """精确计算一条有序干预序列的累计折扣效用 (测试/对拍用)。

    utility(seq) = Σ_t γ^t (ΔP_t − λ·C(a_t)),  其中 ΔP_t 是 a_t 在
    (a_0..a_{t-1} 已干预) 后的边际失败率下降。
    """
    evaluator = _StateEvaluator(causal_dag, cpd_map, failure_node, intervention_value)
    p = evaluator.p_fault(frozenset())
    if p is None:
        return 0.0
    util = 0.0
    done: set[str] = set()
    for t, node in enumerate(sequence):
        done.add(node)
        p_new = evaluator.p_fault(frozenset(done))
        if p_new is None:
            break
        delta = max(0.0, p - p_new)
        cost = (action_cost or {}).get(node, default_action_cost)
        util += (gamma ** t) * (delta - cost_lambda * cost)
        p = p_new
    return util
