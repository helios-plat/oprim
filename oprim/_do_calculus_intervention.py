"""Atomic operator: full do-calculus intervention on a causal DAG.

Supports two modes:
  1. Structural only (always available) – graph mutilation + path enumeration.
  2. Quantitative (when cpd_map of TabularCPD is supplied) – builds a
     DiscreteBayesianNetwork on the mutilated graph, forces a deterministic
     CPD on the intervened node, and returns exact post-intervention
     marginals via VariableElimination.

Formula realised:
    P(Y | do(X = x)) = Σ_z P(Y | X=x, Z=z) P(Z=z)
(via the mutilated graph + fixed CPD of X).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import networkx as nx
import numpy as np

try:
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    from pgmpy.models import DiscreteBayesianNetwork

    _HAS_PGMPY = True
except ImportError:  # pragma: no cover - exercised only in minimal envs
    _HAS_PGMPY = False
    DiscreteBayesianNetwork = None  # type: ignore
    TabularCPD = None  # type: ignore
    VariableElimination = None  # type: ignore


def _extract_p_fault(dist: dict[str, float] | None) -> float | None:
    """Pull the failure probability out of a {state: prob} marginal dict."""
    if not dist:
        return None
    for key in ("fault", "1", "fail", "timeout", "error"):
        if key in dist:
            return float(dist[key])
    # fallback: highest non-"ok" mass
    ok_keys = {"ok", "0", "success", "healthy"}
    candidates = [v for k, v in dist.items() if k not in ok_keys]
    return max(candidates) if candidates else None


def _value_to_state_index(
    value: Any,
    state_names: Sequence[Any] | None,
    cardinality: int,
) -> int:
    """Map an intervention value to a discrete state index."""
    if state_names is not None:
        try:
            return list(state_names).index(value)
        except ValueError:
            # fallback: try string / int coercion
            try:
                return list(state_names).index(str(value))
            except ValueError:
                pass
    # Assume the value itself is already a valid integer index
    if isinstance(value, (int, np.integer)) and 0 <= int(value) < cardinality:
        return int(value)
    # Last resort for binary common case
    if cardinality == 2:
        if value in (1, True, "1", "true", "True", "faulty", "fail", "timeout"):
            return 1
        return 0
    raise ValueError(
        f"Cannot map intervention_value={value!r} to a state index "
        f"(cardinality={cardinality}, state_names={state_names})"
    )


def _make_deterministic_cpd(
    variable: str,
    intervention_value: Any,
    cardinality: int = 2,
    state_names: Sequence[Any] | None = None,
) -> TabularCPD:
    """
    Create a TabularCPD with no parents that puts mass 1.0 on the
    intervened state. This is the mathematical embodiment of do(X=x).
    """
    idx = _value_to_state_index(intervention_value, state_names, cardinality)
    values = np.zeros((cardinality, 1))
    values[idx, 0] = 1.0
    sn = {variable: list(state_names)} if state_names is not None else None
    return TabularCPD(
        variable=variable,
        variable_card=cardinality,
        values=values,
        state_names=sn,
    )


def _extract_card_and_states(
    cpd: TabularCPD,
) -> tuple[int, list[Any] | None]:
    card = cpd.variable_card
    states = None
    if cpd.state_names and cpd.variable in cpd.state_names:
        states = list(cpd.state_names[cpd.variable])
    return card, states


def _do_calculus_intervention(
    causal_dag: nx.DiGraph,
    *,
    target_node: str,
    intervention_value: int | float | str | bool,
    outcome_nodes: list[str] | None = None,
    cpd_map: dict[str, Any] | None = None,
    evidence: dict[str, Any] | None = None,
    use_cache: bool = True,
    enumerate_paths: bool = True,
) -> dict[str, Any]:
    """
    Perform a do(X = x) intervention.

    Parameters
    ----------
    causal_dag : nx.DiGraph
        Edges point cause → effect.
    target_node : str
        Node X to intervene on.
    intervention_value : scalar
        Value x forced by the intervention.
    outcome_nodes : list[str], optional
        Nodes whose post-intervention marginal we want.
    cpd_map : dict[str, TabularCPD], optional
        Full set of CPDs for quantitative inference. Keys must cover
        every node that has a CPD in the original model.
    evidence : dict, optional
        Additional observational evidence to condition on *after*
        the intervention (rarely needed; kept for completeness).
    use_cache : bool
        进程级 LRU 缓存 (默认 True): 同图同版本同干预 → 直接命中,
        省掉 BN 重建 + VariableElimination。键含图指纹 + CPD 指纹 + 引擎版本。
    enumerate_paths : bool
        是否枚举完整结构路径列表 (默认 True, 兼容旧 API)。
        False 时只做存在性判断 + 路径计数 (O(V+E) 拓扑 DP),
        不做 all_simple_paths 的指数级枚举。

    Returns
    -------
    dict with at least:
        status, intervened_node, intervention_value,
        mutilated_edges, remaining_parents,
        structural_effect_paths,
        post_intervention_distribution  (None or {node: {state: prob, ...}}),
        inference_backend
    """
    # ------------------------------------------------------------------
    # 0. LRU 缓存: 同图同版本同干预 → 直接命中 (省 BN 重建 + VE)
    # ------------------------------------------------------------------
    if use_cache:
        from oprim._inference_cache import (  # noqa: PLC0415
            get_intervention_cache,
            graph_fingerprint,
        )

        _key = (
            graph_fingerprint(causal_dag, cpd_map),
            target_node,
            str(intervention_value),
            tuple(sorted(outcome_nodes)) if outcome_nodes else (),
        )
        _hit = get_intervention_cache().get(_key)
        if _hit is not None:
            import copy  # noqa: PLC0415
            return copy.deepcopy(_hit)

    if target_node not in causal_dag:
        return {
            "status": "error",
            "error": f"target_node '{target_node}' not present in DAG",
            "intervened_node": target_node,
            "intervention_value": intervention_value,
        }

    # ------------------------------------------------------------------
    # 1. Graph mutilation
    # ------------------------------------------------------------------
    mutilated = causal_dag.copy()
    incoming = list(mutilated.in_edges(target_node))
    mutilated.remove_edges_from(incoming)
    remaining_parents = list(mutilated.predecessors(target_node))
    cut_edges = [(u, v) for u, v in incoming]

    result: dict[str, Any] = {
        "status": "ok",
        "intervened_node": target_node,
        "intervention_value": intervention_value,
        "mutilated_edges": cut_edges,
        "remaining_parents": remaining_parents,
        "post_intervention_distribution": None,
        "structural_effect_paths": [],
        "delta_p_failure": None,
    }

    # ------------------------------------------------------------------
    # 2. Structural paths
    # ------------------------------------------------------------------
    if outcome_nodes is None:
        outcome_nodes = [
            n
            for n in mutilated.nodes
            if n != target_node and nx.has_path(mutilated, target_node, n)
        ]

    for y in outcome_nodes:
        if y in mutilated and nx.has_path(mutilated, target_node, y):
            if enumerate_paths:
                paths = list(nx.all_simple_paths(mutilated, target_node, y, cutoff=8))
            else:
                # 只做存在性判断 + O(V+E) 计数, 不枚举指数级路径列表
                paths = []
            from oprim._inference_cache import count_simple_paths_dag  # noqa: PLC0415

            result["structural_effect_paths"].append(
                {"outcome": y, "paths": paths, "num_paths":
                 len(paths) if enumerate_paths
                 else count_simple_paths_dag(mutilated, target_node, y)}
            )

    # ------------------------------------------------------------------
    # 3. Quantitative path (full CPD inference)
    # ------------------------------------------------------------------
    if not (_HAS_PGMPY and cpd_map is not None and len(cpd_map) > 0):
        result["inference_backend"] = "structural_only"
        result["formula_reference"] = (
            "P(Y | do(X=x)) = Σ_z P(Y | X=x, Z=z) P(Z=z)  "
            "(Pearl do-calculus / back-door when Z blocks)"
        )
        return result

    try:
        # 3a. Build model on the *mutilated* graph
        #     (intervened node has no parents)
        model = DiscreteBayesianNetwork(list(mutilated.edges()))
        # Ensure isolated nodes are also present
        for n in causal_dag.nodes:
            if n not in model.nodes():
                model.add_node(n)

        # 3b. Attach CPDs
        #     - for target_node: replace with deterministic CPD
        #     - for every other node: keep original CPD
        #       (the functional mechanism of children does not change
        #        under do-intervention.)
        original_cpd = cpd_map.get(target_node)
        if original_cpd is not None and isinstance(original_cpd, TabularCPD):
            card, states = _extract_card_and_states(original_cpd)
        else:
            # sensible default for binary failure nodes
            card, states = 2, ["ok", "fault"]

        det_cpd = _make_deterministic_cpd(
            target_node,
            intervention_value,
            cardinality=card,
            state_names=states,
        )
        model.add_cpds(det_cpd)

        for node, cpd in cpd_map.items():
            if node == target_node:
                continue
            if not isinstance(cpd, TabularCPD):
                continue
            # Other nodes keep their original CPD: only the intervened
            # node's incoming edges were cut, so every other CPD's parent
            # set still matches the mutilated graph.
            model.add_cpds(cpd)

        # 3c. Inference
        infer = VariableElimination(model)

        post_dist: dict[str, dict[Any, float]] = {}
        for y in outcome_nodes:
            if y not in model.nodes():
                continue
            # Query the marginal of y under the intervention
            # (the deterministic CPD already encodes do(X=x))
            q = infer.query(variables=[y], evidence=evidence or {}, show_progress=False)
            # Convert to plain dict {state_name_or_idx: probability}
            states_y = (list(q.state_names[y]) if y in q.state_names
                        else list(range(q.cardinality[0])))
            probs = q.values.ravel()
            post_dist[y] = {
                str(states_y[i]): float(probs[i]) for i in range(len(probs))
            }

        result["post_intervention_distribution"] = post_dist
        result["inference_backend"] = "pgmpy_variable_elimination"

        # Optional convenience: if an outcome looks like a binary failure
        # node, compute delta against a naïve observational baseline.
        # (Caller can also compute this themselves.)
        for y, dist in post_dist.items():
            if "fault" in dist or "1" in dist or "fail" in dist:
                p_fault = dist.get("fault", dist.get("1", dist.get("fail", 0.0)))
                result["delta_p_failure"] = {
                    "outcome": y,
                    "p_fault_after_do": p_fault,
                    "note": "Compare with observational P(fault) supplied by caller",
                }
                break

    except Exception as exc:  # noqa: BLE001
        result["post_intervention_distribution"] = {
            "error": str(exc),
            "fallback": "structural only",
        }
        result["inference_backend"] = "structural_fallback"
        result["status"] = "partial"

    # ------------------------------------------------------------------
    # 4. 写回缓存 (只缓存完整结果)
    # ------------------------------------------------------------------
    if use_cache and result["status"] == "ok":
        get_intervention_cache().set(_key, result)

    return result


def intervene_on_store(
    store,
    *,
    target_node: str,
    intervention_value: Any,
    outcome_nodes: list[str] | None = None,
    cpd_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Adapter that pulls the networkx graph from a CausalGraphStore."""
    dag = store.get_graph()
    return _do_calculus_intervention(
        dag,
        target_node=target_node,
        intervention_value=intervention_value,
        outcome_nodes=outcome_nodes,
        cpd_map=cpd_map,
    )


# ---------------------------------------------------------------------------
# Helper factory: build a realistic binary failure CPD map for diagnosis
# ---------------------------------------------------------------------------

def build_binary_failure_cpd_map(
    dag: nx.DiGraph,
    *,
    fault_prior: float = 0.15,
    transmission: float = 0.7,
    base_failure: float = 0.05,
) -> dict[str, TabularCPD]:
    """
    Construct a complete set of TabularCPDs for a binary (ok/fault) network.

    Conventions
    -----------
    - Every node is binary with states ["ok", "fault"].
    - Root nodes (in-degree 0) get P(fault) = fault_prior.
    - Non-root nodes: noisy-OR style
          P(fault | parents) ≈ 1 - (1-base) * Π (1-transmission if parent=fault)

    This gives a usable quantitative layer for causal_fault_diagnose
    without requiring the caller to hand-craft every CPD.
    """
    if not _HAS_PGMPY:
        raise RuntimeError("pgmpy is required for CPD construction")

    cpd_map: dict[str, TabularCPD] = {}
    nodes = list(dag.nodes)

    for node in nodes:
        parents = list(dag.predecessors(node))
        card = 2
        state_names = {node: ["ok", "fault"]}

        if not parents:
            # Root
            values = np.array([[1.0 - fault_prior], [fault_prior]])
            cpd = TabularCPD(
                variable=node,
                variable_card=card,
                values=values,
                state_names=state_names,
            )
        else:
            n_pa = len(parents)
            # 2^n_pa columns
            n_configs = 2 ** n_pa
            values = np.zeros((2, n_configs))
            for config in range(n_configs):
                # bit representation of parent states (0=ok, 1=fault)
                p_ok = 1.0 - base_failure
                for i, _ in enumerate(parents):
                    parent_is_fault = (config >> i) & 1
                    if parent_is_fault:
                        p_ok *= 1.0 - transmission
                values[0, config] = p_ok
                values[1, config] = 1.0 - p_ok

            evidence = parents
            evidence_card = [2] * n_pa
            sn = {node: ["ok", "fault"]}
            for p in parents:
                sn[p] = ["ok", "fault"]
            cpd = TabularCPD(
                variable=node,
                variable_card=card,
                values=values,
                evidence=evidence,
                evidence_card=evidence_card,
                state_names=sn,
            )
        cpd_map[node] = cpd

    return cpd_map
