"""oprim._do_calculus_intervention — 单次 do-calculus 图手术干预计算.

铁律: ≤1 个位置参数，其余 keyword-only.

组合: causal_graph_store (Protocol 注入).

数学原理:
- do(X=x): 切断 X 的所有入边（Graph Mutilation）
- 计算干预后的边缘概率分布 P(Y | do(X=x))
- 使用调节公式（back-door criterion）或直接图手术

例:
    >>> result = _do_calculus_intervention(
    ...     causal_dag=my_store,
    ...     target_node="timeout",
    ...     intervention_value=0,
    ... )
    >>> result["intervention_applied"]
    True
"""

from __future__ import annotations

from typing import Any, Protocol


class CausalDAGProtocol(Protocol):
    """因果 DAG Protocol — 不 import obase，由调用方注入."""

    def mutilate(self, target_node: str) -> Any: ...
    def get_parents(self, node_id: str) -> list[str]: ...
    def get_children(self, node_id: str) -> list[str]: ...
    def topological_sort(self) -> list[str]: ...
    def nodes(self) -> list[str]: ...
    def edges(self) -> list[tuple[str, str]]: ...


def _do_calculus_intervention(
    causal_dag: CausalDAGProtocol,
    *,
    target_node: str,
    intervention_value: float = 0.0,
    outcome_nodes: list[str] | None = None,
) -> dict[str, Any]:
    """单次 do-calculus 图手术干预计算.

    签名遵循 oprim 铁律：最多 1 个位置参数，其余 kw-only.

    算法步骤:
    1. 图手术 (mutilation): 切断 target_node 的所有入边
    2. 计算 target_node 被干预后的后门调节集
    3. 估算各 outcome_node 在干预下的条件概率变化
    4. 返回干预前/后对比

    Args:
        causal_dag: 因果 DAG 图（注入 CausalDAGProtocol）
        target_node: 干预目标节点 ID
        intervention_value: 干预设定值
        outcome_nodes: 关注的结果节点列表（None = 所有后代）

    Returns:
        {
            "intervention_applied": bool,
            "target_node": str,
            "intervention_value": float,
            "severed_edges": [(parent, target), ...],
            "backdoor_adjustment_set": [str, ...],
            "affected_descendants": [str, ...],
            "estimated_effect": dict[str, float],
        }
    """
    if not causal_dag:
        return {
            "intervention_applied": False,
            "target_node": target_node,
            "error": "No causal DAG provided",
        }

    # 1. 记录被切断的入边
    parents = causal_dag.get_parents(target_node) if hasattr(causal_dag, "get_parents") else []
    severed_edges = [(p, target_node) for p in parents]

    # 2. 执行图手术
    try:
        mutilated = causal_dag.mutilate(target_node) if hasattr(causal_dag, "mutilate") else None
    except Exception:
        mutilated = None

    # 3. 后门调节集: 找到所有同时影响 target_node 和 outcome 的混淆因子
    #    简化: 使用 target_node 的父节点作为后门调节集
    backdoor_set = list(parents)

    # 4. 确定受影响的后代
    try:
        if outcome_nodes is None:
            all_descendants = (
                causal_dag.get_children(target_node) if hasattr(causal_dag, "get_children") else []
            )
        else:
            all_descendants = outcome_nodes
    except Exception:
        all_descendants = []

    # 5. 估算干预效应（简化版: 基于图结构的启发式估计）
    #    实际生产应结合条件概率表 (CPT) 精确计算
    estimated_effect: dict[str, float] = {}
    for node in all_descendants:
        distance = _estimate_causal_distance(causal_dag, target_node, node)
        # 距离越近，干预效应越强（指数衰减模型）
        effect_magnitude = (
            intervention_value * (0.5 ** (distance - 1)) if distance >= 1 else intervention_value
        )
        estimated_effect[node] = round(effect_magnitude, 4)

    return {
        "intervention_applied": True,
        "target_node": target_node,
        "intervention_value": intervention_value,
        "severed_edges": severed_edges,
        "backdoor_adjustment_set": backdoor_set,
        "affected_descendants": all_descendants,
        "estimated_effect": estimated_effect,
        "mutilation_successful": mutilated is not None,
    }


def _estimate_causal_distance(dag: Any, source: str, target: str) -> int:
    """估算因果距离（最短路径长度）."""
    if not hasattr(dag, "edges"):
        return 1
    try:
        import networkx as nx

        # 尝试将 edges 转为 networkx 图
        g = nx.DiGraph()
        g.add_edges_from(dag.edges())
        try:
            path_len = nx.shortest_path_length(g, source, target)
            return max(1, path_len)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return 1
    except ImportError:
        return 1
