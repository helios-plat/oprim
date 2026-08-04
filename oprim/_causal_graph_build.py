"""oprim.causal_graph_build — 基于 decision_trail 构建因果 DAG 图谱.

单次从 Agent 历史执行轨迹提取节点并构建因果依赖拓扑（纯计算）。

Example:
    >>> r = causal_graph_build([
    ...     {"type": "action", "action": "read", "status": "success"},
    ...     {"type": "action", "action": "edit", "output_refs": ["step_0_action"]},
    ... ])
    >>> r["node_count"]
    2
"""

from __future__ import annotations

from typing import Any

from oprim._exceptions import OprimValidationError


def causal_graph_build(
    decision_trail: list[dict[str, Any]],
    *,
    strict_mode: bool = False,
) -> dict[str, Any]:
    """构建因果 DAG。

    Args:
        decision_trail: Agent 执行轨迹步骤列表，每项可含
            type/thought/action/status/output_refs。
        strict_mode: 严格模式（当前保留语义，供上层策略开关）。

    Returns:
        {"node_count": int, "edge_count": int,
         "graph": {"nodes": [...], "edges": [...]}}

    Raises:
        OprimValidationError: decision_trail 为空 / 非列表。
    """
    if not isinstance(decision_trail, list):
        raise OprimValidationError(
            f"causal_graph_build: decision_trail must be a list, got {type(decision_trail).__name__}"
        )
    if not decision_trail:
        raise OprimValidationError("causal_graph_build: decision_trail must not be empty")

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []

    prev_node_id = ""
    for idx, step in enumerate(decision_trail):
        node_id = f"step_{idx}_{step.get('type', 'action')}"
        nodes.append(
            {
                "id": node_id,
                "type": step.get("type"),
                "thought": step.get("thought", ""),
                "action": step.get("action", ""),
                "status": step.get("status", "success"),
            }
        )

        if prev_node_id:
            edges.append(
                {"source": prev_node_id, "target": node_id, "relation": "causes"}
            )

        # 工具调用的输出因果关联
        output_refs = step.get("output_refs") or []
        for ref in output_refs:
            edges.append(
                {"source": str(ref), "target": node_id, "relation": "depends_on"}
            )

        prev_node_id = node_id

    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "graph": {"nodes": nodes, "edges": edges},
    }
