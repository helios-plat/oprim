"""oprim.graph_impact_analysis — networkx-based impact propagation (PageRank + BFS).

Analyses a code graph and computes the downstream impact set of seed nodes
via BFS reachability, plus eigenvector centrality for ranking.

3O element: ``oprim.graph_impact_analysis`` (``_graph_impact_analysis`` legacy name).
"""

from __future__ import annotations

from typing import Any

import networkx as nx


def graph_impact_analysis(
    graph: dict[str, Any],
    seed_nodes: list[str],
    context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Compute the downstream impact of seed nodes on a directed graph.

    Args:
        graph: ``{"nodes": [...], "edges": [...]}`` dict (as produced by ``code_graph_parse``).
        seed_nodes: Starting nodes for reachability analysis.
        context: Optional config (``max_depth``, ``centrality``).

    Returns:
        ``{impacted, edges_traversed, centrality_ranking, status}``
    """
    ctx = context or {}
    G = nx.DiGraph()
    for n in graph.get("nodes", []):
        G.add_node(n.get("name", "?"))
    for e in graph.get("edges", []):
        G.add_edge(e.get("from", ""), e.get("to", ""))

    # BFS reachability from each seed
    impacted: list[str] = []
    seen: set[str] = set()
    queue = list(seed_nodes)
    max_depth = int(ctx.get("max_depth", 10))
    depth: dict[str, int] = {}
    for s in seed_nodes:
        depth[s] = 0
    while queue:
        node = queue.pop(0)
        if node in seen or node not in G:
            continue
        seen.add(node)
        impacted.append(node)
        if depth.get(node, 0) >= max_depth:
            continue
        for _, succ in G.out_edges(node):
            if succ not in seen:
                queue.append(succ)
                depth[succ] = depth.get(node, 0) + 1

    # PageRank centrality for ranking
    try:
        pr = nx.pagerank(G, alpha=0.85)
        ranking = sorted(pr.items(), key=lambda kv: -kv[1])[:50]
        top = [{"node": n, "score": round(s, 4)} for n, s in ranking]
    except Exception:
        top = []

    return {
        "impacted": impacted,
        "edges_traversed": len(impacted),
        "max_depth_reached": max(depth.values()) if depth else 0,
        "centrality_top": top,
        "status": "analyzed",
    }
