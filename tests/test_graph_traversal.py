"""Tests for oprim.graph_traversal."""

from __future__ import annotations

import pytest

from oprim.graph_traversal import graph_traversal

# Simple graph: A->B->C->D, A->E
GRAPH = {
    "A": ["B", "E"],
    "B": ["C"],
    "C": ["D"],
    "D": [],
    "E": [],
}


def _neighbors(graph):
    def get(node):
        return graph.get(node, [])

    return get


def test_bfs_visits_nodes_breadth_first():
    result = graph_traversal(
        start_nodes=["A"], get_neighbors=_neighbors(GRAPH), mode="bfs", max_depth=10
    )
    visited = result["visited"]
    # BFS: A first, then B and E before C
    assert visited.index("A") < visited.index("B")
    assert visited.index("B") < visited.index("C")
    assert visited.index("A") < visited.index("E")
    # B and E both at depth 1, so both before C (depth 2)
    assert visited.index("E") < visited.index("C")


def test_dfs_visits_nodes_depth_first():
    result = graph_traversal(
        start_nodes=["A"], get_neighbors=_neighbors(GRAPH), mode="dfs", max_depth=10
    )
    visited = result["visited"]
    # DFS from A: should go deep before backtracking
    assert "A" in visited
    assert "B" in visited
    assert "C" in visited
    assert "D" in visited
    assert "E" in visited
    # In DFS A→B→C→D path, C should come before E
    assert visited.index("C") < visited.index("E")


def test_max_depth_limits_traversal():
    result = graph_traversal(
        start_nodes=["A"], get_neighbors=_neighbors(GRAPH), mode="bfs", max_depth=1
    )
    # depth 0: A, depth 1: B, E — C should NOT be visited (depth 2)
    assert "A" in result["visited"]
    assert "B" in result["visited"]
    assert "E" in result["visited"]
    assert "C" not in result["visited"]


def test_max_nodes_truncates_and_sets_truncated():
    result = graph_traversal(
        start_nodes=["A"], get_neighbors=_neighbors(GRAPH), mode="bfs", max_nodes=2
    )
    assert result["truncated"] is True
    assert len(result["visited"]) <= 2


def test_cyclic_graph_no_infinite_loop():
    cyclic = {"A": ["B"], "B": ["C"], "C": ["A"]}
    result = graph_traversal(
        start_nodes=["A"], get_neighbors=_neighbors(cyclic), mode="bfs", max_depth=10
    )
    assert set(result["visited"]) == {"A", "B", "C"}
    assert result["truncated"] is False


def test_empty_start_nodes_returns_empty():
    result = graph_traversal(start_nodes=[], get_neighbors=_neighbors(GRAPH))
    assert result["visited"] == []
    assert result["depth_map"] == {}
    assert result["edges_traversed"] == []
    assert result["truncated"] is False


def test_depth_map_correct():
    result = graph_traversal(
        start_nodes=["A"], get_neighbors=_neighbors(GRAPH), mode="bfs", max_depth=10
    )
    dm = result["depth_map"]
    assert dm["A"] == 0
    assert dm["B"] == 1
    assert dm["E"] == 1
    assert dm["C"] == 2
    assert dm["D"] == 3


def test_edges_traversed_populated():
    result = graph_traversal(
        start_nodes=["A"], get_neighbors=_neighbors(GRAPH), mode="bfs", max_depth=10
    )
    edges = result["edges_traversed"]
    assert len(edges) > 0
    assert ("A", "B") in edges or ("A", "E") in edges


def test_single_isolated_node():
    result = graph_traversal(start_nodes=["X"], get_neighbors=lambda n: [], mode="bfs", max_depth=5)
    assert result["visited"] == ["X"]
    assert result["depth_map"] == {"X": 0}
    assert result["edges_traversed"] == []
    assert result["truncated"] is False


def test_multi_root_start_nodes_merged():
    result = graph_traversal(
        start_nodes=["A", "D"],
        get_neighbors=_neighbors(GRAPH),
        mode="bfs",
        max_depth=10,
    )
    assert "A" in result["visited"]
    assert "D" in result["visited"]
    assert result["depth_map"]["A"] == 0
    assert result["depth_map"]["D"] == 0


def test_no_truncation_when_within_limits():
    result = graph_traversal(
        start_nodes=["A"], get_neighbors=_neighbors(GRAPH), mode="bfs", max_nodes=1000
    )
    assert result["truncated"] is False


def test_dfs_depth_map_correct():
    result = graph_traversal(
        start_nodes=["A"], get_neighbors=_neighbors(GRAPH), mode="dfs", max_depth=10
    )
    dm = result["depth_map"]
    assert dm["A"] == 0
    assert dm["B"] == 1
    assert dm["C"] == 2
    assert dm["D"] == 3
