"""Tests for oprim.entity_graph_search — ≥10 tests."""

from __future__ import annotations

import types

import pytest

from oprim.entity_graph_search import entity_graph_search


def make_edges(dst_ids: list[str]):
    """Return a list_edges callable that yields SimpleNamespace edges for given dst_ids."""
    edges = [types.SimpleNamespace(dst_id=d) for d in dst_ids]

    def list_edges(node_id: str):
        return edges

    return list_edges


def empty_list_edges(node_id: str):
    return []


# 1. Empty seeds → []
def test_empty_seeds():
    result = entity_graph_search(seed_ids=[], list_edges=empty_list_edges)
    assert result == []


# 2. Seed with no edges → []
def test_seed_no_edges():
    result = entity_graph_search(seed_ids=["A"], list_edges=empty_list_edges)
    assert result == []


# 3. 1-hop finds direct neighbor
def test_one_hop_direct_neighbor():
    list_edges = make_edges(["B"])
    result = entity_graph_search(seed_ids=["A"], list_edges=list_edges, hops=1)
    assert len(result) == 1
    assert result[0][0] == "B"
    assert result[0][1] > 0


# 4. 2-hop finds indirect neighbor (B→C via second hop)
def test_two_hop_indirect_neighbor():
    call_count = [0]

    def list_edges(node_id: str):
        call_count[0] += 1
        if node_id == "A":
            return [types.SimpleNamespace(dst_id="B")]
        if node_id == "B":
            return [types.SimpleNamespace(dst_id="C")]
        return []

    result = entity_graph_search(seed_ids=["A"], list_edges=list_edges, hops=2)
    node_ids = [r[0] for r in result]
    assert "B" in node_ids
    assert "C" in node_ids


# 5. top_k limits output
def test_top_k_limits():
    list_edges = make_edges(["B", "C", "D", "E", "F"])
    result = entity_graph_search(seed_ids=["A"], list_edges=list_edges, hops=1, top_k=3)
    assert len(result) <= 3


# 6. Multiple paths to same node increase score
def test_multiple_paths_increase_score():
    # Both seeds A and B lead to C
    def list_edges(node_id: str):
        if node_id in ("A", "B"):
            return [types.SimpleNamespace(dst_id="C")]
        return []

    result = entity_graph_search(seed_ids=["A", "B"], list_edges=list_edges, hops=1)
    assert len(result) == 1
    assert result[0][0] == "C"
    assert result[0][1] == pytest.approx(2.0)


# 7. Seeds excluded from results
def test_seeds_excluded():
    # A → B, B → A (back-edge should not put A in results)
    def list_edges(node_id: str):
        if node_id == "A":
            return [types.SimpleNamespace(dst_id="B"), types.SimpleNamespace(dst_id="A")]
        return []

    result = entity_graph_search(seed_ids=["A"], list_edges=list_edges, hops=1)
    node_ids = [r[0] for r in result]
    assert "A" not in node_ids
    assert "B" in node_ids


# 8. Result sorted by score descending
def test_sorted_descending():
    # A → [B, C, D]; B → [C] so C gets more visits over 2 hops
    def list_edges(node_id: str):
        if node_id == "A":
            return [
                types.SimpleNamespace(dst_id="B"),
                types.SimpleNamespace(dst_id="C"),
            ]
        if node_id == "B":
            return [types.SimpleNamespace(dst_id="C")]
        return []

    result = entity_graph_search(seed_ids=["A"], list_edges=list_edges, hops=2)
    scores = [r[1] for r in result]
    assert scores == sorted(scores, reverse=True)


# 9. Multiple seed nodes merged
def test_multiple_seeds_merged():
    def list_edges(node_id: str):
        if node_id == "A":
            return [types.SimpleNamespace(dst_id="X")]
        if node_id == "B":
            return [types.SimpleNamespace(dst_id="Y")]
        return []

    result = entity_graph_search(seed_ids=["A", "B"], list_edges=list_edges, hops=1)
    node_ids = [r[0] for r in result]
    assert "X" in node_ids
    assert "Y" in node_ids


# 10. list_edges returning Edge-like objects with .dst_id (SimpleNamespace)
def test_simplenamespace_edge_objects():
    edges = [types.SimpleNamespace(dst_id="Z", weight=0.9)]

    def list_edges(node_id: str):
        return edges

    result = entity_graph_search(seed_ids=["A"], list_edges=list_edges, hops=1)
    assert result[0][0] == "Z"


# 11. top_k=0 returns empty list
def test_top_k_zero():
    list_edges = make_edges(["B"])
    result = entity_graph_search(seed_ids=["A"], list_edges=list_edges, hops=1, top_k=0)
    assert result == []


# 12. Decay by hop: 1st hop score > 2nd hop score for same distance pattern
def test_decay_by_hop():
    def list_edges(node_id: str):
        if node_id == "A":
            return [types.SimpleNamespace(dst_id="B")]
        if node_id == "B":
            return [types.SimpleNamespace(dst_id="C")]
        return []

    result = entity_graph_search(seed_ids=["A"], list_edges=list_edges, hops=2)
    scores = {r[0]: r[1] for r in result}
    assert scores["B"] > scores["C"]
