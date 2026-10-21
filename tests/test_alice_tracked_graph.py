from __future__ import annotations

import pytest

from oprim.alice_tracked_graph import TrackedGraph, TrackedGraphError, TrackedNode


def _graph() -> TrackedGraph:
    graph = TrackedGraph()
    graph.add_node(TrackedNode("asset:BTC", "asset", "Bitcoin", "watchlist:BTC"))
    graph.add_node(TrackedNode("topic:ETF", "topic", "Spot ETF flows", "research:etf-flows"))
    graph.add_node(TrackedNode("thesis:flows", "thesis", "Flows support demand", "report:thesis-01"))
    return graph


def test_edges_require_endpoints_and_provenance() -> None:
    graph = _graph()
    with pytest.raises(TrackedGraphError, match="endpoints"):
        graph.link("asset:BTC", "missing", "relates_to", evidence_ref="report:1")
    with pytest.raises(TrackedGraphError, match="evidence_ref"):
        graph.link("asset:BTC", "topic:ETF", "relates_to", evidence_ref="")
    with pytest.raises(TrackedGraphError, match=r"\[0, 1\]"):
        graph.link("asset:BTC", "topic:ETF", "relates_to", evidence_ref="report:1", confidence=1.1)


def test_neighbors_are_deterministic_and_confidence_filtered() -> None:
    graph = _graph()
    graph.link("asset:BTC", "topic:ETF", "relates_to", evidence_ref="report:a", confidence=0.7, created_at="2026-10-20T10:00:00+00:00")
    graph.link("asset:BTC", "thesis:flows", "supports", evidence_ref="report:b", confidence=0.95, created_at="2026-10-20T10:00:00+00:00")
    results = graph.neighbors("asset:BTC", direction="outgoing", min_confidence=0.8)
    assert [(item.node.node_id, item.confidence) for item in results] == [("thesis:flows", 0.95)]
    assert graph.neighbors("topic:ETF", direction="incoming")[0].node.node_id == "asset:BTC"


def test_graph_roundtrip_preserves_explicit_links() -> None:
    graph = _graph()
    graph.link("asset:BTC", "topic:ETF", "tracks", evidence_ref="user:tracked-item", confidence=1)
    restored = TrackedGraph.from_dict(graph.to_dict())
    assert restored.to_dict() == graph.to_dict()
    assert len(restored.neighbors("asset:BTC")) == 1


def test_conflicting_nodes_and_links_fail_closed() -> None:
    graph = _graph()
    with pytest.raises(TrackedGraphError, match="conflicting definition"):
        graph.add_node(TrackedNode("asset:BTC", "asset", "Different label", "watchlist:BTC"))
    graph.link("asset:BTC", "topic:ETF", "tracks", evidence_ref="one", confidence=1, created_at="2026-10-20T10:00:00+00:00")
    with pytest.raises(TrackedGraphError, match="conflicting relationship"):
        graph.link("asset:BTC", "topic:ETF", "tracks", evidence_ref="two", confidence=1, created_at="2026-10-20T10:00:00+00:00")


def test_graph_rejects_naive_timestamps_and_unknown_node_kinds() -> None:
    graph = _graph()
    with pytest.raises(TrackedGraphError, match="timezone"):
        graph.link("asset:BTC", "topic:ETF", "tracks", evidence_ref="user:track", created_at="2026-10-20T10:00:00")
    with pytest.raises(TrackedGraphError, match="node kind"):
        TrackedNode("mystery", "other", "Mystery", "report:x")  # type: ignore[arg-type]
