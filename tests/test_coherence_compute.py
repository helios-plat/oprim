"""Tests for oprim.coherence_compute — ≥10 tests."""

from __future__ import annotations

import pytest

from oprim.coherence_compute import (
    GRADE_LADDER,
    INDEPENDENT_SOURCES,
    coherence_compute,
    _grade_index,
    _status_of,
)


def _confirmed_node(grade: str = "moderate", source: str = "reproducible_empirical") -> dict:
    return {
        "is_ku": True,
        "epistemic_status": {"completeness": {"grade": grade, "source": source, "defeaters": []}},
    }


def _unconfirmed_node(grade: str = "low", source: str = "weak_empirical") -> dict:
    return {
        "is_ku": True,
        "epistemic_status": {"completeness": {"grade": grade, "source": source, "defeaters": []}},
    }


# 1. Empty graph → all-zero counts
def test_empty_graph():
    result = coherence_compute(nodes={}, edges=[])
    assert result == {}


# 2. Single node, no edges → all-zero counts
def test_single_node_no_edges():
    nodes = {"A": _confirmed_node()}
    result = coherence_compute(nodes=nodes, edges=[])
    assert result["A"]["supports_from_confirmed"] == 0
    assert result["A"]["contradicts_from_confirmed"] == 0
    assert result["A"]["supporters"] == []
    assert result["A"]["contradictors"] == []


# 3. Node with confirmed supporter → supports_from_confirmed == 1
def test_confirmed_supporter_increments():
    nodes = {
        "SRC": _confirmed_node(grade="moderate"),
        "TGT": _unconfirmed_node(),
    }
    edges = [("SRC", "supports", "TGT")]
    result = coherence_compute(nodes=nodes, edges=edges)
    assert result["TGT"]["supports_from_confirmed"] == 1
    assert "SRC" in result["TGT"]["supporters"]


# 4. Unconfirmed supporter → counts stay 0
def test_unconfirmed_supporter_no_increment():
    nodes = {
        "SRC": _unconfirmed_node(grade="low"),
        "TGT": _unconfirmed_node(),
    }
    edges = [("SRC", "supports", "TGT")]
    result = coherence_compute(nodes=nodes, edges=edges)
    assert result["TGT"]["supports_from_confirmed"] == 0
    assert result["TGT"]["supporters"] == []


# 5. Grade threshold: unverified source not confirmed even with independent source
def test_unverified_grade_not_confirmed():
    nodes = {
        "SRC": {
            "is_ku": True,
            "epistemic_status": {
                "completeness": {
                    "grade": "unverified",
                    "source": "reproducible_empirical",
                    "defeaters": [],
                }
            },
        },
        "TGT": _unconfirmed_node(),
    }
    edges = [("SRC", "supports", "TGT")]
    result = coherence_compute(nodes=nodes, edges=edges)
    assert result["TGT"]["supports_from_confirmed"] == 0


# 6. moderate + independent → confirmed
def test_moderate_independent_is_confirmed():
    nodes = {
        "SRC": _confirmed_node(grade="moderate", source="reproducible_empirical"),
        "TGT": _unconfirmed_node(),
    }
    edges = [("SRC", "supports", "TGT")]
    result = coherence_compute(nodes=nodes, edges=edges)
    assert result["TGT"]["supports_from_confirmed"] == 1


# 7. Contradictor from confirmed increases contradicts count
def test_confirmed_contradictor_increments():
    nodes = {
        "SRC": _confirmed_node(grade="high"),
        "TGT": _unconfirmed_node(),
    }
    edges = [("SRC", "contradicts", "TGT")]
    result = coherence_compute(nodes=nodes, edges=edges)
    assert result["TGT"]["contradicts_from_confirmed"] == 1
    assert "SRC" in result["TGT"]["contradictors"]


# 8. supporters/contradictors lists populated correctly
def test_lists_populated():
    nodes = {
        "S1": _confirmed_node(),
        "S2": _confirmed_node(),
        "TGT": _unconfirmed_node(),
    }
    edges = [("S1", "supports", "TGT"), ("S2", "supports", "TGT")]
    result = coherence_compute(nodes=nodes, edges=edges)
    assert sorted(result["TGT"]["supporters"]) == ["S1", "S2"]
    assert result["TGT"]["supports_from_confirmed"] == 2


# 9. Multiple supporters accumulate
def test_multiple_supporters_accumulate():
    nodes = {
        "S1": _confirmed_node(),
        "S2": _confirmed_node(),
        "S3": _confirmed_node(),
        "TGT": _unconfirmed_node(),
    }
    edges = [
        ("S1", "supports", "TGT"),
        ("S2", "supports", "TGT"),
        ("S3", "supports", "TGT"),
    ]
    result = coherence_compute(nodes=nodes, edges=edges)
    assert result["TGT"]["supports_from_confirmed"] == 3


# 10. INDEPENDENT_SOURCES contains 'reproducible_empirical'
def test_independent_sources_contains_reproducible_empirical():
    assert "reproducible_empirical" in INDEPENDENT_SOURCES


# 11. GRADE_LADDER ordering: unverified < low < moderate < high < proven
def test_grade_ladder_ordering():
    assert _grade_index("unverified") < _grade_index("low")
    assert _grade_index("low") < _grade_index("moderate")
    assert _grade_index("moderate") < _grade_index("high")
    assert _grade_index("high") < _grade_index("proven")


# 12. Edge to unknown node is silently ignored
def test_edge_to_unknown_node_ignored():
    nodes = {"SRC": _confirmed_node()}
    edges = [("SRC", "supports", "UNKNOWN")]
    result = coherence_compute(nodes=nodes, edges=edges)
    assert "UNKNOWN" not in result
    # SRC itself untouched
    assert result["SRC"]["supports_from_confirmed"] == 0


# 13. Non-independent source even at high grade: not confirmed
def test_non_independent_source_not_confirmed():
    nodes = {
        "SRC": {
            "is_ku": True,
            "epistemic_status": {
                "completeness": {
                    "grade": "high",
                    "source": "expert_opinion",  # not in INDEPENDENT_SOURCES
                    "defeaters": [],
                }
            },
        },
        "TGT": _unconfirmed_node(),
    }
    edges = [("SRC", "supports", "TGT")]
    result = coherence_compute(nodes=nodes, edges=edges)
    assert result["TGT"]["supports_from_confirmed"] == 0
