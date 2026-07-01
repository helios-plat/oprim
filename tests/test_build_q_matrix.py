"""Tests for oprim.build_q_matrix — ≥12 tests."""

from __future__ import annotations

import numpy as np
import pytest

from oprim.build_q_matrix import build_q_matrix


# ---------------------------------------------------------------------------
# 1. Empty edges → Q shape (0, 0)
# ---------------------------------------------------------------------------


def test_empty_edges_shape():
    result = build_q_matrix(edges=[])
    assert result["Q"].shape == (0, 0)


# ---------------------------------------------------------------------------
# 2. Non-assesses edges are ignored
# ---------------------------------------------------------------------------


def test_non_assesses_edges_ignored():
    edges = [("item1", "related_to", "skill1"), ("item1", "requires", "skill2")]
    result = build_q_matrix(edges=edges)
    assert result["Q"].shape == (0, 0)
    assert result["item_ids"] == []
    assert result["skill_ids"] == []


# ---------------------------------------------------------------------------
# 3. Single assesses edge → Q[0,0] = 1
# ---------------------------------------------------------------------------


def test_single_assesses_edge():
    edges = [("item1", "assesses", "skill1")]
    result = build_q_matrix(edges=edges)
    assert result["Q"].shape == (1, 1)
    assert result["Q"][0, 0] == 1


# ---------------------------------------------------------------------------
# 4. Q shape (n_items, n_skills) correct
# ---------------------------------------------------------------------------


def test_q_shape_correct():
    edges = [
        ("item1", "assesses", "skill1"),
        ("item1", "assesses", "skill2"),
        ("item2", "assesses", "skill1"),
        ("item3", "assesses", "skill3"),
    ]
    result = build_q_matrix(edges=edges)
    assert result["Q"].shape == (3, 3)


# ---------------------------------------------------------------------------
# 5. Q values are 0 or 1 only
# ---------------------------------------------------------------------------


def test_q_values_binary():
    edges = [
        ("item1", "assesses", "skill1"),
        ("item2", "assesses", "skill2"),
        ("item1", "assesses", "skill2"),
    ]
    result = build_q_matrix(edges=edges)
    unique_vals = set(result["Q"].flatten().tolist())
    assert unique_vals <= {0, 1}


# ---------------------------------------------------------------------------
# 6. item_ids returned
# ---------------------------------------------------------------------------


def test_item_ids_returned():
    edges = [("item1", "assesses", "skill1"), ("item2", "assesses", "skill1")]
    result = build_q_matrix(edges=edges)
    assert set(result["item_ids"]) == {"item1", "item2"}


# ---------------------------------------------------------------------------
# 7. skill_ids returned
# ---------------------------------------------------------------------------


def test_skill_ids_returned():
    edges = [("item1", "assesses", "skillA"), ("item1", "assesses", "skillB")]
    result = build_q_matrix(edges=edges)
    assert set(result["skill_ids"]) == {"skillA", "skillB"}


# ---------------------------------------------------------------------------
# 8. item_index maps correctly
# ---------------------------------------------------------------------------


def test_item_index_maps_correctly():
    edges = [("item1", "assesses", "skill1"), ("item2", "assesses", "skill1")]
    result = build_q_matrix(edges=edges)
    for iid, idx in result["item_index"].items():
        assert result["item_ids"][idx] == iid


# ---------------------------------------------------------------------------
# 9. skill_index maps correctly
# ---------------------------------------------------------------------------


def test_skill_index_maps_correctly():
    edges = [("item1", "assesses", "skill1"), ("item1", "assesses", "skill2")]
    result = build_q_matrix(edges=edges)
    for sid, idx in result["skill_index"].items():
        assert result["skill_ids"][idx] == sid


# ---------------------------------------------------------------------------
# 10. Explicit item_ids / skill_ids respected (ordering preserved)
# ---------------------------------------------------------------------------


def test_explicit_ids_ordering_preserved():
    edges = [
        ("item1", "assesses", "skill1"),
        ("item2", "assesses", "skill2"),
    ]
    item_ids = ["item2", "item1"]
    skill_ids = ["skill2", "skill1"]
    result = build_q_matrix(edges=edges, item_ids=item_ids, skill_ids=skill_ids)
    assert result["item_ids"] == ["item2", "item1"]
    assert result["skill_ids"] == ["skill2", "skill1"]
    # item2 assesses skill2 → row 0, col 0
    assert result["Q"][0, 0] == 1
    # item1 assesses skill1 → row 1, col 1
    assert result["Q"][1, 1] == 1
    # no cross-entries
    assert result["Q"][0, 1] == 0
    assert result["Q"][1, 0] == 0


# ---------------------------------------------------------------------------
# 11. Multiple items assessing same skill
# ---------------------------------------------------------------------------


def test_multiple_items_same_skill():
    edges = [
        ("item1", "assesses", "skill1"),
        ("item2", "assesses", "skill1"),
        ("item3", "assesses", "skill1"),
    ]
    result = build_q_matrix(edges=edges)
    skill_col = result["skill_index"]["skill1"]
    for iid in ["item1", "item2", "item3"]:
        row = result["item_index"][iid]
        assert result["Q"][row, skill_col] == 1


# ---------------------------------------------------------------------------
# 12. Item assessing multiple skills
# ---------------------------------------------------------------------------


def test_item_assesses_multiple_skills():
    edges = [
        ("item1", "assesses", "skill1"),
        ("item1", "assesses", "skill2"),
        ("item1", "assesses", "skill3"),
    ]
    result = build_q_matrix(edges=edges)
    row = result["item_index"]["item1"]
    assert result["Q"][row, :].sum() == 3


# ---------------------------------------------------------------------------
# 13. Q dtype is np.int8 or int-compatible
# ---------------------------------------------------------------------------


def test_q_dtype_int8():
    edges = [("item1", "assesses", "skill1")]
    result = build_q_matrix(edges=edges)
    assert result["Q"].dtype == np.int8


# ---------------------------------------------------------------------------
# 14. Mixed edges (assesses + non-assesses) — only assesses counted
# ---------------------------------------------------------------------------


def test_mixed_edges_only_assesses_counted():
    edges = [
        ("item1", "assesses", "skill1"),
        ("item1", "requires", "skill2"),
        ("item2", "related_to", "skill1"),
    ]
    result = build_q_matrix(edges=edges)
    assert result["Q"].shape == (1, 1)
    assert result["Q"][0, 0] == 1
