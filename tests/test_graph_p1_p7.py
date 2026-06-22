"""Tests for P-G1 through P-G7 graph primitives."""
from __future__ import annotations

import math
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from oprim._ku_conflict_detect import ku_conflict_detect
from oprim._purpose_alignment_score import purpose_alignment_score
from oprim._source_trace import source_trace
from oprim._direct_link_score import direct_link_score
from oprim._source_overlap_score import source_overlap_score
from oprim._adamic_adar_score import adamic_adar_score
from oprim._type_affinity_score import type_affinity_score
from oprim._aii_graph_types import ConflictSignal, SourceTraceResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vec(direction: int, dim: int = 4) -> list[float]:
    v = [0.0] * dim
    v[direction % dim] = 1.0
    return v


def _similar_vec(base: list[float], noise: float = 0.01) -> list[float]:
    return [x + noise for x in base]


# ---------------------------------------------------------------------------
# P-G1: ku_conflict_detect
# ---------------------------------------------------------------------------

class TestKuConflictDetect:
    def test_high_similarity_opposing_polarity_is_candidate(self):
        a = [1.0, 0.0, 0.0, 0.0]
        b = _similar_vec(a, 0.001)
        result = ku_conflict_detect(
            ku_text_a="研究证明该药物增加血压。",
            ku_text_b="研究表明该药物减少血压。",
            embedding_a=a,
            embedding_b=b,
            similarity_threshold=0.5,
        )
        assert result.is_conflict_candidate is True
        assert result.polarity_signal == "opposing"
        assert result.evidence != ""
        assert result.similarity > 0.5

    def test_high_similarity_no_opposing_polarity_not_candidate(self):
        a = [1.0, 0.0, 0.0, 0.0]
        b = _similar_vec(a, 0.001)
        result = ku_conflict_detect(
            ku_text_a="研究表明该方法有效。",
            ku_text_b="该实验证实了这一发现。",
            embedding_a=a,
            embedding_b=b,
            similarity_threshold=0.5,
        )
        assert result.is_conflict_candidate is False
        assert result.polarity_signal in ("neutral", "insufficient")

    def test_low_similarity_immediately_not_candidate(self):
        a = [1.0, 0.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0, 0.0]
        result = ku_conflict_detect(
            ku_text_a="增加血压",
            ku_text_b="减少血压",
            embedding_a=a,
            embedding_b=b,
            similarity_threshold=0.9,
        )
        assert result.is_conflict_candidate is False
        assert result.similarity < 0.9

    def test_empty_texts_insufficient_polarity(self):
        a = [1.0, 0.0, 0.0, 0.0]
        b = _similar_vec(a, 0.001)
        result = ku_conflict_detect(
            ku_text_a="",
            ku_text_b="",
            embedding_a=a,
            embedding_b=b,
            similarity_threshold=0.5,
        )
        assert result.polarity_signal == "insufficient"
        assert result.is_conflict_candidate is False

    def test_polarity_pair_english(self):
        a = [1.0, 0.0, 0.0, 0.0]
        b = _similar_vec(a, 0.001)
        result = ku_conflict_detect(
            ku_text_a="This treatment increases recovery rate.",
            ku_text_b="This treatment decreases recovery rate.",
            embedding_a=a,
            embedding_b=b,
            similarity_threshold=0.5,
        )
        assert result.polarity_signal == "opposing"

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match="dimensions"):
            ku_conflict_detect(
                ku_text_a="a",
                ku_text_b="b",
                embedding_a=[1.0, 0.0],
                embedding_b=[1.0, 0.0, 0.0],
            )

    def test_similarity_field_accurate(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        result = ku_conflict_detect(
            ku_text_a="x",
            ku_text_b="y",
            embedding_a=a,
            embedding_b=b,
        )
        assert abs(result.similarity - 0.0) < 1e-9


# ---------------------------------------------------------------------------
# P-G2: purpose_alignment_score
# ---------------------------------------------------------------------------

class TestPurposeAlignmentScore:
    def test_identical_text_scores_near_one(self):
        v = [1.0, 0.0, 0.0]
        score = purpose_alignment_score(
            purpose_text="机器学习算法",
            ku_text="机器学习算法",
            embedding_purpose=v,
            embedding_ku=v,
        )
        assert score > 0.8

    def test_orthogonal_embeddings_low_score(self):
        score = purpose_alignment_score(
            purpose_text="机器学习",
            ku_text="量子力学",
            embedding_purpose=[1.0, 0.0],
            embedding_ku=[0.0, 1.0],
        )
        assert score < 0.5

    def test_empty_purpose_raises(self):
        with pytest.raises(ValueError):
            purpose_alignment_score(
                purpose_text="  ",
                ku_text="something",
                embedding_purpose=[1.0],
                embedding_ku=[1.0],
            )

    def test_score_between_zero_and_one(self):
        import random
        rng = random.Random(42)
        v1 = [rng.gauss(0, 1) for _ in range(8)]
        v2 = [rng.gauss(0, 1) for _ in range(8)]
        score = purpose_alignment_score(
            purpose_text="some text with keywords",
            ku_text="different text some overlap",
            embedding_purpose=v1,
            embedding_ku=v2,
        )
        assert 0.0 <= score <= 1.0

    def test_keyword_overlap_contributes(self):
        v = [1.0, 0.0]
        # Same embeddings but different keyword overlap
        s_high = purpose_alignment_score(
            purpose_text="neural network deep learning",
            ku_text="neural network architecture",
            embedding_purpose=v,
            embedding_ku=v,
        )
        s_low = purpose_alignment_score(
            purpose_text="neural network deep learning",
            ku_text="quantum entanglement physics",
            embedding_purpose=v,
            embedding_ku=v,
        )
        assert s_high > s_low


# ---------------------------------------------------------------------------
# P-G3: source_trace
# ---------------------------------------------------------------------------

class TestSourceTrace:
    async def test_returns_source_trace_result(self):
        rows = [
            {"source_id": "src1", "page": 1, "chunk_idx": 0, "text_snippet": "snippet A"},
            {"source_id": "src2", "page": 3, "chunk_idx": 2, "text_snippet": "snippet B"},
        ]
        db = MagicMock()
        db.fetch = MagicMock(return_value=rows)
        result = await source_trace(ku_id="ku001", db_conn=db)
        assert isinstance(result, SourceTraceResult)
        assert result.ku_id == "ku001"
        assert "src1" in result.source_ids
        assert "src2" in result.source_ids
        assert result.trace_depth == 2

    async def test_async_db_conn(self):
        rows = [{"source_id": "s1", "page": 0, "chunk_idx": 0, "text_snippet": "t"}]
        db = MagicMock()
        db.fetch = AsyncMock(return_value=rows)
        result = await source_trace(ku_id="ku002", db_conn=db)
        assert result.ku_id == "ku002"
        assert len(result.source_ids) == 1

    async def test_empty_result(self):
        db = MagicMock()
        db.fetch = MagicMock(return_value=[])
        result = await source_trace(ku_id="ku003", db_conn=db)
        assert result.source_ids == []
        assert result.trace_depth == 0

    async def test_db_error_returns_empty(self):
        db = MagicMock()
        db.fetch = MagicMock(side_effect=RuntimeError("db down"))
        result = await source_trace(ku_id="ku004", db_conn=db)
        assert result.source_ids == []
        assert result.trace_depth == 0

    async def test_execute_query_interface(self):
        rows = [{"source_id": "src3", "page": 2, "chunk_idx": 1, "text_snippet": "x"}]
        db = MagicMock(spec=[])
        db.execute_query = MagicMock(return_value=rows)
        result = await source_trace(ku_id="ku005", db_conn=db)
        assert "src3" in result.source_ids


# ---------------------------------------------------------------------------
# P-G4: direct_link_score
# ---------------------------------------------------------------------------

class TestDirectLinkScore:
    def test_one_edge_scores_3(self):
        edges = [{"source": "a", "target": "b"}]
        assert direct_link_score(ku_id_a="a", ku_id_b="b", edges=edges) == 3.0

    def test_three_edges_caps_at_9(self):
        edges = [
            {"source": "a", "target": "b"},
            {"source": "b", "target": "a"},
            {"source": "a", "target": "b"},
            {"source": "a", "target": "b"},  # 4 edges → capped at 9
        ]
        assert direct_link_score(ku_id_a="a", ku_id_b="b", edges=edges) == 9.0

    def test_no_edge_scores_0(self):
        edges = [{"source": "a", "target": "c"}, {"source": "x", "target": "b"}]
        assert direct_link_score(ku_id_a="a", ku_id_b="b", edges=edges) == 0.0

    def test_reverse_edge_counts(self):
        edges = [{"source": "b", "target": "a"}]
        assert direct_link_score(ku_id_a="a", ku_id_b="b", edges=edges) == 3.0


# ---------------------------------------------------------------------------
# P-G5: source_overlap_score
# ---------------------------------------------------------------------------

class TestSourceOverlapScore:
    def test_identical_sources_score_4(self):
        s = ["src1", "src2", "src3"]
        assert source_overlap_score(sources_a=s, sources_b=s) == 4.0

    def test_no_overlap_scores_0(self):
        assert source_overlap_score(sources_a=["a"], sources_b=["b"]) == 0.0

    def test_empty_both_scores_0(self):
        assert source_overlap_score(sources_a=[], sources_b=[]) == 0.0

    def test_partial_overlap(self):
        score = source_overlap_score(sources_a=["a", "b"], sources_b=["b", "c"])
        # |{a,b} ∩ {b,c}| / |{a,b,c}| * 4 = 1/3 * 4 ≈ 1.33
        assert abs(score - 4.0 / 3.0) < 1e-9


# ---------------------------------------------------------------------------
# P-G6: adamic_adar_score
# ---------------------------------------------------------------------------

class TestAdamicAdarScore:
    def test_no_common_neighbors_scores_0(self):
        assert adamic_adar_score(neighbors_a=["x"], neighbors_b=["y"], neighbor_degree={"x": 5, "y": 5}) == 0.0

    def test_common_neighbor_degree_1_skipped(self):
        assert adamic_adar_score(
            neighbors_a=["n"], neighbors_b=["n"], neighbor_degree={"n": 1}
        ) == 0.0

    def test_common_neighbor_contributes(self):
        score = adamic_adar_score(
            neighbors_a=["n"], neighbors_b=["n"], neighbor_degree={"n": 10}
        )
        expected = (1.0 / math.log(10)) * 1.5
        assert abs(score - expected) < 1e-9

    def test_multiple_common_neighbors(self):
        score = adamic_adar_score(
            neighbors_a=["n1", "n2"],
            neighbors_b=["n1", "n2"],
            neighbor_degree={"n1": 4, "n2": 8},
        )
        expected = (1 / math.log(4) + 1 / math.log(8)) * 1.5
        assert abs(score - expected) < 1e-9


# ---------------------------------------------------------------------------
# P-G7: type_affinity_score
# ---------------------------------------------------------------------------

class TestTypeAffinityScore:
    def test_same_type_returns_1(self):
        assert type_affinity_score(type_a="theorem", type_b="theorem") == 1.0

    def test_theorem_definition_returns_08(self):
        assert type_affinity_score(type_a="theorem", type_b="definition") == 0.8
        assert type_affinity_score(type_a="definition", type_b="theorem") == 0.8

    def test_example_theorem_returns_06(self):
        assert type_affinity_score(type_a="example", type_b="theorem") == 0.6

    def test_unknown_pair_returns_02(self):
        assert type_affinity_score(type_a="remark", type_b="conjecture") == 0.2

    def test_custom_matrix_overrides(self):
        matrix = {"theorem": {"proof": 0.95}}
        assert type_affinity_score(type_a="theorem", type_b="proof", affinity_matrix=matrix) == 0.95

    def test_custom_matrix_reverse_lookup(self):
        matrix = {"theorem": {"proof": 0.95}}
        assert type_affinity_score(type_a="proof", type_b="theorem", affinity_matrix=matrix) == 0.95
