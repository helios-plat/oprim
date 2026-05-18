"""Integration tests for oprim.vector_db.lancedb (uses tmp_path)."""
from __future__ import annotations

import random
from pathlib import Path

import pytest

from oprim.vector_db.lancedb import LanceDBVectorDB, VectorRecord, open_vector_db
from oprim.errors import VectorDBError

DIM = 16  # small dimension for fast tests


def _rand_vec(dim: int = DIM) -> list[float]:
    return [random.random() for _ in range(dim)]


class TestLanceDBVectorDB:
    def test_create_and_count_empty(self, tmp_path: Path):
        db = open_vector_db(tmp_path / "db", "test_table", dim=DIM)
        assert db.count() == 0

    def test_upsert_and_count(self, tmp_path: Path):
        db = open_vector_db(tmp_path / "db", "test_table", dim=DIM)
        records = [
            VectorRecord(id=f"doc_{i}", embedding=_rand_vec(), metadata={"n": i})
            for i in range(5)
        ]
        db.upsert(records)
        assert db.count() == 5

    def test_upsert_empty_list_noop(self, tmp_path: Path):
        db = open_vector_db(tmp_path / "db", "test_table", dim=DIM)
        db.upsert([])  # must not raise
        assert db.count() == 0

    def test_search_returns_results(self, tmp_path: Path):
        db = open_vector_db(tmp_path / "db", "test_table", dim=DIM)
        records = [
            VectorRecord(id=f"doc_{i}", embedding=_rand_vec(), metadata={"idx": i})
            for i in range(10)
        ]
        db.upsert(records)
        query = _rand_vec()
        results = db.search(query, top_k=5)
        assert 1 <= len(results) <= 5
        assert all(isinstance(r, VectorRecord) for r in results)
        assert all(r.id.startswith("doc_") for r in results)

    def test_search_respects_top_k(self, tmp_path: Path):
        db = open_vector_db(tmp_path / "db", "test_table", dim=DIM)
        records = [
            VectorRecord(id=f"doc_{i}", embedding=_rand_vec(), metadata={})
            for i in range(20)
        ]
        db.upsert(records)
        results = db.search(_rand_vec(), top_k=3)
        assert len(results) <= 3

    def test_metadata_round_trips(self, tmp_path: Path):
        db = open_vector_db(tmp_path / "db", "test_table", dim=DIM)
        meta = {"title": "test doc", "score": 0.9, "tags": ["a", "b"]}
        db.upsert([VectorRecord(id="doc_1", embedding=_rand_vec(), metadata=meta)])
        results = db.search(_rand_vec(), top_k=10)
        found = next((r for r in results if r.id == "doc_1"), None)
        assert found is not None
        assert found.metadata.get("title") == "test doc"

    def test_upsert_updates_existing(self, tmp_path: Path):
        db = open_vector_db(tmp_path / "db", "test_table", dim=DIM)
        vec = _rand_vec()
        db.upsert([VectorRecord(id="doc_1", embedding=vec, metadata={"v": 1})])
        assert db.count() == 1
        new_vec = _rand_vec()
        db.upsert([VectorRecord(id="doc_1", embedding=new_vec, metadata={"v": 2})])
        assert db.count() == 1  # not duplicated

    def test_delete_removes_records(self, tmp_path: Path):
        db = open_vector_db(tmp_path / "db", "test_table", dim=DIM)
        db.upsert([VectorRecord(id=f"doc_{i}", embedding=_rand_vec(), metadata={}) for i in range(5)])
        assert db.count() == 5
        db.delete(["doc_0", "doc_1"])
        assert db.count() == 3

    def test_delete_empty_list_noop(self, tmp_path: Path):
        db = open_vector_db(tmp_path / "db", "test_table", dim=DIM)
        db.upsert([VectorRecord(id="doc_1", embedding=_rand_vec(), metadata={})])
        db.delete([])  # must not raise
        assert db.count() == 1

    def test_persistence(self, tmp_path: Path):
        """Data written in one instance is readable in a new instance."""
        db_path = tmp_path / "db"
        db1 = LanceDBVectorDB(db_path, "test_table", dim=DIM)
        db1.upsert([VectorRecord(id="doc_1", embedding=_rand_vec(), metadata={"persistent": True})])

        db2 = LanceDBVectorDB(db_path, "test_table", dim=DIM)
        assert db2.count() == 1

    def test_unknown_provider_raises(self, tmp_path: Path):
        with pytest.raises(VectorDBError, match="Unknown vector DB provider"):
            open_vector_db(tmp_path / "db", "table", dim=DIM, provider="qdrant")

    def test_upsert_error_raises_vectordberror(self, tmp_path: Path):
        from unittest.mock import patch, MagicMock
        db = open_vector_db(tmp_path / "db", "test_table", dim=DIM)
        records = [VectorRecord(id="r1", embedding=_rand_vec(), metadata={})]
        # Mock the internal table merge_insert to raise
        with patch.object(db._tbl, "merge_insert", side_effect=RuntimeError("lancedb boom")):
            with pytest.raises(VectorDBError, match="Upsert failed"):
                db.upsert(records)

    def test_search_error_raises_vectordberror(self, tmp_path: Path):
        from unittest.mock import patch
        db = open_vector_db(tmp_path / "db", "test_table", dim=DIM)
        with patch.object(db._tbl, "search", side_effect=RuntimeError("search boom")):
            with pytest.raises(VectorDBError, match="Search failed"):
                db.search(_rand_vec())

    def test_delete_error_raises_vectordberror(self, tmp_path: Path):
        from unittest.mock import patch
        db = open_vector_db(tmp_path / "db", "test_table", dim=DIM)
        db.upsert([VectorRecord(id="r1", embedding=_rand_vec(), metadata={})])
        with patch.object(db._tbl, "delete", side_effect=RuntimeError("delete boom")):
            with pytest.raises(VectorDBError, match="Delete failed"):
                db.delete(["r1"])

    def test_count_error_raises_vectordberror(self, tmp_path: Path):
        from unittest.mock import patch
        db = open_vector_db(tmp_path / "db", "test_table", dim=DIM)
        with patch.object(db._tbl, "count_rows", side_effect=RuntimeError("count boom")):
            with pytest.raises(VectorDBError, match="Count failed"):
                db.count()
