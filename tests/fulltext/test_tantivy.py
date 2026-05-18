"""Integration tests for oprim.fulltext.tantivy (uses tmp_path)."""
from __future__ import annotations

from pathlib import Path

import pytest

from oprim.fulltext.tantivy import (
    FulltextDoc,
    FulltextHit,
    TantivyFulltextIndex,
    open_fulltext_index,
)
from oprim.errors import FulltextError


class TestTantivyFulltextIndex:
    def test_create_empty_index(self, tmp_path: Path):
        idx = open_fulltext_index(tmp_path / "idx")
        # Should not raise

    def test_add_and_search(self, tmp_path: Path):
        idx = TantivyFulltextIndex(tmp_path / "idx")
        idx.add([
            FulltextDoc(id="doc1", fields={"title": "Python Guide", "content": "Learn Python programming", "tags": "python"}),
            FulltextDoc(id="doc2", fields={"title": "Rust Book", "content": "Systems programming with Rust", "tags": "rust"}),
        ])
        results = idx.search("Python")
        assert any(r.id == "doc1" for r in results)

    def test_search_returns_hit_objects(self, tmp_path: Path):
        idx = TantivyFulltextIndex(tmp_path / "idx")
        idx.add([FulltextDoc(id="doc1", fields={"title": "Test", "content": "sample content"})])
        hits = idx.search("sample")
        assert isinstance(hits, list)
        if hits:
            assert isinstance(hits[0], FulltextHit)
            assert hasattr(hits[0], "id")
            assert hasattr(hits[0], "score")

    def test_search_no_results(self, tmp_path: Path):
        idx = TantivyFulltextIndex(tmp_path / "idx")
        idx.add([FulltextDoc(id="doc1", fields={"title": "Hello", "content": "world"})])
        results = idx.search("xyznomatch")
        assert results == []

    def test_top_k_limit(self, tmp_path: Path):
        idx = TantivyFulltextIndex(tmp_path / "idx")
        docs = [
            FulltextDoc(id=f"doc{i}", fields={"content": f"common keyword doc {i}"})
            for i in range(10)
        ]
        idx.add(docs)
        results = idx.search("keyword", top_k=3)
        assert len(results) <= 3

    def test_delete_removes_from_index(self, tmp_path: Path):
        idx = TantivyFulltextIndex(tmp_path / "idx")
        idx.add([
            FulltextDoc(id="keep", fields={"content": "keep this document"}),
            FulltextDoc(id="remove", fields={"content": "remove this document"}),
        ])
        idx.delete(["remove"])
        results = idx.search("remove")
        assert not any(r.id == "remove" for r in results)

    def test_delete_nonexistent_noop(self, tmp_path: Path):
        idx = TantivyFulltextIndex(tmp_path / "idx")
        idx.delete(["nonexistent"])  # must not raise

    def test_persistence(self, tmp_path: Path):
        idx_path = tmp_path / "idx"
        idx1 = TantivyFulltextIndex(idx_path)
        idx1.add([FulltextDoc(id="doc1", fields={"content": "persistent content"})])

        idx2 = TantivyFulltextIndex(idx_path)
        results = idx2.search("persistent")
        assert any(r.id == "doc1" for r in results)

    def test_unknown_provider_raises(self, tmp_path: Path):
        with pytest.raises(FulltextError, match="Unknown fulltext provider"):
            open_fulltext_index(tmp_path / "idx", provider="elasticsearch")

    def test_search_specific_fields(self, tmp_path: Path):
        idx = TantivyFulltextIndex(tmp_path / "idx")
        idx.add([FulltextDoc(id="doc1", fields={"title": "special", "content": "other stuff"})])
        results = idx.search("special", fields=["title"])
        assert any(r.id == "doc1" for r in results)
    def test_add_error_raises_fulltexterror(self, tmp_path: Path):
        from unittest.mock import MagicMock
        idx = TantivyFulltextIndex(tmp_path / "idx")
        mock_index = MagicMock()
        mock_index.writer.side_effect = RuntimeError("writer boom")
        idx._index = mock_index
        with pytest.raises(FulltextError, match="Add failed"):
            idx.add([FulltextDoc(id="x", fields={"content": "y"})])

    def test_search_error_raises_fulltexterror(self, tmp_path: Path):
        from unittest.mock import MagicMock
        idx = TantivyFulltextIndex(tmp_path / "idx")
        mock_index = MagicMock()
        mock_index.searcher.side_effect = RuntimeError("searcher boom")
        idx._index = mock_index
        with pytest.raises(FulltextError, match="Search failed"):
            idx.search("test")

    def test_delete_error_raises_fulltexterror(self, tmp_path: Path):
        from unittest.mock import MagicMock
        idx = TantivyFulltextIndex(tmp_path / "idx")
        mock_index = MagicMock()
        mock_index.writer.side_effect = RuntimeError("writer boom")
        idx._index = mock_index
        with pytest.raises(FulltextError, match="Delete failed"):
            idx.delete(["doc1"])

    def test_index_creation_failure_raises_fulltexterror(self, tmp_path: Path):
        from unittest.mock import patch
        import tantivy
        with patch.object(tantivy, "Index", side_effect=RuntimeError("index creation fail")):
            with pytest.raises(FulltextError, match="Failed to open tantivy index"):
                TantivyFulltextIndex(tmp_path / "idx")
