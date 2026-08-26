"""Tests for oprim.fulltext.codegraph (uses mocking)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from oprim.errors import FulltextError


class TestCodeGraphFulltextIndex:
    """Tests for the CodeGraph full-text index implementation."""

    def _make_index(self, mock_client: MagicMock) -> "CodeGraphFulltextIndex":
        from oprim.fulltext.codegraph import CodeGraphFulltextIndex

        with patch("oprim.fulltext.codegraph.CodeGraphMCPClient", return_value=mock_client):
            idx = CodeGraphFulltextIndex(mcp_url="http://localhost:8080/mcp")
        return idx

    def test_init_creates_client(self):
        mock_client = MagicMock()
        idx = self._make_index(mock_client)
        assert idx._mcp_url == "http://localhost:8080/mcp"
        assert idx._project_path is None

    def test_init_with_project_path(self):
        mock_client = MagicMock()
        with patch("oprim.fulltext.codegraph.CodeGraphMCPClient", return_value=mock_client):
            from oprim.fulltext.codegraph import CodeGraphFulltextIndex
            idx = CodeGraphFulltextIndex(mcp_url="http://localhost:8080/mcp", project_path="/my/project")
        assert idx._project_path == "/my/project"

    def test_add_is_noop(self):
        mock_client = MagicMock()
        idx = self._make_index(mock_client)
        from oprim.fulltext.codegraph import FulltextDoc

        docs = [
            FulltextDoc(id="doc1", fields={"title": "Test", "content": "Content"}),
        ]
        idx.add(docs)  # Should not raise
        # No actual call to MCP for add

    def test_search_returns_hits(self):
        mock_client = MagicMock()
        mock_client.explore.return_value = [
            MagicMock(id="src/main.py", score=0.95, highlight="def main():"),
        ]
        idx = self._make_index(mock_client)

        hits = idx.search("main function")
        assert len(hits) == 1
        assert hits[0].id == "src/main.py"
        assert hits[0].score == 0.95
        assert hits[0].highlight == "def main():"

    def test_search_no_results(self):
        mock_client = MagicMock()
        mock_client.explore.return_value = []
        idx = self._make_index(mock_client)

        results = idx.search("xyznomatch")
        assert results == []

    def test_search_with_top_k(self):
        mock_client = MagicMock()
        mock_client.explore.return_value = [
            MagicMock(id=f"file{i}.py", score=1.0 - i * 0.1, highlight="code")
            for i in range(5)
        ]
        idx = self._make_index(mock_client)

        idx.search("test", top_k=3)
        # Verify explore was called with top_k
        mock_client.explore.assert_called_once()
        # Check that explore was called with the right parameters
        call_args = mock_client.explore.call_args
        assert call_args[0] == ("test",)  # Positional args
        assert call_args[1]["top_k"] == 3  # Keyword arg
        assert call_args[1]["paths"] == []

    def test_search_with_fields_as_paths(self):
        mock_client = MagicMock()
        mock_client.explore.return_value = []
        idx = self._make_index(mock_client)

        idx.search("test", fields=["src/", "tests/"])
        mock_client.explore.assert_called_once()
        call_args = mock_client.explore.call_args
        assert call_args[0] == ("test",)
        assert call_args[1]["top_k"] == 20
        assert call_args[1]["paths"] == ["src/", "tests/"]

    def test_delete_is_noop(self):
        mock_client = MagicMock()
        idx = self._make_index(mock_client)

        idx.delete(["doc1", "doc2"])  # Should not raise

    def test_close(self):
        mock_client = MagicMock()
        idx = self._make_index(mock_client)
        idx.close()
        mock_client.close.assert_called_once()

    def test_search_error_raises_fulltexterror(self):
        mock_client = MagicMock()
        mock_client.explore.side_effect = Exception("MCP connection failed")
        idx = self._make_index(mock_client)

        with pytest.raises(FulltextError, match="CodeGraph search failed"):
            idx.search("test")

    def test_open_codegraph_index_factory(self):
        mock_client = MagicMock()
        with patch("oprim.fulltext.codegraph.CodeGraphMCPClient", return_value=mock_client):
            from oprim.fulltext.codegraph import open_codegraph_index

            idx = open_codegraph_index(mcp_url="http://localhost:8080/mcp")
            assert idx._mcp_url == "http://localhost:8080/mcp"
            assert idx._project_path is None


class TestCodeGraphFactory:
    """Tests for the unified open_fulltext_index factory with CodeGraph provider."""

    def test_factory_with_codegraph_provider(self):
        mock_client = MagicMock()
        with patch("oprim.fulltext.codegraph.CodeGraphMCPClient", return_value=mock_client):
            from oprim.fulltext.tantivy import open_fulltext_index

            idx = open_fulltext_index("http://localhost:8080/mcp", provider="codegraph")
            assert idx._mcp_url == "http://localhost:8080/mcp"

    def test_factory_unknown_provider_raises(self):
        from oprim.fulltext.tantivy import open_fulltext_index
        from oprim.errors import FulltextError
        import pytest

        with pytest.raises(FulltextError, match="Unknown fulltext provider"):
            open_fulltext_index("/tmp/idx", provider="nonexistent")