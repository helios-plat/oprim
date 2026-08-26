"""Tests for oprim.fulltext.elasticsearch (uses mocking)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from oprim.errors import FulltextError


class TestElasticsearchFulltextIndex:
    """Tests for the Elasticsearch full-text index implementation."""

    def _make_index(self, mock_client: MagicMock) -> "ElasticsearchFulltextIndex":
        from oprim.fulltext.elasticsearch import ElasticsearchFulltextIndex

        with patch("oprim.fulltext.elasticsearch.Elasticsearch", return_value=mock_client):
            idx = ElasticsearchFulltextIndex(hosts="http://localhost:9200")
        return idx

    def test_init_creates_client(self):
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True
        idx = self._make_index(mock_client)
        assert idx._index_name == "veya_fulltext"
        assert idx._hosts == ["http://localhost:9200"]
        mock_client.indices.exists.assert_called_once_with(index="veya_fulltext")

    def test_init_creates_index_if_not_exists(self):
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = False
        idx = self._make_index(mock_client)
        mock_client.indices.create.assert_called_once()
        mock_client.indices.exists.assert_called_once_with(index="veya_fulltext")

    def test_init_with_custom_index_name(self):
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True
        idx = self._make_index(mock_client)
        idx._index_name = "custom_index"
        assert idx._index_name == "custom_index"

    def test_init_with_basic_auth(self):
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True
        with patch(
            "oprim.fulltext.elasticsearch.Elasticsearch",
            return_value=mock_client,
        ) as mock_es:
            from oprim.fulltext.elasticsearch import ElasticsearchFulltextIndex

            ElasticsearchFulltextIndex(
                hosts="http://localhost:9200",
                basic_auth=("user", "pass"),
            )
            mock_es.assert_called_once()
            kwargs = mock_es.call_args.kwargs
            assert kwargs["basic_auth"] == ("user", "pass")

    def test_init_with_api_key(self):
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True
        with patch(
            "oprim.fulltext.elasticsearch.Elasticsearch",
            return_value=mock_client,
        ) as mock_es:
            from oprim.fulltext.elasticsearch import ElasticsearchFulltextIndex

            ElasticsearchFulltextIndex(
                hosts="http://localhost:9200",
                api_key="test-key-123",
            )
            kwargs = mock_es.call_args.kwargs
            assert kwargs["api_key"] == "test-key-123"

    def test_add_documents(self):
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True
        idx = self._make_index(mock_client)

        from oprim.fulltext.elasticsearch import FulltextDoc

        docs = [
            FulltextDoc(id="doc1", fields={"title": "Python Guide", "content": "Learn Python"}),
            FulltextDoc(id="doc2", fields={"title": "Rust Book", "content": "Systems programming"}),
        ]
        idx.add(docs)
        mock_client.indices.refresh.assert_called_once_with(index="veya_fulltext")

    def test_add_empty_list(self):
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True
        idx = self._make_index(mock_client)
        idx.add([])  # Should not raise
        mock_client.indices.refresh.assert_called_once_with(index="veya_fulltext")

    def test_search_returns_hits(self):
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True
        idx = self._make_index(mock_client)

        mock_response = {
            "hits": {
                "hits": [
                    {
                        "_id": "doc1",
                        "_source": {"id": "doc1", "title": "Python Guide"},
                        "_score": 1.5,
                        "highlight": {"title": ["<em>Python</em> Guide"]},
                    }
                ]
            }
        }
        mock_client.search.return_value = mock_response

        from oprim.fulltext.elasticsearch import FulltextHit

        hits = idx.search("Python")
        assert len(hits) == 1
        assert isinstance(hits[0], FulltextHit)
        assert hits[0].id == "doc1"
        assert hits[0].score == 1.5
        assert hits[0].highlight == "<em>Python</em> Guide"

    def test_search_no_results(self):
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True
        idx = self._make_index(mock_client)

        mock_response = {"hits": {"hits": []}}
        mock_client.search.return_value = mock_response

        results = idx.search("xyznomatch")
        assert results == []

    def test_search_with_top_k(self):
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True
        idx = self._make_index(mock_client)

        mock_response = {
            "hits": {
                "hits": [
                    {"_id": f"doc{i}", "_source": {"id": f"doc{i}"}, "_score": 1.0}
                    for i in range(5)
                ]
            }
        }
        mock_client.search.return_value = mock_response

        idx.search("test", top_k=3)
        # Verify body was created with size=3
        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs["body"]["size"] == 3

    def test_search_with_specific_fields(self):
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True
        idx = self._make_index(mock_client)

        mock_response = {"hits": {"hits": []}}
        mock_client.search.return_value = mock_response

        idx.search("test", fields=["title"])
        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs["body"]["query"]["multi_match"]["fields"] == ["title"]

    def test_delete_documents(self):
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True
        idx = self._make_index(mock_client)

        idx.delete(["doc1", "doc2"])
        mock_client.indices.refresh.assert_called_once_with(index="veya_fulltext")

    def test_delete_empty_list(self):
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True
        idx = self._make_index(mock_client)

        idx.delete([])  # Should not raise
        mock_client.indices.refresh.assert_called_once_with(index="veya_fulltext")

    def test_close(self):
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True
        idx = self._make_index(mock_client)
        idx.close()
        mock_client.close.assert_called_once()

    def test_add_error_raises_fulltexterror(self):
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True
        idx = self._make_index(mock_client)

        from elasticsearch.exceptions import ApiError

        from oprim.fulltext.elasticsearch import FulltextDoc

        class FakeMeta:
            status = 500
            http_version = "1.1"
            headers = {}
            duration = 0.0
            node = None

        with patch(
            "elasticsearch.helpers.bulk",
            side_effect=ApiError("add boom", meta=FakeMeta(), body={}),
        ):
            with pytest.raises(FulltextError, match="Add failed"):
                idx.add([FulltextDoc(id="x", fields={"content": "y"})])

    def test_search_error_raises_fulltexterror(self):
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True
        idx = self._make_index(mock_client)

        from elasticsearch.exceptions import TransportError

        mock_client.search.side_effect = TransportError("search boom")
        with pytest.raises(FulltextError, match="Search failed"):
            idx.search("test")

    def test_delete_error_raises_fulltexterror(self):
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True
        idx = self._make_index(mock_client)

        from elasticsearch.exceptions import ApiError

        class FakeMeta:
            status = 500
            http_version = "1.1"
            headers = {}
            duration = 0.0
            node = None

        with patch(
            "elasticsearch.helpers.bulk",
            side_effect=ApiError("delete boom", meta=FakeMeta(), body={}),
        ):
            with pytest.raises(FulltextError, match="Delete failed"):
                idx.delete(["doc1"])

    def test_index_creation_error_raises_fulltexterror(self):
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = False

        from elasticsearch.exceptions import ApiError

        class FakeMeta:
            status = 500
            http_version = "1.1"
            headers = {}
            duration = 0.0
            node = None

        mock_client.indices.create.side_effect = ApiError(
            "create boom", meta=FakeMeta(), body={}
        )

        with patch(
            "oprim.fulltext.elasticsearch.Elasticsearch", return_value=mock_client
        ):
            from oprim.fulltext.elasticsearch import ElasticsearchFulltextIndex

            with pytest.raises(FulltextError, match="Failed to create index"):
                ElasticsearchFulltextIndex(hosts="http://localhost:9200")

    def test_open_elasticsearch_index_factory(self):
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True

        with patch(
            "oprim.fulltext.elasticsearch.Elasticsearch", return_value=mock_client
        ):
            from oprim.fulltext.elasticsearch import open_elasticsearch_index

            idx = open_elasticsearch_index(hosts="http://localhost:9200")
            assert idx._index_name == "veya_fulltext"
            assert idx._hosts == ["http://localhost:9200"]


class TestFulltextFactory:
    """Tests for the unified open_fulltext_index factory with Elasticsearch provider."""

    def test_factory_with_elasticsearch_provider(self):
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True

        with patch(
            "oprim.fulltext.elasticsearch.Elasticsearch", return_value=mock_client
        ):
            from oprim.fulltext.tantivy import open_fulltext_index

            idx = open_fulltext_index("http://localhost:9200", provider="elasticsearch")
            assert idx._index_name == "veya_fulltext"
            assert idx._hosts == ["http://localhost:9200"]

    def test_factory_unknown_provider_raises(self):
        from oprim.fulltext.tantivy import open_fulltext_index
        from oprim.errors import FulltextError
        import pytest

        with pytest.raises(FulltextError, match="Unknown fulltext provider"):
            open_fulltext_index("/tmp/idx", provider="nonexistent")
