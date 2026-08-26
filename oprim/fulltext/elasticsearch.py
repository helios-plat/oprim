"""Elasticsearch-based full-text search index."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from elasticsearch import AsyncElasticsearch, Elasticsearch
from elasticsearch.exceptions import ApiError, TransportError

from oprim._logging import log as olog
from oprim.errors import FulltextError


@dataclass
class FulltextDoc:
    id: str
    fields: dict[str, str]


@dataclass
class FulltextHit:
    id: str
    score: float
    highlight: str | None


class FulltextIndex(Protocol):
    def add(self, docs: list[FulltextDoc]) -> None: ...
    def search(
        self,
        query: str,
        top_k: int = 20,
        fields: list[str] | None = None,
    ) -> list[FulltextHit]: ...
    def delete(self, ids: list[str]) -> None: ...


class ElasticsearchFulltextIndex:
    """Full-text index backed by Elasticsearch with id/title/content/tags fields."""

    DEFAULT_INDEX_NAME = "veya_fulltext"

    def __init__(
        self,
        hosts: list[str] | str = "http://localhost:9200",
        index_name: str = DEFAULT_INDEX_NAME,
        basic_auth: tuple[str, str] | None = None,
        api_key: str | None = None,
        verify_certs: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize Elasticsearch full-text index.

        Args:
            hosts: Elasticsearch host(s), e.g., ["http://localhost:9200"] or "http://localhost:9200"
            index_name: Name of the Elasticsearch index to use
            basic_auth: Tuple of (username, password) for basic auth
            api_key: API key for authentication
            verify_certs: Whether to verify SSL certificates
            **kwargs: Additional arguments passed to Elasticsearch client
        """
        if isinstance(hosts, str):
            hosts = [hosts]

        self._hosts = hosts
        self._index_name = index_name
        self._client = Elasticsearch(
            hosts=hosts,
            basic_auth=basic_auth,
            api_key=api_key,
            verify_certs=verify_certs,
            **kwargs,
        )
        self._async_client = None

        # Create index with mapping if it doesn't exist
        self._ensure_index_exists()

    def _ensure_index_exists(self) -> None:
        """Create index with proper mapping if it doesn't exist."""
        if not self._client.indices.exists(index=self._index_name):
            mapping = {
                "mappings": {
                    "properties": {
                        "id": {"type": "keyword"},
                        "title": {"type": "text", "analyzer": "standard"},
                        "content": {"type": "text", "analyzer": "standard"},
                        "tags": {"type": "text", "analyzer": "standard"},
                    }
                }
            }
            try:
                self._client.indices.create(index=self._index_name, **mapping)
                olog.emit("fulltext_index_created", index=self._index_name)
            except (ApiError, TransportError) as e:
                raise FulltextError(f"Failed to create index {self._index_name}: {e}") from e

    def add(self, docs: list[FulltextDoc]) -> None:
        """Add documents to the index."""
        try:
            from elasticsearch.helpers import bulk

            actions = []
            for doc in docs:
                action = {
                    "_op_type": "index",
                    "_index": self._index_name,
                    "_id": doc.id,
                    "_source": {
                        "id": doc.id,
                        "title": doc.fields.get("title", ""),
                        "content": doc.fields.get("content", ""),
                        "tags": doc.fields.get("tags", ""),
                    },
                }
                actions.append(action)

            bulk(self._client, actions)
            self._client.indices.refresh(index=self._index_name)
            olog.emit("fulltext_add", count=len(docs))
        except (ApiError, TransportError) as e:
            olog.error("fulltext_add failed", error=str(e))
            raise FulltextError(f"Add failed: {e}") from e

    def search(
        self,
        query: str,
        top_k: int = 20,
        fields: list[str] | None = None,
    ) -> list[FulltextHit]:
        """Search for documents matching the query."""
        try:
            search_fields = fields or ["title", "content", "tags"]

            # Build multi-match query
            body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": search_fields,
                        "type": "best_fields",
                    }
                },
                "size": top_k,
                "highlight": {
                    "fields": {
                        "title": {},
                        "content": {},
                        "tags": {},
                    }
                },
            }

            response = self._client.search(index=self._index_name, body=body)

            hits: list[FulltextHit] = []
            for hit in response["hits"]["hits"]:
                doc_id = hit["_source"].get("id", hit["_id"])
                score = hit["_score"] or 0.0

                # Extract highlight if available
                highlight = None
                if "highlight" in hit:
                    highlight_parts = []
                    for field in search_fields:
                        if field in hit["highlight"]:
                            highlight_parts.extend(hit["highlight"][field])
                    if highlight_parts:
                        highlight = " ... ".join(highlight_parts)

                hits.append(FulltextHit(id=doc_id, score=score, highlight=highlight))

            return hits
        except (ApiError, TransportError) as e:
            olog.error("fulltext_search failed", error=str(e))
            raise FulltextError(f"Search failed: {e}") from e

    def delete(self, ids: list[str]) -> None:
        """Delete documents by IDs."""
        try:
            from elasticsearch.helpers import bulk

            actions = [
                {"_op_type": "delete", "_index": self._index_name, "_id": doc_id}
                for doc_id in ids
            ]

            bulk(self._client, actions)
            self._client.indices.refresh(index=self._index_name)
            olog.emit("fulltext_delete", count=len(ids))
        except (ApiError, TransportError) as e:
            olog.error("fulltext_delete failed", error=str(e))
            raise FulltextError(f"Delete failed: {e}") from e

    def close(self) -> None:
        """Close the Elasticsearch client connections."""
        self._client.close()
        if self._async_client:
            import asyncio

            asyncio.run(self._async_client.close())

    @property
    def client(self) -> Elasticsearch:
        """Return the synchronous Elasticsearch client."""
        return self._client


def open_elasticsearch_index(
    hosts: list[str] | str = "http://localhost:9200",
    index_name: str = "veya_fulltext",
    **kwargs,
) -> ElasticsearchFulltextIndex:
    """Open or create an Elasticsearch full-text index."""
    return ElasticsearchFulltextIndex(hosts=hosts, index_name=index_name, **kwargs)