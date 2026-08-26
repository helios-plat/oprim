"""CodeGraph MCP-based full-text search index."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Any

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


class CodeGraphMCPClient:
    """MCP client for communicating with CodeGraph server."""

    def __init__(self, mcp_url: str = "http://localhost:8080/mcp") -> None:
        self._mcp_url = mcp_url.rstrip("/")
        self._session = None

    def _get_session(self):
        import httpx
        if self._session is None:
            self._session = httpx.Client(timeout=30.0)
        return self._session

    def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Call an MCP tool via HTTP."""
        import json
        session = self._get_session()
        try:
            # MCP over HTTP - tools/call endpoint
            response = session.post(
                f"{self._mcp_url}/tools/call",
                json={"name": method, "arguments": params},
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            olog.error("codegraph_mcp_call_failed", method=method, error=str(e))
            raise FulltextError(f"MCP call failed: {e}") from e

    def explore(self, query: str, top_k: int = 20, paths: list[str] | None = None) -> list[FulltextHit]:
        """Query CodeGraph for relevant code snippets."""
        result = self.call("codegraph_explore", {
            "query": query,
            "limit": top_k,
            "paths": paths or [],
        })

        hits: list[FulltextHit] = []
        for item in result.get("results", []):
            # CodeGraph returns: {path, symbol, content, score, ...}
            hits.append(FulltextHit(
                id=item.get("path", item.get("symbol", "")),
                score=item.get("score", 1.0),
                highlight=item.get("content", "")[:500] if item.get("content") else None,
            ))
        return hits

    def search(self, query: str, top_k: int = 20, paths: list[str] | None = None) -> list[FulltextHit]:
        """Full-text search via CodeGraph."""
        result = self.call("codegraph_search", {
            "query": query,
            "limit": top_k,
            "paths": paths or [],
        })

        hits: list[FulltextHit] = []
        for item in result.get("results", []):
            hits.append(FulltextHit(
                id=item.get("path", item.get("symbol", "")),
                score=item.get("score", 1.0),
                highlight=item.get("content", "")[:500] if item.get("content") else None,
            ))
        return hits

    def close(self) -> None:
        if self._session:
            self._session.close()
            self._session = None


class CodeGraphFulltextIndex:
    """Full-text index backed by CodeGraph MCP server."""

    def __init__(
        self,
        mcp_url: str = "http://localhost:8080/mcp",
        project_path: str | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize CodeGraph full-text index via MCP.

        Args:
            mcp_url: CodeGraph MCP server URL
            project_path: Project path for context (optional)
            **kwargs: Additional arguments
        """
        self._mcp_url = mcp_url
        self._project_path = project_path
        self._client = CodeGraphMCPClient(mcp_url)

    def add(self, docs: list[FulltextDoc]) -> None:
        """CodeGraph is read-only (auto-synced), add is no-op but logged."""
        olog.emit("codegraph_add_noop", count=len(docs), note="CodeGraph auto-syncs on file changes")

    def search(
        self,
        query: str,
        top_k: int = 20,
        fields: list[str] | None = None,
    ) -> list[FulltextHit]:
        """Search CodeGraph for relevant code."""
        try:
            # CodeGraph's explore is more powerful than simple search
            paths = fields or []
            hits = self._client.explore(query, top_k=top_k, paths=paths)
            olog.emit("codegraph_search", query=query, results=len(hits))
            return hits
        except Exception as e:
            olog.error("codegraph_search failed", error=str(e))
            raise FulltextError(f"CodeGraph search failed: {e}") from e

    def delete(self, ids: list[str]) -> None:
        """CodeGraph is read-only, delete is no-op."""
        olog.emit("codegraph_delete_noop", count=len(ids), note="CodeGraph manages its own index")

    def close(self) -> None:
        self._client.close()


def open_codegraph_index(
    mcp_url: str = "http://localhost:8080/mcp",
    project_path: str | None = None,
    **kwargs,
) -> CodeGraphFulltextIndex:
    """Open a CodeGraph full-text index via MCP."""
    return CodeGraphFulltextIndex(mcp_url=mcp_url, project_path=project_path, **kwargs)