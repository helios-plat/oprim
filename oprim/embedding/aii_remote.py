"""oprim.embedding.aii_remote — embed via the shared AII embed microservice.

HTTP client for ``aii.api.embed_app`` (the aii-embed systemd unit):
POST {base}/embed {"texts": [...]} -> {"embeddings": [[...]], "dim": N, "device": dev}.

Base URL resolution order: constructor arg -> STRATUM_EMBED_URL ->
AII_EMBED_URL -> http://127.0.0.1:8102. Accepts base URLs with or without a
trailing ``/embed`` path. Uses urllib (stdlib) with a short connect timeout;
raises EmbeddingError when the service is unreachable so embedding failures
surface loudly instead of silently degrading (oskill treats embed as
best-effort and would otherwise write substrates with no vectors).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from oprim.errors import EmbeddingError

_DEFAULT_PORT = 8102


def _default_base_url() -> str:
    return (
        os.environ.get("STRATUM_EMBED_URL")
        or os.environ.get("AII_EMBED_URL")
        or f"http://127.0.0.1:{_DEFAULT_PORT}"
    )


class AiiRemoteEmbedder:
    """BGE-M3 embeddings via the shared aii-embed HTTP service (no in-process model)."""

    def __init__(self, base_url: str | None = None, timeout: float = 60.0) -> None:
        base = (base_url or _default_base_url()).rstrip("/")
        self._embed_url = base if base.endswith("/embed") else f"{base}/embed"
        self._timeout = timeout

    def embed(self, texts: list[str], dim: int = 1024) -> list[list[float]]:
        if not texts:
            return []
        body = json.dumps({"texts": list(texts)}).encode("utf-8")
        req = urllib.request.Request(
            self._embed_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise EmbeddingError(f"aii-embed unreachable at {self._embed_url}: {exc}") from exc

        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list) or not embeddings:
            raise EmbeddingError(f"aii-embed returned no embeddings from {self._embed_url}")

        vecs: list[list[float]] = []
        for vec in embeddings:
            if len(vec) >= dim:
                vecs.append(vec[:dim])
            else:
                vecs.append(vec + [0.0] * (dim - len(vec)))
        return vecs

    @property
    def model_name(self) -> str:
        return "bge-m3 (aii-remote)"

    @property
    def native_dim(self) -> int:
        return 1024
