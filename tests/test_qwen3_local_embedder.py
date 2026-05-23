"""Unit tests for Qwen3LocalEmbedder (mocked Ollama HTTP)."""
from __future__ import annotations

import pytest
import respx
import httpx

from oprim.embedding.qwen3_local import Qwen3LocalEmbedder
from oprim.errors import EmbeddingError


@pytest.fixture()
def embedder(monkeypatch):
    import oprim._config as _cfg
    _cfg._store["OLLAMA_BASE_URL"] = "http://test-ollama:11434"
    yield Qwen3LocalEmbedder()
    _cfg._store.pop("OLLAMA_BASE_URL", None)


@respx.mock
def test_embed_single_text(embedder):
    fake_vec = [0.1] * 1024
    respx.post("http://test-ollama:11434/api/embeddings").mock(
        return_value=httpx.Response(200, json={"embedding": fake_vec})
    )
    result = embedder.embed(["hello"], dim=1024)
    assert len(result) == 1
    assert len(result[0]) == 1024
    assert result[0][0] == pytest.approx(0.1)


@respx.mock
def test_embed_multiple_texts(embedder):
    fake_vec = [0.5] * 1024
    respx.post("http://test-ollama:11434/api/embeddings").mock(
        return_value=httpx.Response(200, json={"embedding": fake_vec})
    )
    result = embedder.embed(["a", "b", "c"], dim=1024)
    assert len(result) == 3


@respx.mock
def test_embed_dim_truncation(embedder):
    fake_vec = [0.2] * 1024
    respx.post("http://test-ollama:11434/api/embeddings").mock(
        return_value=httpx.Response(200, json={"embedding": fake_vec})
    )
    result = embedder.embed(["test"], dim=512)
    assert len(result[0]) == 512


@respx.mock
def test_embed_retries_on_failure(embedder):
    route = respx.post("http://test-ollama:11434/api/embeddings")
    route.side_effect = [
        httpx.ConnectError("refused"),
        httpx.Response(200, json={"embedding": [0.3] * 1024}),
    ]
    result = embedder.embed(["retry test"], dim=1024)
    assert len(result) == 1
    assert route.call_count == 2


@respx.mock
def test_embed_raises_after_3_retries(embedder):
    respx.post("http://test-ollama:11434/api/embeddings").mock(
        side_effect=httpx.ConnectError("refused")
    )
    with pytest.raises(EmbeddingError, match="3 retries"):
        embedder.embed(["fail"])


@respx.mock
def test_embed_raises_on_empty_embedding(embedder):
    respx.post("http://test-ollama:11434/api/embeddings").mock(
        return_value=httpx.Response(200, json={"embedding": []})
    )
    with pytest.raises(EmbeddingError, match="empty embedding"):
        embedder.embed(["empty"])


def test_model_name(embedder):
    assert embedder.model_name == "qwen3-embedding:0.6b"


def test_native_dim(embedder):
    assert embedder.native_dim == 1024
