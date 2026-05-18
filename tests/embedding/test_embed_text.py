"""Tests for oprim.embedding.embed_text — mocking DashScope API."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from oprim.embedding.embed_text import embed_text
from oprim.errors import EmbeddingError, QuotaExceededError


def _make_dashscope_response(embeddings: list[list[float]], status_code: int = 200):
    """Build a mock DashScope TextEmbedding response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.output = {
        "embeddings": [{"embedding": vec} for vec in embeddings]
    }
    resp.message = "OK"
    return resp


class TestEmbedText:
    def test_empty_input_returns_empty(self):
        result = embed_text([])
        assert result == []

    def test_single_text_success(self):
        mock_vec = [0.1] * 1024
        resp = _make_dashscope_response([mock_vec])

        with patch("oprim.embedding.qwen3_dashscope.TextEmbedding.call", return_value=resp):
            result = embed_text(["hello world"], provider="qwen3_dashscope", dim=1024)

        assert len(result) == 1
        assert len(result[0]) == 1024

    def test_multiple_texts(self):
        texts = [f"text {i}" for i in range(5)]
        mock_vecs = [[float(i)] * 1024 for i in range(5)]
        resp = _make_dashscope_response(mock_vecs)

        with patch("oprim.embedding.qwen3_dashscope.TextEmbedding.call", return_value=resp):
            result = embed_text(texts, provider="qwen3_dashscope")

        assert len(result) == 5

    def test_batching_at_boundary(self):
        """11 texts should trigger 2 DashScope calls (batch_size=10 at provider level)."""
        texts = [f"text {i}" for i in range(11)]
        mock_vecs_10 = [[0.1] * 1024 for _ in range(10)]
        mock_vecs_1 = [[0.2] * 1024]

        call_count = {"n": 0}

        def side_effect(**kwargs):
            call_count["n"] += 1
            n = len(kwargs["input"])
            return _make_dashscope_response([[float(call_count["n"])] * 1024 for _ in range(n)])

        with patch("oprim.embedding.qwen3_dashscope.TextEmbedding.call", side_effect=side_effect):
            result = embed_text(texts, provider="qwen3_dashscope")

        assert len(result) == 11
        assert call_count["n"] == 2  # 10 + 1

    def test_quota_exceeded_raises(self):
        resp = MagicMock()
        resp.status_code = 429
        resp.message = "Quota exceeded"

        with patch("oprim.embedding.qwen3_dashscope.TextEmbedding.call", return_value=resp):
            with pytest.raises(QuotaExceededError):
                embed_text(["test"], provider="qwen3_dashscope")

    def test_api_error_raises_after_retries(self):
        resp = MagicMock()
        resp.status_code = 500
        resp.message = "Internal error"

        with patch("oprim.embedding.qwen3_dashscope.TextEmbedding.call", return_value=resp):
            with patch("time.sleep"):  # don't actually sleep
                with pytest.raises(EmbeddingError):
                    embed_text(["test"], provider="qwen3_dashscope")

    def test_retry_on_exception(self):
        """Exception on first two calls, success on third."""
        mock_vec = [0.5] * 1024
        success_resp = _make_dashscope_response([mock_vec])

        call_count = {"n": 0}

        def side_effect(**kwargs):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise ConnectionError("transient error")
            return success_resp

        with patch("oprim.embedding.qwen3_dashscope.TextEmbedding.call", side_effect=side_effect):
            with patch("time.sleep"):
                result = embed_text(["test"], provider="qwen3_dashscope")

        assert len(result) == 1
        assert call_count["n"] == 3

    def test_unknown_provider_raises(self):
        with pytest.raises(EmbeddingError, match="Unknown embedding provider"):
            embed_text(["test"], provider="unknown_provider")

    def test_bge_m3_unavailable_raises(self):
        """BgeM3Embedder with no backends raises EmbeddingError."""
        from oprim.embedding.bge_m3 import BgeM3Embedder

        embedder = BgeM3Embedder()
        if embedder._model is None:
            with pytest.raises(EmbeddingError):
                embedder.embed(["test"])


class TestBgeM3Embedder:
    def test_model_name_and_dim(self):
        from oprim.embedding.bge_m3 import BgeM3Embedder
        e = BgeM3Embedder()
        assert e.model_name == "BAAI/bge-m3"
        assert e.native_dim == 1024

    def test_flagembedding_path(self):
        import sys
        from unittest.mock import MagicMock, patch
        from oprim.embedding.bge_m3 import BgeM3Embedder
        import numpy as np

        mock_flag_pkg = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = {"dense_vecs": np.array([[0.1] * 1024, [0.2] * 1024])}
        mock_flag_pkg.BGEM3FlagModel.return_value = mock_model

        with patch.dict(sys.modules, {"FlagEmbedding": mock_flag_pkg}):
            e = BgeM3Embedder()
            result = e.embed(["text1", "text2"], dim=1024)

        assert len(result) == 2
        assert len(result[0]) == 1024
        assert e._use_st is False

    def test_sentence_transformers_path(self):
        import sys
        from unittest.mock import MagicMock, patch
        from oprim.embedding.bge_m3 import BgeM3Embedder
        import numpy as np

        mock_st_pkg = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.3] * 1024, [0.4] * 1024])
        mock_st_pkg.SentenceTransformer.return_value = mock_model

        with patch.dict(sys.modules, {"FlagEmbedding": None, "sentence_transformers": mock_st_pkg}):
            e = BgeM3Embedder()
            result = e.embed(["text1", "text2"], dim=512)

        assert len(result) == 2
        assert len(result[0]) == 512
        assert e._use_st is True

    def test_embed_exception_raises_embeddingerror(self):
        import sys
        from unittest.mock import MagicMock, patch
        from oprim.embedding.bge_m3 import BgeM3Embedder
        from oprim.errors import EmbeddingError

        mock_flag_pkg = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.side_effect = RuntimeError("GPU out of memory")
        mock_flag_pkg.BGEM3FlagModel.return_value = mock_model

        with patch.dict(sys.modules, {"FlagEmbedding": mock_flag_pkg}):
            e = BgeM3Embedder()
            with pytest.raises(EmbeddingError, match="bge-m3 embedding failed"):
                e.embed(["text"])
