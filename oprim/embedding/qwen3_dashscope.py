"""DashScope text-embedding-v3 embedder (Qwen3 model family)."""

from __future__ import annotations

import time
from collections.abc import Sequence

from oprim._config import cfg
from oprim._logging import log as olog
from oprim._optional import require_optional
from oprim.errors import EmbeddingError, QuotaExceededError

try:
    import dashscope
except ImportError:  # optional llm extra
    dashscope = None  # type: ignore[assignment]

TextEmbedding = None


_DASHSCOPE_EMBED_MODEL = "text-embedding-v3"
_MAX_BATCH = 10  # DashScope hard limit per call
_COST_PER_1K_TOKENS = 0.0007  # approximate USD


class Qwen3DashscopeEmbedder:
    """Embed texts via DashScope text-embedding-v3 with retry and cost tracking."""

    def __init__(self) -> None:
        global TextEmbedding
        require_optional(dashscope, feature="qwen", extra="llm", package="dashscope")
        if TextEmbedding is None:
            from dashscope import TextEmbedding as _TextEmbedding

            TextEmbedding = _TextEmbedding
        self._text_embedding = TextEmbedding
        api_key = cfg.get("DASHSCOPE_API_KEY")
        if api_key:
            dashscope.api_key = str(api_key)

    @property
    def model_name(self) -> str:
        return _DASHSCOPE_EMBED_MODEL

    @property
    def native_dim(self) -> int:
        return 1024

    def embed(self, texts: Sequence[str], dim: int = 1024) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), _MAX_BATCH):
            chunk = list(texts[i : i + _MAX_BATCH])
            vectors = self._call_with_retry(chunk, dim)
            results.extend(vectors)
        return results

    def _call_with_retry(self, texts: list[str], dim: int) -> list[list[float]]:
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                resp = self._text_embedding.call(
                    model=_DASHSCOPE_EMBED_MODEL,
                    input=texts,
                    dimension=dim,
                )
                if resp.status_code == 200:
                    embeddings: list[list[float]] = [
                        item["embedding"] for item in resp.output["embeddings"]
                    ]
                    return embeddings
                elif resp.status_code == 429:
                    raise QuotaExceededError(f"DashScope quota exceeded: {resp.message}")
                else:
                    raise EmbeddingError(f"DashScope error {resp.status_code}: {resp.message}")
            except QuotaExceededError:
                raise
            except Exception as e:
                last_err = e
                if attempt < 2:
                    wait = 2**attempt
                    olog.warning(
                        "dashscope embed retry",
                        attempt=attempt,
                        error=str(e),
                        wait_s=wait,
                    )
                    time.sleep(wait)
        raise EmbeddingError(
            f"DashScope embedding failed after 3 retries: {last_err}"
        ) from last_err
