"""oprim.multimodal_embed — single multimodal embedding call (atomic)."""

from __future__ import annotations

from typing import Any


class MultimodalEmbedError(RuntimeError):
    pass


async def multimodal_embed(
    *,
    text: str | None = None,
    image_path: str | None = None,
    code: str | None = None,
    client: Any = None,
    base_url: str = "http://localhost:11434",
    model: str = "embeddinggemma",
) -> list[float]:
    if client is None:
        try:
            from obase.embedding_gemma import get_embedding_gemma_client
        except Exception as exc:
            raise MultimodalEmbedError("embeddinggemma client unavailable") from exc
        client = get_embedding_gemma_client(base_url=base_url, model=model)
    try:
        return await client.embed(text=text, image_path=image_path, code=code)
    except Exception as exc:
        raise MultimodalEmbedError(f"embed failed: {exc}") from exc
