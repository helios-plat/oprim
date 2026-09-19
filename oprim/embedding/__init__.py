from oprim.embedding.bge_m3 import BgeM3Embedder
from oprim.embedding.embed_text import TextEmbedder, embed_text


def __getattr__(name: str):
    if name == "Qwen3DashscopeEmbedder":
        from oprim.embedding.qwen3_dashscope import (
            Qwen3DashscopeEmbedder,
        )

        return Qwen3DashscopeEmbedder
    if name == "Qwen3LocalEmbedder":
        from oprim.embedding.qwen3_local import Qwen3LocalEmbedder

        return Qwen3LocalEmbedder
    raise AttributeError(name)


__all__ = [
    "embed_text",
    "TextEmbedder",
    "Qwen3DashscopeEmbedder",
    "Qwen3LocalEmbedder",
    "BgeM3Embedder",
]
