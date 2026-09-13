from oprim.embedding.embed_text import TextEmbedder, embed_text

__all__ = ["embed_text", "TextEmbedder", "Qwen3DashscopeEmbedder", "Qwen3LocalEmbedder", "BgeM3Embedder"]


def __getattr__(name: str):
    """Load provider implementations only when their public symbol is used."""
    providers = {
        "Qwen3DashscopeEmbedder": ("oprim.embedding.qwen3_dashscope", name),
        "Qwen3LocalEmbedder": ("oprim.embedding.qwen3_local", name),
        "BgeM3Embedder": ("oprim.embedding.bge_m3", name),
    }
    target = providers.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, symbol = target
    import importlib

    value = getattr(importlib.import_module(module_name), symbol)
    globals()[name] = value
    return value
