"""oprim.vector_encode — single-call text vector encoding.

3O layer: oprim (single atomic call, delegates to embedding provider).
Production: uses obase.ProviderRegistry for embedding backend.
Test/stub: deterministic hash-based pseudo-vectors (no model needed).
"""

from __future__ import annotations

import numpy as np


def vector_encode(
    *,
    texts: list[str],
    provider: str = "default",
    normalize: bool = True,
) -> np.ndarray:
    """Encode list of texts to dense float32 vectors. Returns (n, dim) array.

    Attempts obase.ProviderRegistry first; falls back to deterministic stub.
    Stub produces hash-seeded vectors — consistent per text, usable in tests.
    """
    try:
        from obase import ProviderRegistry

        reg = ProviderRegistry.get_instance()
        embed_fn = reg.get("embedding", provider)
        vecs = np.array(embed_fn(texts), dtype=np.float32)
        if normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms
        return vecs
    except Exception:
        # deterministic stub: hash-based pseudo-vectors
        dim = 128
        result = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = abs(hash(t)) % (2**31)
            rng = np.random.default_rng(h)
            v = rng.standard_normal(dim).astype(np.float32)
            if normalize:
                v = v / (np.linalg.norm(v) + 1e-8)
            result[i] = v
        return result
