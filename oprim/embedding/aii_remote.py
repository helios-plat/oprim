"""Remote embedder — HTTP client for the shared AII embed microservice.

Consumers (flywheels, aii-backend) call the always-on `aii-embed` service
instead of loading their own in-process BGE-M3. The service lazy-loads the
model (GPU when free, else CPU) and unloads on idle. Vectors are identical to
the in-process sentence-transformers path (server replicates it).

★2026-07-08: 远程服务(笔记本)挂了会让消费方(飞轮)干等到超时才报错, 进而让整个
pipeline卡死几小时(实测: 断连13小时, 4个飞轮全"active"但0产出). 加两层兜底:
①connect超时给短(远程真的不可达时快速判死, 不用等到read超时才反应); ②失败后
退回本机CPU嵌入(BgeM3Embedder, 惰性单例复用不重复加载) —— 消费方脚本本身已设
CUDA_VISIBLE_DEVICES=""(见各*_flywheel.sh), 这里退回的本地嵌入天然只会用CPU,
不违反"本机禁止用GPU做嵌入"这条规矩。宁可慢(CPU编码), 不能停。
"""

from __future__ import annotations

import os
import threading
from collections.abc import Sequence

import httpx

from oprim._logging import log as olog
from oprim.errors import EmbeddingError

_DEFAULT_URL = "http://127.0.0.1:8091"

_local_fallback = None
_local_fallback_lock = threading.Lock()


def _get_local_fallback():
    """惰性单例: 只在远程真的连不上时才第一次加载(本机进程已CUDA_VISIBLE_DEVICES=""→纯CPU)."""
    global _local_fallback
    if _local_fallback is None:
        with _local_fallback_lock:
            if _local_fallback is None:
                from oprim.embedding.bge_m3 import BgeM3Embedder

                _local_fallback = BgeM3Embedder()
    return _local_fallback


class AiiRemoteEmbedder:
    def __init__(self, base_url: str | None = None) -> None:
        self._base = (base_url or os.getenv("AII_EMBED_URL") or _DEFAULT_URL).rstrip("/")

    @property
    def model_name(self) -> str:
        return "aii-embed:BAAI/bge-m3"

    @property
    def native_dim(self) -> int:
        return 1024

    def embed(self, texts: Sequence[str], dim: int = 1024) -> list[list[float]]:
        # ★connect短超时(10s)快速判死; read给足量(300s)容纳真实大batch处理时间.
        timeout = httpx.Timeout(connect=10, read=300, write=30, pool=10)
        try:
            resp = httpx.post(f"{self._base}/embed", json={"texts": list(texts)}, timeout=timeout)
            resp.raise_for_status()
            return [v[:dim] for v in resp.json()["embeddings"]]
        except Exception as e:
            olog.warning(f"aii_remote embedding failed ({self._base}): {e} — 退回本机CPU嵌入")
            try:
                return _get_local_fallback().embed(texts, dim=dim)
            except Exception as e2:
                raise EmbeddingError(
                    f"aii_remote embedding failed ({self._base}): {e}; 本机CPU兜底也失败: {e2}"
                ) from e2
