"""oprim.vector_search — 单次向量数据库近邻检索.

通过注入的 store（obase.cognitive_store.MemoryStore / oskill.VectorStoreHandle
兼容协议）执行近邻检索，输出标准化 dict 列表。适配两种调用形态：
- keyword-only: store.search(*, vector, top_k, filter)
- positional:   store.search(vector, *, top_k)

Example:
    >>> hits = await vector_search([0.1, 0.2, ...], store=memory_store, top_k=5)
    >>> hits[0]["chunk_id"]
    'c-42'
"""

from __future__ import annotations

import inspect
from typing import Any, Protocol, runtime_checkable

from oprim._exceptions import OprimError, OprimValidationError


class VectorSearchError(OprimError):
    """向量检索失败。"""


@runtime_checkable
class VectorSearchStore(Protocol):
    """可检索存储协议（vector_search 的注入面）。"""

    async def search(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]: ...


def _norm_row(row: Any) -> dict[str, Any]:
    """把存储返回的行归一化为 {chunk_id, content, score, path}。"""
    if isinstance(row, dict):
        out = dict(row)
        if "score" not in out and "distance" in out:
            out["score"] = 1.0 - float(out["distance"])
        out.setdefault("chunk_id", out.get("id") or out.get("chunk_id") or "")
        out.setdefault("content", out.get("text") or out.get("content") or "")
        out.setdefault("path", None)
        return out
    # tuple (chunk, score) 形态 —— MemoryStore.search 返回
    chunk, score = row[0], row[1]
    if isinstance(chunk, dict):
        out = dict(chunk)
    else:
        out = {
            "chunk_id": getattr(chunk, "chunk_id", ""),
            "content": getattr(chunk, "content", ""),
            "path": getattr(chunk, "path", None),
        }
    out.setdefault("score", score)
    return out


async def vector_search(
    query_vector: list[float],
    *,
    store: VectorSearchStore,
    top_k: int = 10,
    filter: dict[str, Any] | None = None,  # noqa: A002 - 与 VectorStoreHandle 协议对齐
) -> list[dict[str, Any]]:
    """单次近邻检索。

    Args:
        query_vector: 查询向量。
        store: 注入的检索存储（MemoryStore 兼容协议）。
        top_k: 返回条数上限。
        filter: 元数据过滤条件（后端支持时生效）。

    Returns:
        按相似度降序的 dict 列表，每项含 chunk_id / content / score / path。

    Raises:
        VectorSearchError: 检索失败。
        OprimValidationError: query_vector 为空。
    """
    if not query_vector:
        raise OprimValidationError("vector_search: query_vector must not be empty")
    if top_k < 1:
        raise OprimValidationError("vector_search: top_k must be >= 1")
    if store is None:
        raise OprimValidationError("vector_search: store must be injected")

    search_fn = getattr(store, "search", None)
    if search_fn is None or not callable(search_fn):
        raise VectorSearchError("vector_search: store has no callable search()")

    try:
        sig = inspect.signature(search_fn)
        vector_param = sig.parameters.get("vector")
        keyword_style = vector_param is not None and (
            vector_param.kind is inspect.Parameter.KEYWORD_ONLY
            or vector_param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        if keyword_style:
            kwargs: dict[str, Any] = {"vector": query_vector, "top_k": top_k}
            if filter is not None and "filter" in sig.parameters:
                kwargs["filter"] = filter
            rows = await search_fn(**kwargs)
        else:
            rows = await search_fn(query_vector, top_k=top_k)
    except Exception as exc:
        raise VectorSearchError(
            f"vector_search failed: {type(exc).__name__}: {exc}", cause=exc
        ) from exc

    return [_norm_row(r) for r in (rows or [])]
