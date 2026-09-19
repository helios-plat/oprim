"""oprim.embedding.aii_remote — embed via the shared AII embed microservice.

HTTP client for ``aii.api.embed_app`` (the aii-embed systemd unit):
POST {base}/embed {"texts": [...]} -> {"embeddings": [[...]], "dim": N, "device": dev}.

Base URL resolution order: constructor arg -> STRATUM_EMBED_URL ->
AII_EMBED_URL -> http://127.0.0.1:8102. Accepts base URLs with or without a
trailing ``/embed`` path. Uses urllib (stdlib) with a short connect timeout

raises EmbeddingError when the service is unreachable so embedding failures
surface loudly instead of silently degrading (oskill treats embed as
best-effort and would otherwise write substrates with no vectors).

★2026-08-06 加固: embed 服务有 idle-unload 机制(空闲卸载模型), 重载窗口内
返回 502/503 — 对 502/503/504 退避重试(默认 5 次 ~44s, 覆盖重载窗口)

仍失败且配置了 fallback_base_url 时降级到兜底服务(如本机 8102), 最后才抛错。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from oprim.errors import EmbeddingError

_DEFAULT_PORT = 8102
_RETRYABLE = frozenset({502, 503, 504})
_RETRIES = 8
_BACKOFF = (5, 10, 20, 40, 60, 60, 60, 60)  # 总窗口 ~5min, 覆盖 WSL VM 复活周期


def _default_base_url() -> str:
    return (
        os.environ.get("STRATUM_EMBED_URL")
        or os.environ.get("AII_EMBED_URL")
        or f"http://127.0.0.1:{_DEFAULT_PORT}"
    )


def _normalize(url: str) -> str:
    u = url.rstrip("/")
    return u if u.endswith("/embed") else f"{u}/embed"


def _extract(payload: dict, url: str, dim: int) -> list[list[float]]:
    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, list) or not embeddings:
        raise EmbeddingError(f"aii-embed returned no embeddings from {url}")
    vecs: list[list[float]] = []
    for vec in embeddings:
        if len(vec) >= dim:
            vecs.append(vec[:dim])
        else:
            vecs.append(vec + [0.0] * (dim - len(vec)))
    return vecs


class AiiRemoteEmbedder:
    """BGE-M3 embeddings via the shared aii-embed HTTP service (no in-process model)."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 90.0,
        fallback_base_url: str | None = None,
    ) -> None:
        self._embed_url = _normalize(base_url or _default_base_url())
        self._timeout = timeout
        # ★2026-08-16 修复: 显式无代理 opener。否则 urllib 会读 http_proxy/https_proxy
        #   环境变量(如 Claude Code 的 127.0.0.1:7890), 而 Python 的 no_proxy 不支持
        #   CIDR 写法(100.64.0.0/10 不生效) → 尾网内网 embed 请求被代理吃掉 →
        #   sing-box 回 502 Bad Gateway → 飞轮全部 err=N。embed 是内网服务, 必须直连。
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        # ★降级链: 主服务(如笔记本 GPU)间歇窗口/宕机时退到兜底(如本机 8102)。
        self._fallback_url = _normalize(fallback_base_url) if fallback_base_url else None

    def embed(self, texts: list[str], dim: int = 1024) -> list[list[float]]:
        if not texts:
            return []
        body = json.dumps({"texts": list(texts)}).encode("utf-8")
        try:
            return self._post_with_retry(self._embed_url, body, dim, _RETRIES, _BACKOFF)
        except EmbeddingError:
            if self._fallback_url is not None:
                return self._post_with_retry(
                    self._fallback_url, body, dim, retries=2, backoff=(3, 5)
                )
            raise

    def _post_with_retry(
        self,
        url: str,
        body: bytes,
        dim: int,
        retries: int,
        backoff: tuple[int, ...],
    ) -> list[list[float]]:
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        last: Exception | None = None
        for attempt in range(retries):
            try:
                with self._opener.open(req, timeout=self._timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                return _extract(payload, url, dim)
            except urllib.error.HTTPError as exc:
                last = exc
                if exc.code not in _RETRYABLE or attempt == retries - 1:
                    break
                time.sleep(backoff[attempt])
            except (urllib.error.URLError, OSError) as exc:
                # ★2026-08-16: 连接失败/超时也重试(笔记本 WSL VM 会周期性闲置关闭,
                #   KeepWSL-boot 1 分钟内拉活; 重试窗口须覆盖 VM 复活周期 ~2-3min)。
                #   原来 URLError 直接 break → VM 死亡窗口内的请求全部失败。
                last = exc
                if attempt == retries - 1:
                    break
                time.sleep(backoff[attempt])
            except ValueError as exc:
                last = exc
                break
        raise EmbeddingError(f"aii-embed unreachable at {url}: {last}")

    @property
    def model_name(self) -> str:
        return "bge-m3 (aii-remote)"

    @property
    def native_dim(self) -> int:
        return 1024
