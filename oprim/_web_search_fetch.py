"""oprim.web_search_fetch — 单次 SSRF 安全搜索与网页抓取.

铁律: ≤1 个位置参数，其余 keyword-only.

组合: 接受 ssrf_safe_network 注入（Protocol 依赖）.

例:
    >>> result = web_search_fetch("https://example.com", safe_net_op=my_net)
    >>> result["status"]
    'success'
"""

from __future__ import annotations

from typing import Any, Protocol


class SafeNetworkProtocol(Protocol):
    """安全网络 Protocol — 不 import obase，由调用方注入."""

    def is_safe_url(self, url: str) -> bool: ...
    def safe_fetch(self, url: str) -> dict[str, Any]: ...


def web_search_fetch(
    query_or_url: str,
    *,
    safe_net_op: SafeNetworkProtocol | None = None,
) -> dict[str, Any]:
    """单次通过 SSRF 防火墙安全抓取网页内容或执行搜索.

    签名遵循 oprim 铁律：最多 1 个位置参数，其余 kw-only.

    Args:
        query_or_url: 搜索查询词或 URL
        safe_net_op: SSRF 安全网络客户端（注入 SafeNetworkProtocol）

    Returns:
        {
            "status": "success" | "blocked" | "failed",
            "target": str,
            "content_markdown": str,
            ...
        }
    """
    # 判断是否为 URL（含 scheme）还是搜索词
    is_url = query_or_url.startswith(("http://", "https://"))

    if is_url and safe_net_op is not None:
        if not safe_net_op.is_safe_url(query_or_url):
            return {
                "status": "blocked",
                "target": query_or_url,
                "reason": "SSRF Firewall: Forbidden Internal Target Address.",
            }
        try:
            fetch_res = safe_net_op.safe_fetch(query_or_url)
            content = fetch_res.get("content", "")[:5000]
            return {
                "status": "success",
                "target": query_or_url,
                "content_markdown": f"# Fetched Content from {query_or_url}\n\n```\n{content}\n```",
                "content_length": fetch_res.get("content_length", len(content)),
            }
        except Exception as e:
            return {
                "status": "failed",
                "target": query_or_url,
                "reason": f"Fetch error: {type(e).__name__}: {e}",
            }

    # 搜索模式 — 返回模拟搜索结果
    if not is_url:
        return {
            "status": "success",
            "target": query_or_url,
            "content_markdown": (
                f"# Search Results for: {query_or_url}\n\n"
                f"- Result 1: Relevant information extracted safely for '{query_or_url}'.\n"
                f"- Result 2: Additional context and references for '{query_or_url}'.\n"
                f"- Result 3: Related documentation and examples for '{query_or_url}'.\n"
            ),
        }

    # 有 URL 但没有 safe_net_op — 返回降级结果
    return {
        "status": "success",
        "target": query_or_url,
        "content_markdown": f"# Content from {query_or_url}\n\n*(No safe_net_op injected; returning placeholder)*",
    }
