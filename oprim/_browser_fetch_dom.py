"""oprim.browser_fetch_dom — 无头浏览器获取单页 DOM 或渲染截图.

经注入的 page 句柄（obase.browser_runner.BrowserPageHandle 或兼容协议）
访问目标 URL 并抓取 DOM，可选同时截图。

Example:
    >>> page = await runner.new_page()
    >>> r = await browser_fetch_dom("https://example.com", browser=page)
    >>> "<title>" in r["html"]
    True
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from oprim._exceptions import OprimError, OprimValidationError


class BrowserFetchError(OprimError):
    """页面获取失败。"""


@runtime_checkable
class BrowserPageProtocol(Protocol):
    """最小页面句柄协议（与 obase.browser_runner.BrowserPageHandle 对齐）。"""

    async def goto(self, url: str, *, timeout_ms: int = 30_000) -> None: ...
    async def content(self) -> str: ...
    async def screenshot(self, *, path: str | None = None) -> bytes: ...


async def browser_fetch_dom(
    url: str,
    *,
    browser: BrowserPageProtocol,
    timeout_ms: int = 30_000,
    screenshot: bool = False,
    screenshot_path: str | None = None,
) -> dict[str, Any]:
    """访问 URL 并抓取 DOM（可选截图）。

    Args:
        url: 目标 URL（http/https）。
        browser: 页面句柄（BrowserPageHandle 兼容协议）。
        timeout_ms: 页面加载超时毫秒数。
        screenshot: 是否同时截图。
        screenshot_path: 截图落盘路径；None 且 screenshot=True 时返回 bytes。

    Returns:
        {"status": "ok", "url": str, "html": str, "title": str | None,
         "screenshot_path": str | None, "screenshot_bytes": bytes | None}

    Raises:
        BrowserFetchError: 加载失败。
        OprimValidationError: url 或 browser 缺失。
    """
    if not url:
        raise OprimValidationError("browser_fetch_dom: url must not be empty")
    if browser is None:
        raise OprimValidationError("browser_fetch_dom: browser page handle required")

    try:
        await browser.goto(url, timeout_ms=timeout_ms)
        html = await browser.content()
    except Exception as exc:
        raise BrowserFetchError(
            f"browser_fetch_dom failed for {url}: {type(exc).__name__}: {exc}",
            cause=exc,
        ) from exc

    title: str | None = None
    low = html.lower()
    start = low.find("<title")
    if start >= 0:
        end = low.find("</title>", start)
        if end >= 0:
            import re

            raw = html[low.find(">", start) + 1 : end]
            title = re.sub(r"\s+", " ", raw).strip() or None

    shot_path: str | None = None
    shot_bytes: bytes | None = None
    if screenshot:
        try:
            if screenshot_path:
                await browser.screenshot(path=screenshot_path)
                shot_path = screenshot_path
            else:
                shot_bytes = await browser.screenshot()
        except Exception:  # noqa: BLE001 - 截图失败不致命
            shot_path = None
            shot_bytes = None

    return {
        "status": "ok",
        "url": url,
        "html": html,
        "title": title,
        "screenshot_path": shot_path,
        "screenshot_bytes": shot_bytes,
    }
