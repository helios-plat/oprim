"""oprim.browser_click_element — 无头浏览器单次元素点击操作.

经注入的页面句柄定位 CSS 选择器并点击；元素不存在或超时抛
BrowserClickError。

Example:
    >>> r = await browser_click_element("#submit", browser=page)
    >>> r["clicked"]
    True
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from oprim._exceptions import OprimError, OprimValidationError


class BrowserClickError(OprimError):
    """元素点击失败（未找到 / 超时 / 页面错误）。"""


@runtime_checkable
class BrowserClickPageProtocol(Protocol):
    """最小点击协议（BrowserPageHandle 对齐）。"""

    async def click(self, selector: str, *, timeout_ms: int = 10_000) -> None: ...


async def browser_click_element(
    selector: str,
    *,
    browser: BrowserClickPageProtocol,
    timeout_ms: int = 10_000,
) -> dict[str, Any]:
    """点击页面元素。

    Args:
        selector: CSS 选择器。
        browser: 页面句柄（须支持 click）。
        timeout_ms: 元素出现/可点击超时毫秒数。

    Returns:
        {"status": "ok", "selector": str, "clicked": True}

    Raises:
        BrowserClickError: 点击失败（未找到 / 超时）。
        OprimValidationError: selector 或 browser 缺失。
    """
    if not selector.strip():
        raise OprimValidationError("browser_click_element: selector must not be empty")
    if browser is None:
        raise OprimValidationError("browser_click_element: browser page handle required")
    click_fn = getattr(browser, "click", None)
    if click_fn is None or not callable(click_fn):
        raise BrowserClickError("browser_click_element: browser has no click()")

    try:
        await click_fn(selector, timeout_ms=timeout_ms)
    except Exception as exc:
        raise BrowserClickError(
            f"browser_click_element failed for {selector!r}: {type(exc).__name__}: {exc}",
            cause=exc,
        ) from exc

    return {"status": "ok", "selector": selector, "clicked": True}
