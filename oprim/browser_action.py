"""浏览器动作原子操作 (3O 范式全栈 — oprim 元素 12).

浏览器适配器由上层注入 (BrowserProvider 协议), 原子自身零实现。
"""

from __future__ import annotations

from typing import Protocol


class BrowserProvider(Protocol):
    """浏览器适配器协议 (由上层注入实现)"""

    async def click_element(self, selector: str) -> bool: ...


async def browser_click(
    selector: str,
    *,
    browser: BrowserProvider,
) -> bool:
    """单次原子动作: 点击页面元素。

    Args:
        selector: CSS 选择器。
        browser: 浏览器适配器 (依赖注入)。

    Returns:
        是否点击成功。
    """
    return await browser.click_element(selector)
