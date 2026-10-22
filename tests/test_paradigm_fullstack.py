"""3O 范式全栈 — oprim 新增原子测试 (元素 10/11/12)."""

from __future__ import annotations

import pytest
from obase import SQLiteKVStore
from obase.daemon_bus import InProcessDaemonBus

from oprim import browser_click, emit_global_event, fetch_state
from oprim.browser_action import BrowserProvider


async def test_fetch_state_roundtrip(tmp_path):
    store = SQLiteKVStore(str(tmp_path / "kv.db"))
    await store.put("k/1", {"n": 1})
    assert await fetch_state("k/1", store=store) == {"n": 1}


async def test_fetch_state_missing_raises(tmp_path):
    store = SQLiteKVStore(str(tmp_path / "kv.db"))
    with pytest.raises(KeyError, match="k/missing"):
        await fetch_state("k/missing", store=store)


async def test_emit_global_event():
    bus = InProcessDaemonBus()
    sub = await bus.subscribe("events/user-action")
    await emit_global_event("events/user-action", payload={"kind": "click"}, bus=bus)
    assert await sub.receive() == {"kind": "click"}


class _FakeBrowser:
    """结构化实现 BrowserProvider 协议的假浏览器。"""

    def __init__(self, ok: bool = True):
        self.ok = ok
        self.clicked: list[str] = []

    async def click_element(self, selector: str) -> bool:
        self.clicked.append(selector)
        return self.ok


async def test_browser_click():
    browser = _FakeBrowser()
    assert await browser_click("#submit", browser=browser) is True
    assert browser.clicked == ["#submit"]


async def test_browser_click_failure():
    browser = _FakeBrowser(ok=False)
    assert await browser_click("#ghost", browser=browser) is False


def test_browser_provider_is_protocol():
    with pytest.raises(TypeError):
        BrowserProvider()  # type: ignore[abstract]
