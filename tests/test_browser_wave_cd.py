from __future__ import annotations

import asyncio

from obase.provider_registry import ProviderRegistry

import oprim.browser as browser_module
from oprim import browser_scroll, browser_wait


class Adapter:
    def scroll(self, handle, **kwargs):
        return {"ok": True, "operation": "scroll", **kwargs}

    def wait(self, handle, milliseconds):
        return {"ok": True, "operation": "wait", "milliseconds": milliseconds}


def test_browser_scroll_is_one_injected_operation():
    result = asyncio.run(browser_scroll({"session_id": "s"}, x=0, y=500, adapter=Adapter()))
    assert result == {"ok": True, "operation": "scroll", "x": 0, "y": 500}


def test_browser_wait_is_bounded_and_normalized():
    result = asyncio.run(browser_wait({"session_id": "s"}, milliseconds=25, adapter=Adapter()))
    assert result["operation"] == "wait"
    assert result["milliseconds"] == 25

    rejected = asyncio.run(
        browser_wait({"session_id": "s"}, milliseconds=30_001, adapter=Adapter())
    )
    assert rejected["ok"] is False


def test_default_browser_adapter_uses_the_single_provider_registry(monkeypatch):
    ProviderRegistry.clear()
    monkeypatch.setattr(browser_module, "_DEFAULT_ADAPTER", None)

    adapter = browser_module._default_adapter()

    assert ProviderRegistry.get().generic("browser", "default") is adapter
