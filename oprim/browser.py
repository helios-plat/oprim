"""Atomic browser session and browser action operations.

Each function performs exactly one adapter operation.  The browser driver is
owned by ``obase.browser`` and is injected for tests or alternate drivers;
policy and approval remain outside this layer.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any

from obase.browser import (
    BrowserAdapter,
    BrowserControlState,
    BrowserProfile,
    BrowserSessionHandle,
    PlaywrightBrowserAdapter,
)

_DEFAULT_ADAPTER: BrowserAdapter | None = None


def _default_adapter() -> BrowserAdapter:
    global _DEFAULT_ADAPTER
    if _DEFAULT_ADAPTER is None:
        _DEFAULT_ADAPTER = PlaywrightBrowserAdapter()
    return _DEFAULT_ADAPTER


def _adapter(value: BrowserAdapter | None) -> BrowserAdapter:
    return value or _default_adapter()


async def _invoke(
    operation: str,
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    try:
        result = fn(*args, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, Mapping):
            return dict(result)
        return {"ok": True, "operation": operation, "result": result}
    except Exception as exc:  # noqa: BLE001 - atomic boundary returns structured failure
        return {
            "ok": False,
            "operation": operation,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


async def browser_create(
    profile: BrowserProfile | Mapping[str, Any], *, adapter: BrowserAdapter | None = None
) -> dict[str, Any]:
    """Create a browser session without starting a browser process."""
    return await _invoke("create", _adapter(adapter).create, profile)


async def browser_start(
    handle: BrowserSessionHandle | Mapping[str, Any], *, adapter: BrowserAdapter | None = None
) -> dict[str, Any]:
    """Start one created/stopped browser session."""
    return await _invoke("start", _adapter(adapter).start, handle)


async def browser_status(
    handle: BrowserSessionHandle | Mapping[str, Any], *, adapter: BrowserAdapter | None = None
) -> dict[str, Any]:
    """Read browser session and control status."""
    return await _invoke("status", _adapter(adapter).status, handle)


async def browser_attach(
    handle: BrowserSessionHandle | Mapping[str, Any], *, adapter: BrowserAdapter | None = None
) -> dict[str, Any]:
    """Attach a user-facing view to a running browser session."""
    return await _invoke("attach", _adapter(adapter).attach, handle)


async def browser_stop(
    handle: BrowserSessionHandle | Mapping[str, Any], *, adapter: BrowserAdapter | None = None
) -> dict[str, Any]:
    """Stop a browser session while retaining its computer/workspace."""
    return await _invoke("stop", _adapter(adapter).stop, handle)


async def browser_reset(
    handle: BrowserSessionHandle | Mapping[str, Any], *, adapter: BrowserAdapter | None = None
) -> dict[str, Any]:
    """Reset browser runtime state without deleting the bound computer."""
    return await _invoke("reset", _adapter(adapter).reset, handle)


async def browser_set_control_state(
    handle: BrowserSessionHandle | Mapping[str, Any],
    *,
    state: BrowserControlState,
    adapter: BrowserAdapter | None = None,
) -> dict[str, Any]:
    """Change the explicit AGENT_CONTROL/HUMAN_CONTROL session state."""
    return await _invoke("set_control_state", _adapter(adapter).set_control_state, handle, state)


async def browser_navigate(
    handle: BrowserSessionHandle | Mapping[str, Any],
    *,
    url: str,
    adapter: BrowserAdapter | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Navigate the current browser page."""
    return await _invoke("navigate", _adapter(adapter).navigate, handle, url, **kwargs)


async def browser_snapshot(
    handle: BrowserSessionHandle | Mapping[str, Any],
    *,
    adapter: BrowserAdapter | None = None,
    selector: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Read a text/HTML snapshot from the current page."""
    return await _invoke(
        "snapshot", _adapter(adapter).snapshot, handle, selector=selector, **kwargs
    )


async def browser_click(
    handle: BrowserSessionHandle | Mapping[str, Any],
    *,
    selector: str,
    adapter: BrowserAdapter | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Click one page element."""
    return await _invoke("click", _adapter(adapter).click, handle, selector, **kwargs)


async def browser_type(
    handle: BrowserSessionHandle | Mapping[str, Any],
    *,
    selector: str,
    text: str,
    adapter: BrowserAdapter | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Type/fill one page element."""
    return await _invoke("type", _adapter(adapter).type, handle, selector, text, **kwargs)


async def browser_download(
    handle: BrowserSessionHandle | Mapping[str, Any],
    *,
    selector: str,
    adapter: BrowserAdapter | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Download one file initiated by one page element."""
    return await _invoke("download", _adapter(adapter).download, handle, selector, **kwargs)


async def browser_upload(
    handle: BrowserSessionHandle | Mapping[str, Any],
    *,
    selector: str,
    file_paths: Any,
    adapter: BrowserAdapter | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Upload one or more caller-provided files through one input element."""
    return await _invoke("upload", _adapter(adapter).upload, handle, selector, file_paths, **kwargs)


async def browser_screenshot(
    handle: BrowserSessionHandle | Mapping[str, Any],
    *,
    adapter: BrowserAdapter | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Capture the current browser page or selected element."""
    return await _invoke("screenshot", _adapter(adapter).screenshot, handle, **kwargs)


__all__ = [
    "browser_attach",
    "browser_click",
    "browser_create",
    "browser_download",
    "browser_navigate",
    "browser_reset",
    "browser_screenshot",
    "browser_set_control_state",
    "browser_snapshot",
    "browser_start",
    "browser_status",
    "browser_stop",
    "browser_type",
    "browser_upload",
]
