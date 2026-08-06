"""oprim.browser_element_interact — Playwright-driven real browser automation.

Launches a headless Chromium instance, navigates to a URL, and performs a
sequence of actions (click, type, scroll, screenshot) against real DOM
elements — including computing bounding-box coordinates.

3O element: ``oprim.browser_element_interact`` (``_browser_element_interact`` legacy name).
"""

from __future__ import annotations

import base64
import time
from typing import Any


def browser_element_interact(
    url: str,
    actions: list[dict[str, Any]] | None = None,
    context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Navigate to a URL and execute a sequence of browser actions.

    Args:
        url: Target page URL.
        actions: List of action dicts:
            ``{"type": "click", "selector": "...", "wait": 1000}``,
            ``{"type": "type", "selector": "...", "value": "text"}``,
            ``{"type": "scroll", "y": 500}``,
            ``{"type": "screenshot"}``,
            ``{"type": "wait", "ms": 2000}``.
        context: Optional config (``viewport``, ``headless``, ``timeout``).

    Returns:
        ``{results, screenshot_b64, elapsed_ms, status}``
    """
    ctx = context or {}
    headless = ctx.get("headless", True)
    viewport = ctx.get("viewport", {"width": 1280, "height": 720})
    timeout = int(ctx.get("timeout", 30000))

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return {"status": "failed", "error": "playwright not installed", "results": []}

    results: list[dict[str, Any]] = []
    screenshot_b64: str | None = None
    t0 = time.time()

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=headless)
            page = browser.new_page(viewport=viewport)
            page.goto(url, timeout=timeout)

            for act in (actions or []):
                act_type = act.get("type", "click")
                selector = act.get("selector", "")
                wait_ms = int(act.get("wait", 500))

                if act_type == "click":
                    el = page.locator(selector).first
                    box = el.bounding_box()
                    el.click(timeout=timeout)
                    results.append({
                        "action": "click",
                        "selector": selector,
                        "x": round(box["x"], 1) if box else None,
                        "y": round(box["y"], 1) if box else None,
                        "status": "ok",
                    })
                elif act_type == "type":
                    el = page.locator(selector).first
                    el.fill(str(act.get("value", "")), timeout=timeout)
                    results.append({"action": "type", "selector": selector, "status": "ok"})
                elif act_type == "scroll":
                    page.evaluate(f"window.scrollBy(0, {act.get('y', 500)})")
                    results.append({"action": "scroll", "y": act.get("y", 500), "status": "ok"})
                elif act_type == "screenshot":
                    buf = page.screenshot(full_page=act.get("full_page", False))
                    screenshot_b64 = base64.b64encode(buf).decode("ascii")
                    results.append({"action": "screenshot", "status": "ok"})
                elif act_type == "wait":
                    page.wait_for_timeout(wait_ms)
                    results.append({"action": "wait", "ms": wait_ms, "status": "ok"})
                else:
                    results.append({"action": act_type, "status": "skipped", "reason": "unknown type"})

            browser.close()
    except Exception as exc:
        results.append({"action": "error", "error": str(exc)[:300]})

    elapsed = int((time.time() - t0) * 1000)
    return {
        "results": results,
        "screenshot_b64": screenshot_b64,
        "elapsed_ms": elapsed,
        "url": url,
        "status": "completed" if all(r.get("status") == "ok" for r in results) else "partial",
    }
