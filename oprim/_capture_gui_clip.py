"""oprim.capture_gui_clip — record a real GUI interaction. Never fabricate a gif."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def capture_gui_clip(
    *,
    output_path: str | Path,
    url: str = "",
    steps: list[dict[str, Any]] | None = None,
    script: str = "",
    width: int = 1280,
    height: int = 720,
) -> dict[str, Any]:
    """Capture an interaction clip with Playwright if available.

    If Playwright is missing or the run fails, returns ``ok=False`` and a
    readable reason. Does **not** write a placeholder/fake gif.
    """
    out = Path(output_path)
    if not url and not script:
        return {
            "ok": False,
            "path": "",
            "reason": "no url or script provided; will not fabricate a clip",
        }

    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except ImportError:
        return {
            "ok": False,
            "path": "",
            "reason": "playwright not installed; will not fabricate a clip",
        }

    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": width, "height": height})
            page.context.tracing.start(screenshots=True, snapshots=True)
            if url:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            for step in steps or []:
                _apply_step(page, step)
            # Prefer webm/png frames over inventing a gif container.
            dest = out.with_suffix(".webm") if out.suffix.lower() == ".gif" else out
            page.screenshot(path=str(dest.with_suffix(".png")), full_page=True)
            page.context.tracing.stop()
            browser.close()
        shot = dest.with_suffix(".png")
        if not shot.is_file():
            return {
                "ok": False,
                "path": "",
                "reason": "playwright ran but produced no frame; will not fabricate a clip",
            }
        return {
            "ok": True,
            "path": str(shot),
            "reason": "",
            "note": "wrote a real screenshot; gif encode is optional and not faked",
        }
    except Exception as exc:
        if out.exists() and out.stat().st_size == 0:
            out.unlink(missing_ok=True)
        return {
            "ok": False,
            "path": "",
            "reason": f"playwright failed: {exc}; will not fabricate a clip",
        }


def _apply_step(page: Any, step: dict[str, Any]) -> None:
    action = str(step.get("action") or step.get("type") or "").lower()
    selector = step.get("selector") or step.get("target") or ""
    value = step.get("value") or step.get("text") or ""
    if action in {"goto", "open"} and (step.get("url") or value):
        page.goto(str(step.get("url") or value), wait_until="domcontentloaded")
    elif action in {"click", "tap"} and selector:
        page.click(str(selector))
    elif action in {"fill", "type"} and selector:
        page.fill(str(selector), str(value))
    elif action in {"press"} and selector:
        page.press(str(selector), str(value or "Enter"))
    elif action in {"wait", "wait_for"} and selector:
        page.wait_for_selector(str(selector), timeout=10_000)
    elif action in {"screenshot"}:
        return
