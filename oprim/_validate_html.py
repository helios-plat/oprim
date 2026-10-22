"""oprim.validate_html — sandboxed HTML safety check (no LLM, no IO).

The public capability. Implementation lives in `_html_safety` (infra) so that
`render_html_to_mp4` can reach the same scanner without importing a sibling
OPrim — element → element is a forbidden dependency direction.
"""

from __future__ import annotations

from oprim._animation_types import HtmlValidationResult
from oprim._html_safety import scan_html


def validate_html(*, html: str, allow_external_src: bool = False) -> HtmlValidationResult:
    """Scan html for unsafe patterns. Pure function — no LLM, no IO.

    Detected patterns:
      external_script_src    <script src="http://...">
      inline_event_handler   onerror=, onload=, onclick=, … attribute
      eval_usage             eval( anywhere in content
      javascript_uri         href/src/action="javascript:..."
      external_iframe        <iframe src="http://...">
      fetch_usage            fetch( anywhere in content
      xmlhttprequest_usage   XMLHttpRequest anywhere in content
      external_src           any src="http://..." (only when allow_external_src=False)

    Returns HtmlValidationResult(is_safe, violations, sanitized).
    sanitized is None when html is safe
    otherwise the html with dangerous
    patterns neutralised (best-effort — not a full HTML sanitiser).
    """
    return scan_html(html=html, allow_external_src=allow_external_src)
