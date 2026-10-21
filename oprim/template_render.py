"""Public facade for template rendering."""
from __future__ import annotations
from typing import Any
from oprim._template_render import template_render as _template_render

def template_render(*args: Any, **kwargs: Any) -> Any:
    return _template_render(*args, **kwargs)

__all__ = ["template_render"]
