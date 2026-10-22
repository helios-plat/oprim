"""Public facade for email delivery."""
from __future__ import annotations
from typing import Any
from oprim._push_email import EmailResult, push_email as _push_email

def push_email(*args: Any, **kwargs: Any) -> Any:
    return _push_email(*args, **kwargs)

__all__ = ["EmailResult", "push_email"]
