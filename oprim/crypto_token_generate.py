"""Public facade for cryptographically secure token generation."""
from __future__ import annotations
from typing import Any
from oprim._crypto_token_generate import crypto_token_generate as _crypto_token_generate

def crypto_token_generate(*args: Any, **kwargs: Any) -> Any:
    return _crypto_token_generate(*args, **kwargs)

__all__ = ["crypto_token_generate"]
