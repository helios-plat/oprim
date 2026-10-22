"""Public facade for one-time password helpers."""
from __future__ import annotations
from typing import Any
from oprim._otp_generate import OTPResult, otp_generate as _otp_generate, otp_verify as _otp_verify

def otp_generate(*args: Any, **kwargs: Any) -> Any:
    return _otp_generate(*args, **kwargs)

def otp_verify(*args: Any, **kwargs: Any) -> Any:
    return _otp_verify(*args, **kwargs)

__all__ = ["OTPResult", "otp_generate", "otp_verify"]
