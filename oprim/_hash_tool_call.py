"""oprim.hash_tool_call — deterministic tool-call digest."""

from __future__ import annotations

from typing import Any

from obase.canonical_json import canonical_json
from obase.sha256_hash import sha256_hash


def hash_tool_call(tool_name: str, *, arguments: dict[str, Any] | None = None) -> str:
    payload = canonical_json({"tool": tool_name, "arguments": arguments or {}})
    return sha256_hash(payload).hex()
