"""Pure controller-generation primitives."""
from __future__ import annotations

import hashlib


def next_controller_generation(previous: int | None) -> int:
    """Return the next generation number, starting at 1."""
    if previous is not None and previous < 0:
        raise ValueError("previous generation must be >= 0")
    return 1 if previous is None else previous + 1


def controller_generation_id(root_run_id: str, generation: int) -> str:
    """Build a stable provider-neutral controller-generation identifier."""
    if not root_run_id:
        raise ValueError("root_run_id must not be empty")
    if generation < 1:
        raise ValueError("generation must be >= 1")
    payload = f"{root_run_id}\x00{generation}".encode()
    return "cg_" + hashlib.sha256(payload).hexdigest()[:24]
