"""Sandbox deploy profile: local (laptop) vs hosted (multi-user).

Reads ``VEYA_SANDBOX_PROFILE``. Default ``local`` so ``veya start`` stays
process/docker. Hosted forbids process isolation at the oprim gate.
"""

from __future__ import annotations

import os

_HOSTED_ALIASES = frozenset({"hosted", "host", "cloud", "prod"})


def sandbox_profile() -> str:
    raw = os.environ.get("VEYA_SANDBOX_PROFILE", "local").strip().lower()
    if raw in _HOSTED_ALIASES:
        return "hosted"
    return "local"


def hosted_forbids_process() -> bool:
    return sandbox_profile() == "hosted"
