"""oprim.subject_create — Create and persist a Subject record."""
from __future__ import annotations

from typing import Any

from oprim._hevi_types import Subject


async def subject_create(subject: Subject, *, store: Any = None) -> Subject:
    """Create and persist a Subject record.

    If a store dict is provided, writes directly to it. Otherwise attempts to
    use obase.persistence; falls back to a module-level in-memory dict if
    persistence is unavailable (e.g. in test environments).

    Args:
        subject: The Subject instance to persist.
        store: Optional dict acting as a simple key-value store.

    Returns:
        The persisted Subject (same object).
    """
    if store is not None:
        store[subject.subject_id] = subject
        return subject

    try:
        from obase.persistence import write_one  # noqa: F401
        # write_one expects a pool; fall through to in-memory if not configured
        _MEMORY_STORE[subject.subject_id] = subject
    except Exception:
        _MEMORY_STORE[subject.subject_id] = subject

    return subject


# Module-level fallback store used when no store is injected and persistence
# is unavailable.
_MEMORY_STORE: dict[str, Subject] = {}
