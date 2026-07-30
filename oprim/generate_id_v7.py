"""oprim.generate_id_v7 — prefixed, time-sortable ID generation (UUID v7).

Thin wrapper over obase.uuid7 (RFC 9562 UUIDv7) that prepends a Medusa-style
entity prefix (e.g. "cust", "prod") — same precedent as oprim._new_session_id.
"""

from __future__ import annotations

from obase import uuid7


def generate_id_v7(prefix: str) -> str:
    """Generate a prefixed, lexicographically time-sortable ID.

    Args:
        prefix: Entity prefix (e.g. "cust", "prod", "order"). Must be non-empty.

    Returns:
        `f"{prefix}_{uuid7()}"`.

    Raises:
        ValueError: prefix is empty.

    Example:
        >>> generate_id_v7("cust")  # doctest: +SKIP
        'cust_018f4d2e-...'
    """
    if not prefix:
        raise ValueError("prefix must not be empty")
    return f"{prefix}_{uuid7()}"
