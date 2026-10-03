"""Atomic capability-set intersection primitive.

Pure execution mechanic: computes the effective capability set from independently
authorized capability sets. Policy meaning and wildcard interpretation stay outside
this primitive.
"""
from __future__ import annotations

from collections.abc import Iterable


def intersect_capabilities(*capability_sets: Iterable[str]) -> frozenset[str]:
    """Return the exact intersection of all supplied capability sets.

    An empty input sequence is fail-closed and returns an empty set. Inputs are
    normalized to strings and surrounding whitespace is rejected rather than
    silently widened.
    """
    if not capability_sets:
        return frozenset()

    normalized: list[set[str]] = []
    for values in capability_sets:
        current: set[str] = set()
        for value in values:
            if not isinstance(value, str) or not value:
                raise ValueError("capability names must be non-empty strings")
            if value != value.strip():
                raise ValueError("capability names must not contain surrounding whitespace")
            current.add(value)
        normalized.append(current)

    result = normalized[0]
    for current in normalized[1:]:
        result.intersection_update(current)
    return frozenset(result)
