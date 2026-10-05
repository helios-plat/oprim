"""Word boundary snap oprim — move a time boundary off the middle of a word.

Canonical primitive for transcript-aware cut planning. A cut that lands inside a
word truncates it audibly; a cut that lands on a word edge does not. This element
performs the move, so a planner can *produce* clean boundaries rather than only
detect unclean ones afterwards.

The nearest existing item, `edge_tts_word_boundary`, *produces* word timestamps.
It does not snap to them. Detection of truncation is the caller's business.

Read-only (R0). No element-to-element call: this module imports only stdlib and
oprim's own exception helper.
"""

from __future__ import annotations

from typing import Any

from oprim._exceptions import OprimError

#: A boundary within this many seconds of a word edge counts as already clean.
DEFAULT_TOLERANCE_S = 0.02


class WordBoundarySnapError(OprimError):
    """Raised when no clean boundary exists to snap to."""


def _word_spans(words: list[dict[str, Any]]) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    for word in words:
        start = word.get("start")
        end = word.get("end")
        if start is None or end is None:
            continue
        start_f, end_f = float(start), float(end)
        if end_f > start_f:
            spans.append((start_f, end_f))
    spans.sort()
    return spans


def _snap_one(
    boundary: float,
    spans: list[tuple[float, float]],
    *,
    direction: str,
    tolerance_s: float,
    lo: float | None,
    hi: float | None,
) -> float:
    """Nearest edge of the containing word, or the boundary itself if it is clean."""
    for start, end in spans:
        if start + tolerance_s < boundary < end - tolerance_s:
            return start if direction == "out" else end
    return boundary


def word_boundary_snap(
    *,
    boundaries_s: list[float],
    words: list[dict[str, Any]],
    direction: str = "nearest",
    tolerance_s: float = DEFAULT_TOLERANCE_S,
    media_duration_s: float | None = None,
) -> list[float]:
    """Snap time boundaries onto word edges so a cut cannot truncate a word.

    Args:
        boundaries_s: Boundary timestamps in seconds. Order is not preserved on
            output only insofar as snapping can reorder equal or near-equal
            values; each returned value corresponds to the input at the same index.
        words: Word timings, each a mapping with ``start`` and ``end`` keys in
            seconds (the shape `edge_tts_word_boundary` returns). Words missing a
            start or end are ignored.
        direction: ``"out"`` snaps to a word's start (cut before the word),
            ``"in"`` snaps to its end (cut after it), ``"nearest"`` picks the
            closer edge and, on a tie, the word end — snapping back to a word
            start would truncate it.
        tolerance_s: A boundary already within this distance of a word edge is
            left alone, so snapping is idempotent.
        media_duration_s: When given, every returned boundary is clamped into
            ``[0, media_duration_s]``.

    Returns:
        list[float]: One snapped boundary per input boundary, same length and
        order.

    Raises:
        ValueError: ``boundaries_s`` is empty, or ``direction`` is not one of
            ``"out"``/``"in"``/``"nearest"``.
        RuntimeError: ``tolerance_s`` is negative.
        WordBoundarySnapError: ``words`` contains no usable word timings, so
            there is nothing to snap to. An unevaluable snap must not return the
            input unchanged and let the caller believe it was verified.
    """
    if not boundaries_s:
        raise ValueError("boundaries_s must contain at least one boundary")
    if direction not in {"out", "in", "nearest"}:
        raise ValueError(f"direction must be 'out', 'in' or 'nearest'; got {direction!r}")
    if tolerance_s < 0:
        raise RuntimeError(f"tolerance_s must be >= 0; got {tolerance_s}")

    spans = _word_spans(words)
    if not spans:
        raise WordBoundarySnapError(
            "WORD_BOUNDARY_SNAP_NO_WORDS:no usable word timings were supplied, "
            "so no boundary can be verified as clean"
        )

    snapped: list[float] = []
    for boundary in boundaries_s:
        value = float(boundary)
        if direction == "nearest":
            best: float | None = None
            for start, end in spans:
                if start + tolerance_s < value < end - tolerance_s:
                    # Tie-break is deliberate: a boundary equidistant from both
                    # edges moves *forward* to the word end. Snapping back to the
                    # start would truncate the word, and truncation is the defect
                    # this element exists to prevent.
                    candidate = start if (value - start) < (end - value) else end
                    if best is None or abs(candidate - value) < abs(best - value):
                        best = candidate
            value = value if best is None else best
        else:
            value = _snap_one(
                value, spans, direction=direction,
                tolerance_s=tolerance_s, lo=None, hi=None,
            )
        if media_duration_s is not None:
            value = min(max(value, 0.0), float(media_duration_s))
        snapped.append(value)
    return snapped


__all__ = [
    "DEFAULT_TOLERANCE_S",
    "WordBoundarySnapError",
    "word_boundary_snap",
]