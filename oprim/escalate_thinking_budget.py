"""Atomic prompt inspection for selecting an explicit thinking budget."""
from __future__ import annotations

_THINKING_KEYWORDS = [
    (["ultrathink", "think very hard", "think extremely hard"], 31_000),
    (["think hard", "think carefully", "think deeply", "think step by step"], 10_000),
    (["think", "reason", "analyze", "consider", "reflect"], 5_000),
]


def escalate_thinking_budget(prompt: str) -> int | None:
    """Return the explicit thinking budget requested by a prompt."""
    lower = prompt.lower()
    for keywords, budget in _THINKING_KEYWORDS:
        if any(keyword in lower for keyword in keywords):
            return budget
    return None
