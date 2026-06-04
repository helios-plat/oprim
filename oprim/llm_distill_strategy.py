"""oprim.llm_distill_strategy — single LLM call to distill solution_strategy from Episode.

3O layer: oprim (single LLM call, obase.ProviderRegistry, no state).
Produces default unverified strategy (A19: LLM proposes, never certifies).
13-Learning-SPEC: solution_strategy has 3-part format: title/description/content.
"""

from __future__ import annotations

import json
import uuid

_DISTILL_PROMPT = """Distill a reusable solution strategy from this problem-solving episode.

Return JSON with:
- title: short strategy name (< 60 chars)
- description: what problem this solves and when to apply (1-2 sentences)
- content: the strategy steps or approach (structured detail)

Episode:
Event: {event}
Outcome: {outcome}
Context: {context}

Respond with valid JSON only."""


def llm_distill_strategy(
    *,
    episode: dict,
    project_id: str = "default",
    provider: str = "default",
) -> dict:
    """Distill a solution_strategy KU from an Episode via single LLM call.

    episode: {event, outcome, context, ...}
    Returns KU dict with knowledge_type="solution_strategy", epistemic_status.verified=False (A19).
    """
    event = episode.get("event", "")
    outcome = episode.get("outcome", "")
    context = str(episode.get("context", ""))

    try:
        from obase import ProviderRegistry

        reg = ProviderRegistry.get_instance()
        llm = reg.get("llm", provider)
        prompt = _DISTILL_PROMPT.format(
            event=event[:500], outcome=outcome[:200], context=context[:500]
        )
        response = llm(prompt)
        raw = json.loads(response.strip())
        title = raw.get("title", f"Strategy from: {event[:40]}")
        description = raw.get("description", "")
        content = raw.get("content", "")
    except Exception:
        # Deterministic stub
        title = f"Strategy: {outcome[:40]}" if outcome else "Strategy from episode"
        description = f"Approach used when: {event[:80]}"
        content = f"Outcome: {outcome}"

    return {
        "ku_id": str(uuid.uuid4()),
        "knowledge_type": "solution_strategy",
        "natural_text": f"{title}: {description}",
        "symbolic_form": {"title": title, "description": description, "content": content},
        "vector": None,
        "vector_frozen": False,
        "epistemic_status": {
            "grade": "unverified",
            "source": None,
            "defeaters": [],
            "verified": False,  # A19: LLM proposal, default unverified
        },
        "provenance": {"source": "llm_distill", "episode_id": episode.get("episode_id")},
        "project_id": project_id,
    }
