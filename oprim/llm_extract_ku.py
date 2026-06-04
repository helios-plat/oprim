"""oprim.llm_extract_ku — single LLM call to extract a KU candidate from text.

3O layer: oprim (single LLM call, obase.ProviderRegistry, no state).
Produces default unverified KU candidate (A19: LLM proposes, never certifies).
"""

from __future__ import annotations

import json
import uuid

_EXTRACT_PROMPT = """Extract a knowledge unit from the following text chunk.

Return JSON with these fields:
- knowledge_type: one of [proposition, rule, case, opinion, procedure, query, solution_strategy, relation, formula, theorem]
- natural_text: the key claim or knowledge in one clear sentence
- symbolic_form: structured representation (dict with type-specific fields, or null)
- tags: list of relevant keywords

Text:
{text}

Respond with valid JSON only."""


def llm_extract_ku(
    *,
    text: str,
    project_id: str = "default",
    knowledge_type_hint: str | None = None,
    provider: str = "default",
) -> dict:
    """Extract a KU candidate from text via single LLM call.

    Returns KU dict with epistemic_status.verified=False (A19: default unverified).
    Falls back to stub extraction if LLM unavailable.
    """
    try:
        from obase import ProviderRegistry

        reg = ProviderRegistry.get_instance()
        llm = reg.get("llm", provider)
        prompt = _EXTRACT_PROMPT.format(text=text[:3000])
        response = llm(prompt)
        # Parse JSON from response
        raw = json.loads(response.strip())
        knowledge_type = raw.get("knowledge_type", "proposition")
        natural_text = raw.get("natural_text", text[:200])
        symbolic_form = raw.get("symbolic_form")
        tags = raw.get("tags", [])
    except Exception:
        # Deterministic stub: extract first sentence as proposition
        natural_text = text.split(".")[0].strip()[:200] or text[:200]
        knowledge_type = knowledge_type_hint or "proposition"
        symbolic_form = None
        tags = []

    return {
        "ku_id": str(uuid.uuid4()),
        "knowledge_type": knowledge_type,
        "natural_text": natural_text,
        "symbolic_form": symbolic_form,
        "vector": None,  # filled later by vector_encode
        "vector_frozen": False,
        "epistemic_status": {
            "grade": "unverified",
            "source": None,
            "defeaters": [],
            "verified": False,  # A19: LLM proposal, default unverified
        },
        "provenance": {"source": "llm_extract", "chunk_id": None},
        "project_id": project_id,
        "tags": tags,
    }
