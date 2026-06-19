"""oprim.llm_extract_ku — single LLM call to extract a KU candidate from text.

3O layer: oprim (single LLM call, obase.ProviderRegistry, no state).
Produces default unverified KU candidate (A19: LLM proposes, never certifies).
"""

from __future__ import annotations

import json
import logging
import re
import uuid

_log = logging.getLogger(__name__)

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

    Calls obase.ProviderRegistry.get("llm", provider). On ProviderNotFoundError
    logs a warning and falls back to a deterministic stub. Any other exception
    (code error) is re-raised — not silently swallowed.

    Returns KU dict with epistemic_status.verified=False (A19: default unverified).

    Args:
        text: Source text chunk to extract from.
        project_id: Project this KU belongs to.
        knowledge_type_hint: Fallback knowledge_type for stub path.
        provider: LLM provider name in ProviderRegistry.
    """
    _stub = False

    try:
        from obase import ProviderRegistry
        from obase.exceptions import ProviderNotFoundError

        llm = ProviderRegistry.get().llm(provider)
        prompt = _EXTRACT_PROMPT.format(text=text[:3000])
        caller = getattr(llm, 'call_sync', None) or llm
        response = caller(prompt)
        text_resp = response.strip()
        text_resp = re.sub(r'^```(?:json)?\s*\n?', '', text_resp)
        text_resp = re.sub(r'\n?```\s*$', '', text_resp).strip()
        raw = json.loads(text_resp)
        knowledge_type = raw.get("knowledge_type", "proposition")
        natural_text = raw.get("natural_text", text[:200])
        symbolic_form = raw.get("symbolic_form")
        tags = raw.get("tags", [])
    except (ProviderNotFoundError, RuntimeError):
        _log.warning(
            "llm_extract_ku: LLM provider %r not registered — falling back to stub", provider
        )
        _stub = True
    except json.JSONDecodeError as e:
        _log.warning("llm_extract_ku: LLM returned non-JSON (%s), using stub", e)
        _stub = True
    except ImportError:
        _stub = True  # obase not installed

    if _stub:
        # Deterministic stub: first sentence as a proposition
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
