"""Tests for oprim.llm_distill_strategy — uses stub path (no actual LLM)."""

import uuid
import pytest
from oprim._llm_distill_strategy import llm_distill_strategy


def _episode(**kwargs) -> dict:
    base = {
        "episode_id": "ep-001",
        "event": "User reported a memory leak in the service.",
        "outcome": "Fixed by releasing unused references in the cache layer.",
        "context": {"service": "cache-service", "environment": "production"},
    }
    base.update(kwargs)
    return base


def test_returns_knowledge_type_solution_strategy():
    result = llm_distill_strategy(episode=_episode())
    assert result["knowledge_type"] == "solution_strategy"


def test_epistemic_status_verified_is_false():
    result = llm_distill_strategy(episode=_episode())
    assert result["epistemic_status"]["verified"] is False


def test_grade_is_unverified():
    result = llm_distill_strategy(episode=_episode())
    assert result["epistemic_status"]["grade"] == "unverified"


def test_symbolic_form_has_title_description_content():
    result = llm_distill_strategy(episode=_episode())
    sf = result["symbolic_form"]
    assert isinstance(sf, dict)
    assert "title" in sf
    assert "description" in sf
    assert "content" in sf


def test_natural_text_is_non_empty():
    result = llm_distill_strategy(episode=_episode())
    assert isinstance(result["natural_text"], str)
    assert len(result["natural_text"]) > 0


def test_ku_id_is_uuid_string():
    result = llm_distill_strategy(episode=_episode())
    parsed = uuid.UUID(result["ku_id"])
    assert str(parsed) == result["ku_id"]


def test_vector_is_none():
    result = llm_distill_strategy(episode=_episode())
    assert result["vector"] is None


def test_project_id_matches_input():
    result = llm_distill_strategy(episode=_episode(), project_id="aii-proj")
    assert result["project_id"] == "aii-proj"


def test_episode_with_event_and_outcome_produces_meaningful_title():
    result = llm_distill_strategy(episode=_episode())
    title = result["symbolic_form"]["title"]
    assert len(title) > 0
    # Stub path: title contains outcome fragment or "Strategy"
    assert "Strategy" in title or len(title) > 5


def test_provenance_source_is_llm_distill():
    result = llm_distill_strategy(episode=_episode())
    assert result["provenance"]["source"] == "llm_distill"


def test_provenance_episode_id_propagated():
    result = llm_distill_strategy(episode=_episode())
    assert result["provenance"]["episode_id"] == "ep-001"


def test_empty_episode_handled_gracefully():
    result = llm_distill_strategy(episode={})
    assert result["knowledge_type"] == "solution_strategy"
    assert isinstance(result["natural_text"], str)
    assert result["symbolic_form"] is not None


def test_different_episodes_produce_different_ku_ids():
    r1 = llm_distill_strategy(episode=_episode(event="Event A", outcome="Outcome A"))
    r2 = llm_distill_strategy(episode=_episode(event="Event B", outcome="Outcome B"))
    assert r1["ku_id"] != r2["ku_id"]


def test_default_project_id_is_default():
    result = llm_distill_strategy(episode=_episode())
    assert result["project_id"] == "default"


def test_vector_frozen_is_false():
    result = llm_distill_strategy(episode=_episode())
    assert result["vector_frozen"] is False


def test_epistemic_status_has_defeaters_list():
    result = llm_distill_strategy(episode=_episode())
    assert isinstance(result["epistemic_status"]["defeaters"], list)


# ---------------------------------------------------------------------------
# ProviderRegistry fix tests (v2.29.1)
# ---------------------------------------------------------------------------


class TestProviderRegistryPath:
    def test_provider_not_registered_falls_back_to_stub_with_warning(self):
        """ProviderNotFoundError → stub + log.warning, no raise."""
        from unittest.mock import patch
        from obase.exceptions import ProviderNotFoundError

        ep = {"event": "deploy failed", "outcome": "rollback", "context": {}}
        with patch("obase.ProviderRegistry.get", side_effect=ProviderNotFoundError("llm", "x")):
            with patch("oprim._llm_distill_strategy._log") as mock_log:
                result = llm_distill_strategy(episode=ep)
        assert result["knowledge_type"] == "solution_strategy"
        assert result["epistemic_status"]["verified"] is False
        mock_log.warning.assert_called_once()

    def test_provider_registered_calls_llm_messages_passthrough(self):
        """Registered provider called; episode fields appear in prompt."""
        from unittest.mock import patch, MagicMock
        import json

        fake_response = json.dumps({
            "title": "Rollback strategy",
            "description": "Use rollback on failure.",
            "content": "Steps: 1. detect 2. rollback",
        })
        mock_llm = MagicMock(return_value=fake_response)
        mock_registry = MagicMock()
        mock_registry.llm.return_value = mock_llm
        ep = {"event": "prod deploy", "outcome": "success", "context": {"env": "prod"}}
        with patch("obase.ProviderRegistry.get", return_value=mock_registry) as mock_get:
            result = llm_distill_strategy(episode=ep, provider="deepseek")
        mock_get.assert_called_once_with()
        mock_registry.llm.assert_called_once_with("deepseek")
        prompt = mock_llm.call_args[0][0]
        assert "prod deploy" in prompt
        assert result["symbolic_form"]["title"] == "Rollback strategy"

    def test_code_error_reraises(self):
        """Non-ProviderNotFoundError must propagate."""
        import pytest
        from unittest.mock import patch

        with patch("obase.ProviderRegistry.get", side_effect=ValueError("bad")):
            with pytest.raises(ValueError):
                llm_distill_strategy(episode={})
