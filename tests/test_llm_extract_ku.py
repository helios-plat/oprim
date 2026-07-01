"""Tests for oprim.llm_extract_ku — uses stub path (no actual LLM)."""

import uuid
import pytest
from oprim.llm_extract_ku import llm_extract_ku
from oprim.ku_gate_validate import VALID_KNOWLEDGE_TYPES


def test_returns_dict_with_all_required_ku_keys():
    result = llm_extract_ku(text="The sky is blue on a clear day.")
    required_keys = {
        "ku_id",
        "knowledge_type",
        "natural_text",
        "symbolic_form",
        "vector",
        "vector_frozen",
        "epistemic_status",
        "provenance",
        "project_id",
        "tags",
    }
    assert required_keys.issubset(set(result.keys()))


def test_epistemic_status_verified_is_false():
    result = llm_extract_ku(text="Gravity pulls objects toward the Earth.")
    assert result["epistemic_status"]["verified"] is False


def test_epistemic_status_grade_is_unverified():
    result = llm_extract_ku(text="Light travels at approximately 300,000 km/s.")
    assert result["epistemic_status"]["grade"] == "unverified"


def test_knowledge_type_is_valid():
    result = llm_extract_ku(text="Python is a dynamically typed language.")
    assert result["knowledge_type"] in VALID_KNOWLEDGE_TYPES


def test_natural_text_is_non_empty_string():
    result = llm_extract_ku(text="Neurons fire in response to stimulation.")
    assert isinstance(result["natural_text"], str)
    assert len(result["natural_text"]) > 0


def test_ku_id_is_uuid_string():
    result = llm_extract_ku(text="DNA carries genetic information.")
    # Should parse as a valid UUID
    parsed = uuid.UUID(result["ku_id"])
    assert str(parsed) == result["ku_id"]


def test_vector_is_none():
    result = llm_extract_ku(text="Mars is the fourth planet from the Sun.")
    assert result["vector"] is None


def test_project_id_matches_input():
    result = llm_extract_ku(text="Some knowledge here.", project_id="my-project")
    assert result["project_id"] == "my-project"


def test_knowledge_type_hint_influences_fallback_type():
    # obase is not available so stub path runs; hint should set knowledge_type
    result = llm_extract_ku(text="If condition then action.", knowledge_type_hint="rule")
    assert result["knowledge_type"] == "rule"


def test_different_texts_produce_different_ku_ids():
    r1 = llm_extract_ku(text="The cat sat on the mat.")
    r2 = llm_extract_ku(text="The dog ran through the park.")
    assert r1["ku_id"] != r2["ku_id"]


def test_provenance_source_is_llm_extract():
    result = llm_extract_ku(text="Photosynthesis converts light into energy.")
    assert result["provenance"]["source"] == "llm_extract"


def test_epistemic_status_has_defeaters_list():
    result = llm_extract_ku(text="Temperature affects reaction rates.")
    assert isinstance(result["epistemic_status"]["defeaters"], list)


def test_default_project_id_is_default():
    result = llm_extract_ku(text="Some fact about the world.")
    assert result["project_id"] == "default"


def test_vector_frozen_is_false():
    result = llm_extract_ku(text="Knowledge units are structured representations.")
    assert result["vector_frozen"] is False


# ---------------------------------------------------------------------------
# ProviderRegistry fix tests (v2.29.1)
# ---------------------------------------------------------------------------


class TestProviderRegistryPath:
    def test_provider_not_registered_falls_back_to_stub_with_warning(self):
        """ProviderNotFoundError → stub + log.warning, no raise."""
        import logging
        from unittest.mock import patch
        from obase.exceptions import ProviderNotFoundError

        with patch("obase.ProviderRegistry.get", side_effect=ProviderNotFoundError("llm", "missing")):
            with self._capture_warnings() as records:
                result = llm_extract_ku(text="The sky is blue.")
        assert result["natural_text"]  # stub produced output
        assert result["epistemic_status"]["verified"] is False
        assert any("not registered" in r.message for r in records)

    def test_provider_registered_calls_llm_with_prompt(self):
        """Registered provider is called; prompt contains the input text."""
        from unittest.mock import patch, MagicMock
        import json

        fake_response = json.dumps({
            "knowledge_type": "proposition",
            "natural_text": "Extracted claim from LLM.",
            "symbolic_form": None,
            "tags": ["test"],
        })
        mock_llm = MagicMock(return_value=fake_response)
        with patch("obase.ProviderRegistry.get", return_value=mock_llm) as mock_get:
            result = llm_extract_ku(text="Some text chunk.", provider="deepseek")
        mock_get.assert_called_once_with("llm", "deepseek")
        mock_llm.assert_called_once()
        prompt_used = mock_llm.call_args[0][0]
        assert "Some text chunk." in prompt_used
        assert result["natural_text"] == "Extracted claim from LLM."

    def test_code_error_reraises(self):
        """Non-ProviderNotFoundError must propagate, not fall to stub."""
        import pytest
        from unittest.mock import patch

        with patch("obase.ProviderRegistry.get", side_effect=TypeError("bad type")):
            with pytest.raises(TypeError):
                llm_extract_ku(text="hello")

    @staticmethod
    def _capture_warnings():
        import logging
        from unittest.mock import patch
        import contextlib

        class _Records:
            def __init__(self):
                self.message = []
            def __iter__(self):
                return iter(self.message)

        @contextlib.contextmanager
        def _ctx():
            records = []
            class _R:
                def __init__(self, msg): self.message = msg
            with patch("oprim.llm_extract_ku._log") as mock_log:
                mock_log.warning.side_effect = lambda msg, *a, **k: records.append(_R(msg % a if a else msg))
                yield records

        return _ctx()
