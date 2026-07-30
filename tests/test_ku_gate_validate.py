"""Tests for oprim.ku_gate_validate."""

import pytest
from oprim.ku_gate_validate import ku_gate_validate


def _valid_proposition() -> dict:
    return {
        "ku_id": "abc-123",
        "knowledge_type": "proposition",
        "natural_text": "Water boils at 100 degrees Celsius at sea level.",
        "symbolic_form": None,
        "vector": None,
        "vector_frozen": False,
        "epistemic_status": {
            "grade": "moderate",
            "source": "wikipedia",
            "defeaters": [],
            "verified": False,
        },
        "provenance": {"source": "test", "chunk_id": None},
        "project_id": "proj-001",
    }


def test_valid_proposition_passes():
    result = ku_gate_validate(ku=_valid_proposition())
    assert result["valid"] is True
    assert result["errors"] == []


def test_missing_natural_text_error():
    ku = _valid_proposition()
    del ku["natural_text"]
    result = ku_gate_validate(ku=ku)
    assert result["valid"] is False
    assert any("natural_text" in e for e in result["errors"])


def test_empty_natural_text_error():
    ku = _valid_proposition()
    ku["natural_text"] = "   "
    result = ku_gate_validate(ku=ku)
    assert result["valid"] is False
    assert any("natural_text" in e for e in result["errors"])


def test_missing_epistemic_status_error():
    ku = _valid_proposition()
    del ku["epistemic_status"]
    result = ku_gate_validate(ku=ku)
    assert result["valid"] is False
    assert any("epistemic_status" in e for e in result["errors"])


def test_invalid_knowledge_type_error():
    ku = _valid_proposition()
    ku["knowledge_type"] = "fantasy"
    result = ku_gate_validate(ku=ku)
    assert result["valid"] is False
    assert any("knowledge_type" in e for e in result["errors"])


def test_invalid_grade_error():
    ku = _valid_proposition()
    ku["epistemic_status"]["grade"] = "mythical"
    result = ku_gate_validate(ku=ku)
    assert result["valid"] is False
    assert any("grade" in e for e in result["errors"])


def test_theorem_without_symbolic_form_error():
    ku = _valid_proposition()
    ku["knowledge_type"] = "theorem"
    ku["symbolic_form"] = None
    result = ku_gate_validate(ku=ku)
    assert result["valid"] is False
    assert any("symbolic_form" in e for e in result["errors"])


def test_proposition_without_symbolic_form_passes():
    ku = _valid_proposition()
    ku["symbolic_form"] = None
    result = ku_gate_validate(ku=ku)
    assert result["valid"] is True
    assert result["errors"] == []


def test_missing_project_id_warning_not_error():
    ku = _valid_proposition()
    del ku["project_id"]
    result = ku_gate_validate(ku=ku)
    assert result["valid"] is True
    assert result["errors"] == []
    assert any("project_id" in w for w in result["warnings"])


def test_valid_rule_with_symbolic_form_passes():
    ku = _valid_proposition()
    ku["knowledge_type"] = "rule"
    ku["symbolic_form"] = {"if": "condition", "then": "action"}
    result = ku_gate_validate(ku=ku)
    assert result["valid"] is True
    assert result["errors"] == []


def test_multiple_errors_reported_together():
    ku = {}
    result = ku_gate_validate(ku=ku)
    assert result["valid"] is False
    # Should have at least natural_text, knowledge_type, and epistemic_status errors
    assert len(result["errors"]) >= 3


def test_warnings_key_always_present():
    ku = _valid_proposition()
    result = ku_gate_validate(ku=ku)
    assert "warnings" in result


def test_errors_key_always_present():
    ku = _valid_proposition()
    result = ku_gate_validate(ku=ku)
    assert "errors" in result


def test_formula_without_symbolic_form_error():
    ku = _valid_proposition()
    ku["knowledge_type"] = "formula"
    ku["symbolic_form"] = None
    result = ku_gate_validate(ku=ku)
    assert result["valid"] is False
    assert any("symbolic_form" in e for e in result["errors"])


def test_valid_formula_with_symbolic_form_passes():
    ku = _valid_proposition()
    ku["knowledge_type"] = "formula"
    ku["symbolic_form"] = {
        "expression": "E = mc^2",
        "variables": {"E": "energy", "m": "mass", "c": "speed_of_light"},
    }
    result = ku_gate_validate(ku=ku)
    assert result["valid"] is True
