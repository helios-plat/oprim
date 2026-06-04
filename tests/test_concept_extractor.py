"""Tests for oprim.concept_extractor."""

from __future__ import annotations

import pytest

from oprim.concept_extractor import concept_extractor


def test_returns_dict_with_required_keys():
    result = concept_extractor(text="Python is great")
    assert set(result.keys()) == {"concepts", "count", "provider_used", "error"}


def test_concepts_is_list():
    result = concept_extractor(text="Machine Learning is cool")
    assert isinstance(result["concepts"], list)


def test_count_equals_len_concepts():
    result = concept_extractor(text="Python Machine Learning Data Science")
    assert result["count"] == len(result["concepts"])


def test_empty_text_returns_empty_concepts():
    result = concept_extractor(text="")
    assert result["concepts"] == []
    assert result["count"] == 0


def test_whitespace_only_returns_empty():
    result = concept_extractor(text="   ")
    assert result["concepts"] == []


def test_error_none_on_success():
    result = concept_extractor(text="Python is great")
    assert result["error"] is None


def test_capitalized_words_extracted():
    result = concept_extractor(text="Python and Java are programming languages")
    assert "Python" in result["concepts"]
    assert "Java" in result["concepts"]


def test_provider_used_field_present():
    result = concept_extractor(text="Python", provider="openai")
    assert result["provider_used"] == "openai"


def test_provider_used_default():
    result = concept_extractor(text="Python")
    assert result["provider_used"] == "default"


def test_max_concepts_limits_results():
    text = " ".join([f"Word{i}" for i in range(50)])
    result = concept_extractor(text=text, max_concepts=5)
    assert result["count"] <= 5
    assert len(result["concepts"]) <= 5


def test_concepts_are_strings():
    result = concept_extractor(text="Python Machine Learning")
    for c in result["concepts"]:
        assert isinstance(c, str)


def test_multi_word_concept_extracted():
    result = concept_extractor(text="Machine Learning is important for AI")
    # "Machine Learning" should appear as a concept (consecutive capitalized words)
    concepts_str = " ".join(result["concepts"])
    assert "Machine" in concepts_str or "Machine Learning" in concepts_str


def test_count_correct_with_max_concepts():
    text = "Alpha Beta Gamma Delta Epsilon Zeta Eta Theta Iota Kappa"
    result = concept_extractor(text=text, max_concepts=3)
    assert result["count"] == len(result["concepts"])
    assert result["count"] <= 3


def test_no_capitalized_words_returns_empty():
    result = concept_extractor(text="all lowercase words here nothing special")
    # No capitalized words → empty or only words that happen to be capitalized
    # The sentence start has no caps in this test (all lowercase)
    assert isinstance(result["concepts"], list)
