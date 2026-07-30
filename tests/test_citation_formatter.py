"""Tests for oprim.citation_formatter."""

from __future__ import annotations

import pytest

from oprim.citation_formatter import citation_formatter

BASE = {
    "authors": ["Smith, J."],
    "title": "A Study of Things",
    "year": 2023,
    "journal": "Journal of Science",
    "volume": "10",
    "issue": "2",
    "pages": "100-110",
}


def test_apa_format_correct():
    result = citation_formatter(citation=BASE, style="apa")
    assert result["error"] is None
    f = result["formatted"]
    assert "Smith" in f
    assert "2023" in f
    assert "Journal of Science" in f
    assert "10" in f
    assert "100-110" in f


def test_mla_format_correct():
    result = citation_formatter(citation=BASE, style="mla")
    assert result["error"] is None
    f = result["formatted"]
    assert "Smith" in f
    assert "A Study of Things" in f
    assert "2023" in f
    assert "Journal of Science" in f


def test_chicago_format_correct():
    result = citation_formatter(citation=BASE, style="chicago")
    assert result["error"] is None
    f = result["formatted"]
    assert "Smith" in f
    assert "A Study of Things" in f
    assert "no." in f
    assert "2023" in f


def test_missing_year_handled_gracefully():
    cit = {**BASE}
    del cit["year"]
    result = citation_formatter(citation=cit, style="apa")
    assert result["error"] is None
    assert "n.d." in result["formatted"]


def test_multiple_authors_et_al():
    cit = {
        **BASE,
        "authors": [
            "Smith, J.",
            "Doe, A.",
            "Brown, B.",
            "White, C.",
            "Green, D.",
            "Black, E.",
            "Gray, F.",
        ],
    }
    result = citation_formatter(citation=cit, style="apa")
    assert "et al." in result["formatted"]


def test_no_journal_uses_publisher():
    cit = {
        "authors": ["Smith, J."],
        "title": "A Book",
        "year": 2020,
        "publisher": "Academic Press",
    }
    result = citation_formatter(citation=cit, style="apa")
    assert result["error"] is None
    assert "Academic Press" in result["formatted"]


def test_url_included():
    cit = {**BASE, "url": "https://example.com/paper"}
    result = citation_formatter(citation=cit, style="apa")
    assert "https://example.com/paper" in result["formatted"]


def test_doi_included():
    cit = {**BASE, "doi": "10.1234/test"}
    result = citation_formatter(citation=cit, style="apa")
    assert "10.1234/test" in result["formatted"]


def test_unknown_style_returns_error():
    result = citation_formatter(citation=BASE, style="harvard")
    assert result["error"] is not None


def test_empty_citation_handled():
    result = citation_formatter(citation={}, style="apa")
    assert result["error"] is not None


def test_formatted_is_nonempty_string():
    result = citation_formatter(citation=BASE, style="apa")
    assert isinstance(result["formatted"], str)
    assert len(result["formatted"]) > 0


def test_style_field_returned():
    result = citation_formatter(citation=BASE, style="mla")
    assert result["style"] == "mla"


def test_doi_preferred_over_url():
    cit = {**BASE, "doi": "10.9999/x", "url": "https://fallback.com"}
    result = citation_formatter(citation=cit, style="apa")
    f = result["formatted"]
    assert "doi.org/10.9999/x" in f
    assert "fallback.com" not in f
