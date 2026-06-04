"""Tests for oprim.backlink_resolver."""

from __future__ import annotations

import pytest

from oprim.backlink_resolver import backlink_resolver


def test_single_link_extracted():
    docs = {"A": "See [[B]] for details."}
    result = backlink_resolver(documents=docs)
    assert "B" in result["forward_links"]["A"]


def test_multiple_links_in_one_doc():
    docs = {"A": "Links to [[B]] and [[C]] and [[D]]."}
    result = backlink_resolver(documents=docs)
    assert set(result["forward_links"]["A"]) == {"B", "C", "D"}


def test_backlinks_built():
    docs = {"A": "See [[B]].", "B": ""}
    result = backlink_resolver(documents=docs)
    assert "A" in result["backlinks"].get("B", [])


def test_unresolved_links_detected():
    docs = {"A": "See [[NonExistent]]."}
    result = backlink_resolver(documents=docs)
    assert "NonExistent" in result["unresolved"]


def test_link_to_existing_doc_not_in_unresolved():
    docs = {"A": "See [[B]].", "B": "Hello."}
    result = backlink_resolver(documents=docs)
    assert "B" not in result["unresolved"]


def test_total_links_count():
    docs = {"A": "[[B]] and [[C]]", "B": "[[C]]"}
    result = backlink_resolver(documents=docs)
    assert result["total_links"] == 3


def test_empty_docs_returns_all_empty():
    result = backlink_resolver(documents={})
    assert result["forward_links"] == {}
    assert result["backlinks"] == {}
    assert result["unresolved"] == []
    assert result["total_links"] == 0


def test_custom_link_pattern():
    docs = {"A": "Ref: {B} and {C}."}
    result = backlink_resolver(documents=docs, link_pattern=r"\{([^}]+)\}")
    assert "B" in result["forward_links"]["A"]
    assert "C" in result["forward_links"]["A"]


def test_duplicate_links_counted_once_per_source():
    docs = {"A": "[[B]] and [[B]] again."}
    result = backlink_resolver(documents=docs)
    # Forward links should deduplicate B per source
    assert result["forward_links"]["A"].count("B") == 1
    assert result["total_links"] == 1


def test_backlinks_reverse_index():
    docs = {"A": "[[C]]", "B": "[[C]]", "C": "Nothing."}
    result = backlink_resolver(documents=docs)
    assert set(result["backlinks"].get("C", [])) == {"A", "B"}


def test_doc_with_no_links_has_empty_forward():
    docs = {"A": "No links here.", "B": "[[A]]"}
    result = backlink_resolver(documents=docs)
    assert result["forward_links"]["A"] == []


def test_total_links_across_multiple_docs():
    docs = {
        "A": "[[B]] [[C]]",
        "B": "[[C]] [[D]]",
        "C": "[[A]]",
    }
    result = backlink_resolver(documents=docs)
    assert result["total_links"] == 5
