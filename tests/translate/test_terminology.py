"""Tests for TerminologyGlossary."""
import pytest
from oprim.translate.terminology import TerminologyGlossary


def test_add_and_retrieve():
    g = TerminologyGlossary()
    g.add("substrate", "底层文档", "en", "zh")
    entries = g.entries_for("en", "zh")
    assert len(entries) == 1
    assert entries[0].source_term == "substrate"
    assert entries[0].target_term == "底层文档"


def test_empty_addendum_when_no_entries():
    g = TerminologyGlossary()
    assert g.build_system_addendum("en", "zh") == ""


def test_addendum_contains_terms():
    g = TerminologyGlossary()
    g.add("wikilink", "维基链接", "en", "zh")
    addendum = g.build_system_addendum("en", "zh")
    assert "wikilink" in addendum
    assert "维基链接" in addendum


def test_protect_and_restore():
    g = TerminologyGlossary()
    g.add("substrate", "底层文档", "en", "zh")
    text = "The substrate is important."
    protected, token_map = g.protect(text, "en", "zh")
    assert "substrate" not in protected
    assert "__TERM_" in protected
    restored = g.restore(protected, token_map)
    assert "底层文档" in restored


def test_protect_no_entries():
    g = TerminologyGlossary()
    text = "Nothing to protect."
    protected, token_map = g.protect(text, "en", "zh")
    assert protected == text
    assert token_map == {}


def test_len():
    g = TerminologyGlossary()
    assert len(g) == 0
    g.add("a", "b", "en", "zh")
    g.add("c", "d", "en", "zh")
    assert len(g) == 2


def test_wrong_lang_not_matched():
    g = TerminologyGlossary()
    g.add("substrate", "底层文档", "en", "zh")
    _, token_map = g.protect("substrate here", "zh", "en")
    assert token_map == {}
