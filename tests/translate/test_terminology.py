"""Tests for TerminologyGlossary and TerminologyExtractor."""
import pytest
from oprim.translate.terminology import TerminologyGlossary, TerminologyExtractor


# ── TerminologyGlossary ───────────────────────────────────────────────────────

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


# ── TerminologyExtractor ──────────────────────────────────────────────────────

_SAMPLE_LLM_OUTPUT = """\
夏普比率尽管被广泛采用，但在应用于非正态分布的金融收益时存在若干局限性。

贝利、博维和洛佩斯·德·普拉多（2014）提出了通缩夏普比率（DSR）。

===术语===
Sharpe ratio → 夏普比率
Deflated Sharpe Ratio (DSR) → 通缩夏普比率

===人名地名===
Bailey → 贝利
Lopez de Prado → 洛佩斯·德·普拉多
"""


def test_extractor_finds_terminology():
    terms, proper = TerminologyExtractor.extract_from_response(_SAMPLE_LLM_OUTPUT)
    assert terms["Sharpe ratio"] == "夏普比率"
    assert terms["Deflated Sharpe Ratio (DSR)"] == "通缩夏普比率"


def test_extractor_finds_proper_nouns():
    terms, proper = TerminologyExtractor.extract_from_response(_SAMPLE_LLM_OUTPUT)
    assert proper["Bailey"] == "贝利"
    assert proper["Lopez de Prado"] == "洛佩斯·德·普拉多"


def test_extractor_empty_output():
    terms, proper = TerminologyExtractor.extract_from_response("Just translation text.")
    assert terms == {}
    assert proper == {}


def test_extractor_strips_sections():
    cleaned = TerminologyExtractor.strip_sections(_SAMPLE_LLM_OUTPUT)
    assert "===术语===" not in cleaned
    assert "===人名地名===" not in cleaned
    assert "夏普比率尽管" in cleaned


def test_extractor_strip_noop_without_sections():
    text = "Simple translation without sections."
    assert TerminologyExtractor.strip_sections(text) == text
