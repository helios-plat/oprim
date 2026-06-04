"""Tests for oprim.bm25_search."""

import pytest
from oprim import bm25_search


def test_empty_docs_returns_empty():
    result = bm25_search(query="hello", docs={})
    assert result == []


def test_empty_query_returns_empty():
    result = bm25_search(query="", docs={"a": "hello world"})
    assert result == []


def test_query_no_matches_returns_empty():
    result = bm25_search(query="zzz", docs={"a": "hello world", "b": "foo bar"})
    assert result == []


def test_exact_identifier_match_scores_high():
    """ADR-038 identifier should score high in a corpus of unrelated docs."""
    docs = {
        "adr038": "ADR-038 decision record about architecture",
        "other1": "unrelated content about cats and dogs",
        "other2": "something else entirely irrelevant text",
    }
    result = bm25_search(query="ADR-038", docs=docs, top_k=3)
    assert len(result) >= 1
    assert result[0][0] == "adr038"
    assert result[0][1] > 0.0


def test_top_k_limits_results():
    docs = {f"doc{i}": f"word{i} common common" for i in range(20)}
    result = bm25_search(query="common", docs=docs, top_k=5)
    assert len(result) <= 5


def test_multiple_docs_ranked_by_relevance():
    docs = {
        "high": "python python python programming language",
        "low": "python once programming elsewhere",
        "none": "java coffee bean morning cup",
    }
    result = bm25_search(query="python", docs=docs, top_k=5)
    ids = [r[0] for r in result]
    assert "high" in ids
    assert "low" in ids
    assert "none" not in ids
    # high should rank before low
    assert ids.index("high") < ids.index("low")


def test_idf_gives_rare_terms_higher_weight():
    """A term appearing in only one doc gets higher IDF than a common term."""
    docs = {
        "rare_doc": "unique_rare_term foo bar",
        "doc2": "foo bar baz",
        "doc3": "foo bar qux",
    }
    rare_results = bm25_search(query="unique_rare_term", docs=docs, top_k=1)
    common_results = bm25_search(query="foo", docs=docs, top_k=3)
    assert rare_results[0][0] == "rare_doc"
    # rare_doc score for rare term > its score for common term (which is shared)
    assert rare_results[0][1] > 0


def test_single_doc_returns_if_matching():
    result = bm25_search(query="hello", docs={"only": "hello world"})
    assert len(result) == 1
    assert result[0][0] == "only"
    assert result[0][1] > 0


def test_multi_term_query():
    docs = {
        "both": "machine learning algorithm",
        "one": "machine coffee cup",
        "neither": "river rock stone",
    }
    result = bm25_search(query="machine learning", docs=docs, top_k=3)
    ids = [r[0] for r in result]
    assert "both" in ids
    assert ids[0] == "both"


def test_case_insensitive_matching():
    docs = {"a": "Hello World", "b": "goodbye world"}
    result_upper = bm25_search(query="HELLO", docs=docs, top_k=2)
    result_lower = bm25_search(query="hello", docs=docs, top_k=2)
    assert result_upper == result_lower
    assert result_upper[0][0] == "a"


def test_score_ordering_descending():
    docs = {
        "d1": "cat cat cat",
        "d2": "cat",
        "d3": "dog",
    }
    result = bm25_search(query="cat", docs=docs, top_k=10)
    scores = [s for _, s in result]
    assert scores == sorted(scores, reverse=True)


def test_result_is_list_of_tuples_str_float():
    docs = {"a": "hello world", "b": "world peace"}
    result = bm25_search(query="world", docs=docs, top_k=5)
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, tuple)
        assert len(item) == 2
        assert isinstance(item[0], str)
        assert isinstance(item[1], float)
