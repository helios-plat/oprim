import pytest
from unittest.mock import MagicMock
from oprim.llm_query_expand import llm_query_expand
from oprim._exceptions import OprimError

def test_llm_query_expand_basic():
    def mock_llm(**kwargs):
        return {"content": "variant 1\nvariant 2\nvariant 3"}
    
    res = llm_query_expand(query="base", llm=mock_llm, num_variants=3)
    assert len(res) == 4
    assert res[0] == "base"
    assert res[1] == "variant 1"
    assert res[2] == "variant 2"
    assert res[3] == "variant 3"

def test_llm_query_expand_empty_query():
    with pytest.raises(OprimError, match="Query cannot be empty"):
        llm_query_expand(query="  ", llm=MagicMock())

def test_llm_query_expand_invalid_num_variants():
    with pytest.raises(OprimError, match="num_variants must be > 0"):
        llm_query_expand(query="base", llm=MagicMock(), num_variants=0)

def test_llm_query_expand_llm_error():
    def mock_llm(**kwargs):
        raise ValueError("LLM is down")
    
    res = llm_query_expand(query="base", llm=mock_llm, num_variants=2)
    assert len(res) == 1
    assert res[0] == "base"

def test_llm_query_expand_cleanup_and_pad():
    def mock_llm(**kwargs):
        return {"content": "1. var 1\n- var 2"}
    
    res = llm_query_expand(query="base", llm=mock_llm, num_variants=3)
    assert len(res) == 4
    assert res[0] == "base"
    assert res[1] == "var 1"
    assert res[2] == "var 2"
    assert res[3] == "base" # Padded
