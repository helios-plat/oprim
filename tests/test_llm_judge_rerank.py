import pytest
from unittest.mock import MagicMock
from oprim.llm_judge_rerank import llm_judge_rerank, RerankResult
from oprim._exceptions import OprimError

def test_llm_judge_rerank_basic():
    def mock_llm(**kwargs):
        return {"content": "0: 10\n1: 1"}
    
    docs = ["very relevant", "not relevant"]
    res = llm_judge_rerank(query="test", documents=docs, llm=mock_llm)
    assert len(res) == 2
    assert res[0].original_index == 0
    assert res[0].score == 1.0
    assert res[1].original_index == 1
    assert res[1].score == 0.0

def test_llm_judge_rerank_empty_query():
    with pytest.raises(OprimError, match="Query cannot be empty"):
        llm_judge_rerank(query="   ", documents=["doc1"], llm=MagicMock())

def test_llm_judge_rerank_empty_docs():
    res = llm_judge_rerank(query="test", documents=[], llm=MagicMock())
    assert res == []

def test_llm_judge_rerank_llm_error():
    def mock_llm(**kwargs):
        raise ValueError("LLM is down")
    
    docs = ["doc1", "doc2"]
    res = llm_judge_rerank(query="test", documents=docs, llm=mock_llm)
    assert len(res) == 2
    assert res[0].score == 0.0
    assert res[1].score == 0.0

def test_llm_judge_rerank_partial_missing():
    def mock_llm(**kwargs):
        return {"content": "1: 5"}
    
    docs = ["doc0", "doc1"]
    res = llm_judge_rerank(query="test", documents=docs, llm=mock_llm)
    assert len(res) == 2
    # doc1 gets 5 -> (5-1)/9 = 4/9 = 0.444
    assert res[0].original_index == 1
    assert round(res[0].score, 2) == 0.44
    # doc0 missing -> 0.0
    assert res[1].original_index == 0
    assert res[1].score == 0.0

def test_llm_judge_rerank_top_k():
    def mock_llm(**kwargs):
        return {"content": "0: 2\n1: 8\n2: 10"}
    
    docs = ["doc0", "doc1", "doc2"]
    res = llm_judge_rerank(query="test", documents=docs, llm=mock_llm, top_k=2)
    assert len(res) == 2
    assert res[0].original_index == 2
    assert res[1].original_index == 1
