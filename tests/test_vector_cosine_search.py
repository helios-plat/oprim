from oprim._vector_cosine_search import vector_cosine_search


def test_basic():
    corpus = [
        {"id": "a", "vector": [1.0, 0.0]},
        {"id": "b", "vector": [0.0, 1.0]},
    ]
    res = vector_cosine_search(query=[1.0, 0.0], corpus=corpus, top_k=1)
    assert res[0]["id"] == "a"
    assert res[0]["score"] == 1.0


def test_empty_query():
    assert vector_cosine_search(query=[], corpus=[{"id": "a", "vector": [1.0]}]) == []


def test_top_k():
    corpus = [{"id": str(i), "vector": [float(i), 1.0]} for i in range(5)]
    res = vector_cosine_search(query=[1.0, 1.0], corpus=corpus, top_k=2)
    assert len(res) == 2
