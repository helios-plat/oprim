from oprim.fulltext.tantivy import FulltextDoc, FulltextHit, FulltextIndex, TantivyFulltextIndex, open_fulltext_index

__all__ = [
    "open_fulltext_index",
    "open_elasticsearch_index",
    "open_codegraph_index",
    "TantivyFulltextIndex",
    "ElasticsearchFulltextIndex",
    "CodeGraphFulltextIndex",
    "FulltextDoc",
    "FulltextHit",
    "FulltextIndex",
]


def __getattr__(name: str):
    if name in {"ElasticsearchFulltextIndex", "open_elasticsearch_index"}:
        from oprim.fulltext.elasticsearch import ElasticsearchFulltextIndex, open_elasticsearch_index

        return {"ElasticsearchFulltextIndex": ElasticsearchFulltextIndex,
                "open_elasticsearch_index": open_elasticsearch_index}[name]
    if name in {"CodeGraphFulltextIndex", "open_codegraph_index"}:
        from oprim.fulltext.codegraph import CodeGraphFulltextIndex, open_codegraph_index

        return {"CodeGraphFulltextIndex": CodeGraphFulltextIndex,
                "open_codegraph_index": open_codegraph_index}[name]
    raise AttributeError(name)
