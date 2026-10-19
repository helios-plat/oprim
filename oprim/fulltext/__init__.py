from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oprim.fulltext.tantivy import FulltextDoc, FulltextHit, FulltextIndex, TantivyFulltextIndex


def open_fulltext_index(*args, **kwargs):
    from oprim.fulltext.tantivy import open_fulltext_index as _open

    return _open(*args, **kwargs)


def __getattr__(name: str):
    if name in {"FulltextDoc", "FulltextHit", "FulltextIndex", "TantivyFulltextIndex"}:
        from oprim.fulltext.tantivy import __dict__ as module_dict

        return module_dict[name]
    raise AttributeError(name)


__all__ = [
    "open_fulltext_index",
    "TantivyFulltextIndex",
    "FulltextDoc",
    "FulltextHit",
    "FulltextIndex",
]
