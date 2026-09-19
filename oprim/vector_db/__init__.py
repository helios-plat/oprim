from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oprim.vector_db.lancedb import LanceDBVectorDB, VectorDB, VectorRecord


def __getattr__(name: str):
    if name in {"LanceDBVectorDB", "VectorDB", "VectorRecord", "open_vector_db"}:
        from oprim.vector_db import lancedb

        return getattr(lancedb, name)
    raise AttributeError(name)


__all__ = ["open_vector_db", "LanceDBVectorDB", "VectorRecord", "VectorDB"]
