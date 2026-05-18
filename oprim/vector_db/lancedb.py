"""LanceDB vector database implementation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import lancedb
import pyarrow as pa

from oprim._logging import log as olog
from oprim.errors import VectorDBError


@dataclass
class VectorRecord:
    id: str
    embedding: list[float]
    metadata: dict


class VectorDB(Protocol):
    def upsert(self, records: list[VectorRecord]) -> None: ...
    def search(
        self,
        query_vec: list[float],
        top_k: int = 20,
        filter: dict | None = None,
    ) -> list[VectorRecord]: ...
    def delete(self, ids: list[str]) -> None: ...
    def count(self) -> int: ...


class LanceDBVectorDB:
    """LanceDB-backed vector store with merge-insert upsert semantics."""

    def __init__(self, db_path: Path, table_name: str, dim: int) -> None:
        self._path = Path(db_path)
        self._table_name = table_name
        self._dim = dim
        self._db = lancedb.connect(str(self._path))
        self._schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("embedding", pa.list_(pa.float32(), dim)),
                pa.field("metadata", pa.string()),
            ]
        )
        self._tbl = self._get_or_create_table()

    def _get_or_create_table(self):
        try:
            existing_tables = self._db.list_tables().tables
            if self._table_name in existing_tables:
                return self._db.open_table(self._table_name)
            empty = pa.table(
                {
                    "id": pa.array([], type=pa.string()),
                    "embedding": pa.array([], type=pa.list_(pa.float32(), self._dim)),
                    "metadata": pa.array([], type=pa.string()),
                }
            )
            return self._db.create_table(
                self._table_name, data=empty, schema=self._schema
            )
        except Exception as e:
            raise VectorDBError(
                f"Failed to open/create table {self._table_name}: {e}"
            ) from e

    def upsert(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        try:
            data = pa.table(
                {
                    "id": [r.id for r in records],
                    "embedding": [[float(x) for x in r.embedding] for r in records],
                    "metadata": [json.dumps(r.metadata) for r in records],
                }
            )
            (
                self._tbl.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(data)
            )
            olog.emit("vector_upsert", table=self._table_name, count=len(records))
        except Exception as e:
            olog.error("vector_upsert failed", error=str(e))
            raise VectorDBError(f"Upsert failed: {e}") from e

    def search(
        self,
        query_vec: list[float],
        top_k: int = 20,
        filter: dict | None = None,
    ) -> list[VectorRecord]:
        try:
            q = self._tbl.search(query_vec).limit(top_k)
            rows = q.to_list()
            results = []
            for row in rows:
                meta: dict = {}
                try:
                    meta = json.loads(row["metadata"])
                except Exception:
                    pass
                results.append(
                    VectorRecord(
                        id=row["id"],
                        embedding=list(row["embedding"]),
                        metadata=meta,
                    )
                )
            return results
        except Exception as e:
            olog.error("vector_search failed", error=str(e))
            raise VectorDBError(f"Search failed: {e}") from e

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        try:
            ids_str = ", ".join(f"'{i}'" for i in ids)
            self._tbl.delete(f"id IN ({ids_str})")
            olog.emit("vector_delete", table=self._table_name, count=len(ids))
        except Exception as e:
            olog.error("vector_delete failed", error=str(e))
            raise VectorDBError(f"Delete failed: {e}") from e

    def count(self) -> int:
        try:
            return self._tbl.count_rows()
        except Exception as e:
            raise VectorDBError(f"Count failed: {e}") from e


def open_vector_db(
    path: Path,
    table_name: str,
    dim: int,
    provider: str = "lancedb",
) -> LanceDBVectorDB:
    """Open (or create) a vector database table.

    Raises:
        VectorDBError: unknown provider.
    """
    if provider != "lancedb":
        raise VectorDBError(f"Unknown vector DB provider: {provider}")
    return LanceDBVectorDB(path, table_name, dim)
