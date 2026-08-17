"""oprim.write_fact_node — one MVCC fact write."""

from __future__ import annotations

from hashlib import sha256

from obase.graph_store.models import GraphDBPool, new_fact


async def write_fact_node(
    entity_id: str,
    *,
    predicate: str,
    object_val: str,
    evidence_chunk: str,
    pool: GraphDBPool,
) -> str:
    """Archive the previous active (entity, predicate) and insert the new fact."""
    evidence_id = sha256(evidence_chunk.encode("utf-8")).hexdigest()[:16]
    old = pool.find_active(entity_id, predicate=predicate)
    fact = new_fact(
        entity_id,
        predicate=predicate,
        object_value=object_val,
        evidence_id=evidence_id,
    )
    await pool.upsert_and_archive(fact, old.node_id if old else None)
    return fact.node_id
