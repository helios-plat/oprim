"""Explicit asset/topic tracking graph; edges always retain a provenance reference."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

NodeKind = Literal["asset", "topic", "thesis", "workspace", "report"]
Relation = Literal["tracks", "relates_to", "supports", "contradicts", "mentions"]
_NODE_KINDS = {"asset", "topic", "thesis", "workspace", "report"}
_RELATIONS = {"tracks", "relates_to", "supports", "contradicts", "mentions"}
_NODE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


class TrackedGraphError(ValueError):
    """Invalid or unsupported tracked graph data."""


def _timestamp(value: str | datetime | None = None) -> str:
    moment = datetime.now(UTC) if value is None else (datetime.fromisoformat(value) if isinstance(value, str) else value)
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise TrackedGraphError("timestamps must include a timezone")
    return moment.astimezone(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class TrackedNode:
    node_id: str
    kind: NodeKind
    label: str
    source_ref: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _NODE_ID.fullmatch(self.node_id):
            raise TrackedGraphError("node_id must be a safe identifier")
        if self.kind not in _NODE_KINDS:
            raise TrackedGraphError(f"unsupported node kind: {self.kind}")
        if not self.label.strip() or len(self.label) > 200:
            raise TrackedGraphError("label must contain 1..200 characters")
        if not self.source_ref.strip() or len(self.source_ref) > 2000:
            raise TrackedGraphError("each tracked node requires a source_ref")
        try:
            json.dumps(self.metadata, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise TrackedGraphError("metadata must be JSON-serializable") from exc

    def to_dict(self) -> dict[str, Any]:
        return {"node_id": self.node_id, "kind": self.kind, "label": self.label, "source_ref": self.source_ref, "metadata": self.metadata}


@dataclass(frozen=True)
class TrackedEdge:
    source_id: str
    target_id: str
    relation: Relation
    confidence: float
    evidence_ref: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {"source_id": self.source_id, "target_id": self.target_id, "relation": self.relation, "confidence": self.confidence, "evidence_ref": self.evidence_ref, "created_at": self.created_at}


@dataclass(frozen=True)
class TrackedNeighbor:
    node: TrackedNode
    relation: Relation
    direction: Literal["incoming", "outgoing"]
    confidence: float
    evidence_ref: str


class TrackedGraph:
    """Small deterministic graph for explicitly tracked assets and research topics.

    No relation is inferred from labels or metadata. A caller must supply an
    evidence reference for every node and relationship.
    """

    schema_version = 1

    def __init__(self) -> None:
        self._nodes: dict[str, TrackedNode] = {}
        self._edges: dict[tuple[str, str, str], TrackedEdge] = {}

    def add_node(self, node: TrackedNode) -> None:
        previous = self._nodes.get(node.node_id)
        if previous is not None and previous != node:
            raise TrackedGraphError(f"conflicting definition for node {node.node_id}")
        self._nodes[node.node_id] = node

    def link(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        *,
        evidence_ref: str,
        confidence: float = 1.0,
        created_at: str | datetime | None = None,
    ) -> TrackedEdge:
        if source_id not in self._nodes or target_id not in self._nodes:
            raise TrackedGraphError("both edge endpoints must exist before linking")
        if source_id == target_id:
            raise TrackedGraphError("self-links are not supported")
        if relation not in _RELATIONS:
            raise TrackedGraphError(f"unsupported relationship: {relation}")
        if not isinstance(evidence_ref, str) or not evidence_ref.strip() or len(evidence_ref) > 2000:
            raise TrackedGraphError("each relationship requires an evidence_ref")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise TrackedGraphError("confidence must be a finite number in [0, 1]")
        edge = TrackedEdge(source_id, target_id, relation, float(confidence), evidence_ref.strip(), _timestamp(created_at))
        key = (source_id, target_id, relation)
        previous = self._edges.get(key)
        if previous is not None and previous != edge:
            raise TrackedGraphError(f"conflicting relationship definition: {key}")
        self._edges[key] = edge
        return edge

    def get_node(self, node_id: str) -> TrackedNode:
        try:
            return self._nodes[node_id]
        except KeyError as exc:
            raise TrackedGraphError(f"unknown tracked node: {node_id}") from exc

    def neighbors(
        self,
        node_id: str,
        *,
        direction: Literal["incoming", "outgoing", "both"] = "both",
        relation: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[TrackedNeighbor]:
        self.get_node(node_id)
        if direction not in {"incoming", "outgoing", "both"}:
            raise TrackedGraphError("direction must be incoming, outgoing, or both")
        if relation is not None and relation not in _RELATIONS:
            raise TrackedGraphError(f"unsupported relationship: {relation}")
        if isinstance(min_confidence, bool) or not isinstance(min_confidence, (int, float)) or not 0 <= min_confidence <= 1:
            raise TrackedGraphError("min_confidence must be in [0, 1]")
        found: list[TrackedNeighbor] = []
        for edge in self._edges.values():
            if relation is not None and edge.relation != relation:
                continue
            if edge.confidence < min_confidence:
                continue
            if edge.source_id == node_id and direction in {"outgoing", "both"}:
                found.append(TrackedNeighbor(self._nodes[edge.target_id], edge.relation, "outgoing", edge.confidence, edge.evidence_ref))
            if edge.target_id == node_id and direction in {"incoming", "both"}:
                found.append(TrackedNeighbor(self._nodes[edge.source_id], edge.relation, "incoming", edge.confidence, edge.evidence_ref))
        return sorted(found, key=lambda item: (-item.confidence, item.node.node_id, item.direction, item.relation))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "nodes": [self._nodes[key].to_dict() for key in sorted(self._nodes)],
            "edges": [self._edges[key].to_dict() for key in sorted(self._edges)],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TrackedGraph":
        if value.get("schema_version") != cls.schema_version:
            raise TrackedGraphError("unsupported tracked graph schema_version")
        graph = cls()
        for item in value.get("nodes", []):
            graph.add_node(TrackedNode(**item))
        for item in value.get("edges", []):
            graph.link(**item)
        return graph
