"""Shared graph relation types for AII deep-understanding batch.

Used across oprim (P-AII-3), oskill (K-AII-3/4), and omodul (M-AII-3/4).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RelationCandidate:
    """Result of rule-based relation extraction (P-AII-3)."""
    relation_type: str   # special_case_of/prerequisite_of/basis_of/references/contradicts
    target_ref: str      # target KU name or ID
    evidence: str        # matched pattern / evidence string for traceability
    confidence_signal: str  # "rule_match"|"symbol_dep"|"citation"|"ambiguous"


@dataclass
class RelationResult:
    """Result of LLM-based relation extraction (K-AII-3).

    grade is hardcoded to "unverified" — not settable by callers.
    """
    relation_type: str
    direction: str       # "a_to_b"|"b_to_a"|"bidirectional"
    rationale: str
    grade: str = field(init=False)

    def __post_init__(self) -> None:
        self.grade = "unverified"


@dataclass
class Community:
    """One community produced by community_cluster (K-AII-4)."""
    label: str              # representative ku_id or auto label
    ku_ids: list[str]
    centroid: list[float]   # centroid vector
    size: int


@dataclass
class SummarySynthesizeInput:
    """Input for summary_synthesize (M-AII-3)."""
    ku_ids: list[str]
    ku_texts: list[str]
    source_grades: list[str]


@dataclass
class BookUnderstandingInput:
    """Input for book_understanding_synthesize (M-AII-4)."""
    ku_ids: list[str]
    ku_texts: list[str]
    ku_grades: list[str]


@dataclass
class TheoremVerifyResult:
    """Result of three-way theorem verification (K-AII-5).

    verdict: "verified" | "rejected" | "ambiguous"
    lean_name and type_signature are populated only when verdict=="verified".
    Both come exclusively from mathlib_lookup, never from LLM output.
    """
    verdict: str              # "verified" | "rejected" | "ambiguous"
    lean_name: str | None     # only set when verified
    type_signature: str | None  # only set when verified
    reason: str               # rejection/ambiguity reason; "" when verified
