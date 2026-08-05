"""oprim.task_tier — cost-aware task-to-compute-tier classification (pure logic).

3O layer: oprim (single atomic classification, pure heuristics, no LLM).
Maps task metadata + content complexity onto compute tiers so the host can
route cheap work to economy models and hard reasoning to flagship models.
"""

from __future__ import annotations

import re

# Task types that demand flagship-grade reasoning
FLAGSHIP_TASK_TYPES: frozenset[str] = frozenset(
    {
        "3o_architecture",
        "quant_coding",
        "red_team_audit",
        "complex_reasoning",
        "code_generation",
        "architecture_design",
        "security_audit",
        "mathematical_proof",
    }
)
# Task types that are fine on economy models
ECONOMY_TASK_TYPES: frozenset[str] = frozenset(
    {
        "summary_simple",
        "extract_keywords",
        "classify",
        "format",
        "clean",
        "translate",
        "tag",
        "dedupe",
        "general",
    }
)

TIER_FLAGSHIP = "FLAGSHIP"
TIER_ECONOMY = "ECONOMY"

# Content complexity signals: code blocks, math symbols, length
_CODE_BLOCK_RE = re.compile(r"```|def |class |function |=>", re.IGNORECASE)
_MATH_RE = re.compile(r"[∑∫√π±×÷≤≥]|\\frac|\bproof\b", re.IGNORECASE)


def classify_tier(task_type: str = "general", complexity_hint: float | None = None) -> str:
    """Decide the compute tier for a task type (+ optional complexity hint).

    Explicit task type wins; unknown types fall back to the complexity hint
    (>= 0.5 -> flagship), else economy.
    """
    t = str(task_type or "general").lower()
    if t in FLAGSHIP_TASK_TYPES:
        return TIER_FLAGSHIP
    if t in ECONOMY_TASK_TYPES:
        return TIER_ECONOMY
    if complexity_hint is not None:
        return TIER_FLAGSHIP if complexity_hint >= 0.5 else TIER_ECONOMY
    return TIER_ECONOMY  # default: cheap-first


def complexity_score(content: str) -> float:
    """Heuristic 0..1 complexity: code density + math density + length."""
    if not content:
        return 0.0
    n = len(content)
    code_hits = len(_CODE_BLOCK_RE.findall(content))
    math_hits = len(_MATH_RE.findall(content))
    code_density = min(1.0, code_hits / 10.0)
    math_density = min(1.0, math_hits / 5.0)
    length_factor = min(1.0, n / 4000.0)
    return round(max(code_density, math_density, length_factor * 0.7), 3)
