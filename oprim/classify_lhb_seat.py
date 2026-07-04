"""龙虎榜(LHB)席位分类 — 暴露 8 类, 3 类为无需配置的结构性投影.

Ported from Tide's ``domain.lhb.seat_service.classify_seat`` (the DB-bound
``SeatService``/relay-chain shell stays in Tide). Tide's original hardcodes
its own curated seat/keyword lists (``a_share_config.py``) — proprietary,
evolving market knowledge that doesn't belong baked into a generic quant
primitives library. Here that data is instead **injected** via a ``registry``
dict at call time; without one, only the 3 structural classes that need no
market-knowledge config are reachable (northbound / institutional / retail).

ASSUMPTION (flagged for review): the registry-injection design itself — Tide
must now pass its ``a_share_config`` lists in via ``registry=`` rather than
oprim importing them directly.
"""

from __future__ import annotations

from typing import Any

# All 8 seat classes classify_lhb_seat() can emit. northbound / institutional /
# retail are structural projections reachable with no registry at all.
SEAT_CLASSES: tuple[str, ...] = (
    "institutional",
    "northbound",
    "foreign_institution",
    "quant_lhasa",
    "internet_brokerage",
    "tycoon",
    "quant",
    "retail",
)


def _seat_alias(trader_name: str, registry: dict[str, Any]) -> str | None:
    """Return a market-known 游资 alias for a seat name, or None."""
    overrides = registry.get("overrides", {})
    override = overrides.get(trader_name)
    if override and override.get("alias"):
        return override["alias"]
    known_aliases = registry.get("tycoon_aliases", {})
    alias = known_aliases.get(trader_name)
    if alias:
        return alias
    for kw, label in registry.get("tycoon_alias_keywords", {}).items():
        if kw in trader_name:
            return label
    return None


def classify_lhb_seat(
    trader_name: str,
    *,
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify a LHB trader seat into one of 8 classes.

    Priority: northbound > foreign > quant_lhasa > internet > institutional >
    tycoon > quant > retail.

    Args:
        trader_name: 营业部/交易者全称.
        registry: Optional market-knowledge lists to classify against —
            ``overrides`` (dict[name, {classification, alias}], hot-updatable,
            wins over every rule below), ``foreign_institutions`` (list[str]),
            ``quant_lhasa_seats`` (list[str]), ``quant_lhasa_keywords`` (list[str]),
            ``internet_brokerages`` (list[str]), ``internet_brokerage_keywords``
            (list[str]), ``tycoon_seats`` (list[str]), ``tycoon_aliases``
            (dict[name, alias]), ``tycoon_alias_keywords`` (dict[kw, alias]),
            ``quant_keywords`` (list[str]). Missing keys default to empty —
            without a registry, only northbound/institutional/retail are reachable.

    Returns:
        dict with classification (one of SEAT_CLASSES), confidence, alias.
    """
    reg = registry or {}

    if not trader_name:
        return {"classification": "retail", "confidence": 0.5, "alias": None}

    name = trader_name.strip()
    alias = _seat_alias(name, reg)

    def _r(classification: str, confidence: float) -> dict[str, Any]:
        return {"classification": classification, "confidence": confidence, "alias": alias}

    # 0. Registry override wins over every hardcoded rule (hot-updatable).
    override = reg.get("overrides", {}).get(name)
    if override and override.get("classification"):
        return _r(str(override["classification"]), 1.0)

    # 1. Northbound (highest priority) — structural, no registry needed.
    if any(k in name for k in ("深股通", "沪股通", "陆股通")):
        return _r("northbound", 1.0)

    # 2. Foreign institution
    for fi in reg.get("foreign_institutions", []):
        if fi in name or name in fi:
            return _r("foreign_institution", 1.0)

    # 3. Quant Lhasa
    if name in reg.get("quant_lhasa_seats", []):
        return _r("quant_lhasa", 1.0)
    for kw in reg.get("quant_lhasa_keywords", []):
        if kw in name and "东方财富" in name:
            return _r("quant_lhasa", 0.9)

    # 4. Internet brokerage
    if name in reg.get("internet_brokerages", []):
        return _r("internet_brokerage", 1.0)
    for kw in reg.get("internet_brokerage_keywords", []):
        if kw in name:
            return _r("internet_brokerage", 0.85)

    # 5. Institutional — structural, no registry needed.
    if "机构专用" in name:
        return _r("institutional", 1.0)

    # 6. Tycoon (known seat list OR a recognised 游资 alias keyword)
    tycoon_seats = reg.get("tycoon_seats", [])
    if name in tycoon_seats or alias is not None:
        return _r("tycoon", 1.0 if name in tycoon_seats else 0.8)

    # 7. Quant
    for kw in reg.get("quant_keywords", []):
        if kw in name:
            return _r("quant", 0.9)

    # 8. Retail (default) — structural, no registry needed.
    return _r("retail", 0.6)
