"""Pure market-evidence date, freshness, provenance, and conflict primitives.

Inspired by strict market-MCP contracts; implemented independently with no
network, storage, model, or sibling-element dependencies.
"""
from __future__ import annotations

__oprim_layer__ = "element"

import math
import re
from collections import defaultdict
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qsl, urlsplit

_SOURCE_AUTHORITY = {
    "primary_exchange": 70,
    "official_exchange": 70,
    "regulator": 70,
    "official_filing": 65,
    "official_company": 60,
    "market_api": 55,
    "newswire": 45,
    "news": 35,
    "research": 30,
    "blog": 20,
    "unknown": 10,
}
_DATE_FORMAT = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _parse_market_date(value: Any, *, field: str) -> date:
    if not isinstance(value, str) or not _DATE_FORMAT.fullmatch(value):
        raise ValueError(f"{field} must use strict YYYY-MM-DD format")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} is not a valid calendar date") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{field} must use strict YYYY-MM-DD format")
    return parsed


def validate_market_date_window(
    start_date: str,
    *,
    end_date: str,
    max_days: int = 3660,
    today: date | None = None,
) -> dict[str, Any]:
    """Validate a bounded historical market window before vendor IO.

    `today` is injectable to make callers and tests deterministic. End dates
    after the UTC/current reference date are rejected rather than silently
    falling back to a vendor default range.
    """
    start = _parse_market_date(start_date, field="start_date")
    end = _parse_market_date(end_date, field="end_date")
    reference_day = today or datetime.now(timezone.utc).date()
    if not isinstance(reference_day, date) or isinstance(reference_day, datetime):
        raise ValueError("today must be a date instance, not a datetime")
    if isinstance(max_days, bool) or not isinstance(max_days, int) or not 1 <= max_days <= 3660:
        raise ValueError("max_days must be an integer between 1 and 3660")
    if end < start:
        raise ValueError("end_date must be greater than or equal to start_date")
    if end > reference_day:
        raise ValueError("end_date cannot be in the future for historical market data")
    days = (end - start).days + 1
    if days > max_days:
        raise ValueError(f"date window spans {days} calendar days; max_days is {max_days}")
    return {
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "calendar_days": days,
        "inclusive": True,
    }


def score_evidence_source(
    source: Mapping[str, Any],
    *,
    now_utc: datetime | None = None,
    stale_after_hours: float = 24.0,
) -> dict[str, Any]:
    """Return an explainable 0–100 quality index, not a probability of truth."""
    if not isinstance(source, Mapping):
        return {"score": 0.0, "status": "unverified", "reasons": ["source_not_a_mapping"]}
    uri = source.get("source_uri") or source.get("url")
    if not isinstance(uri, str) or not uri.strip():
        return {"score": 0.0, "status": "unverified", "reasons": ["missing_source_uri"]}
    try:
        parts = urlsplit(uri.strip())
        if parts.scheme.lower() not in {"http", "https", "file", "artifact", "local"}:
            return {"score": 0.0, "status": "unverified", "reasons": ["unsupported_source_uri_scheme"]}
        if parts.scheme.lower() in {"http", "https"} and (not parts.hostname or parts.username is not None or parts.password is not None):
            return {"score": 0.0, "status": "unverified", "reasons": ["source_uri_host_or_credentials_invalid"]}
        if any(key.lower() in {"access_token", "api_key", "apikey", "auth", "authorization", "password", "secret", "signature", "token"} for key, _ in parse_qsl(parts.query, keep_blank_values=True)):
            return {"score": 0.0, "status": "unverified", "reasons": ["source_uri_contains_credential_parameter"]}
    except ValueError:
        return {"score": 0.0, "status": "unverified", "reasons": ["invalid_source_uri"]}
    timestamp = source.get("retrieved_at_utc") or source.get("retrieved_at")
    if not isinstance(timestamp, str):
        return {"score": 0.0, "status": "unverified", "reasons": ["missing_retrieval_timestamp"]}
    try:
        retrieved = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return {"score": 0.0, "status": "unverified", "reasons": ["invalid_retrieval_timestamp"]}
    if retrieved.tzinfo is None or retrieved.utcoffset() is None:
        return {"score": 0.0, "status": "unverified", "reasons": ["timestamp_missing_timezone"]}
    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        return {"score": 0.0, "status": "unverified", "reasons": ["reference_time_missing_timezone"]}
    if isinstance(stale_after_hours, bool) or not isinstance(stale_after_hours, (int, float)) or not math.isfinite(stale_after_hours) or stale_after_hours <= 0:
        return {"score": 0.0, "status": "unverified", "reasons": ["invalid_stale_after_hours"]}
    age_hours = (now.astimezone(timezone.utc) - retrieved.astimezone(timezone.utc)).total_seconds() / 3600
    if age_hours < -5 / 60:
        return {"score": 0.0, "status": "unverified", "reasons": ["retrieval_timestamp_in_future"]}
    source_type = str(source.get("source_type", "unknown")).strip().lower()
    authority = _SOURCE_AUTHORITY.get(source_type, _SOURCE_AUTHORITY["unknown"])
    freshness = 20.0 * max(0.0, 1.0 - max(0.0, age_hours) / float(stale_after_hours))
    raw_directness = source.get("directness", 0.5)
    try:
        directness = float(raw_directness)
    except (TypeError, ValueError):
        directness = -1.0
    if not math.isfinite(directness) or not 0.0 <= directness <= 1.0:
        return {"score": 0.0, "status": "unverified", "reasons": ["directness_must_be_between_0_and_1"]}
    status = "stale" if age_hours > stale_after_hours else "current"
    reasons = [f"source_type:{source_type or 'unknown'}", "timestamp_checked"]
    if status == "stale":
        reasons.append("source_is_stale")
    if "error" in source and source.get("error"):
        return {"score": 0.0, "status": "unverified", "reasons": reasons + ["source_has_error"]}
    return {
        "score": round(min(100.0, authority + freshness + 10.0 * directness), 2),
        "status": status,
        "source_type": source_type or "unknown",
        "age_hours": round(max(0.0, age_hours), 4),
        "authority_points": authority,
        "freshness_points": round(freshness, 2),
        "directness_points": round(10.0 * directness, 2),
        "reasons": reasons,
    }


def _decimal_or_none(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = Decimal(str(value).strip())
    except (InvalidOperation, ValueError, AttributeError):
        return None
    return parsed if parsed.is_finite() else None


def detect_claim_conflicts(
    claims: Sequence[Mapping[str, Any]],
    *,
    numeric_tolerance: float = 0.0,
) -> list[dict[str, Any]]:
    """Find disagreements for identical subject/predicate/unit claim keys.

    Each claim must include `subject`, `predicate`, `value`, and `source_id`.
    Numeric facts conflict only when the max/min difference exceeds tolerance;
    non-numeric facts conflict when case/whitespace-normalized values disagree.
    """
    if not isinstance(claims, Sequence) or isinstance(claims, (str, bytes)):
        raise ValueError("claims must be a sequence of claim mappings")
    if isinstance(numeric_tolerance, bool) or not isinstance(numeric_tolerance, (int, float)) or not math.isfinite(numeric_tolerance) or numeric_tolerance < 0:
        raise ValueError("numeric_tolerance must be a finite non-negative number")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for index, claim in enumerate(claims):
        if not isinstance(claim, Mapping):
            raise ValueError(f"claim at index {index} must be a mapping")
        missing = [key for key in ("subject", "predicate", "value", "source_id") if key not in claim]
        if missing:
            raise ValueError(f"claim at index {index} is missing required fields: {', '.join(missing)}")
        if claim["value"] is None:
            raise ValueError(f"claim at index {index} value cannot be null")
        subject = str(claim["subject"]).strip().casefold()
        predicate = str(claim["predicate"]).strip().casefold()
        unit = str(claim.get("unit", "")).strip().casefold()
        if not subject or not predicate or not str(claim["source_id"]).strip():
            raise ValueError(f"claim at index {index} has an empty subject, predicate, or source_id")
        groups[(subject, predicate, unit)].append({
            "subject": str(claim["subject"]).strip(),
            "predicate": str(claim["predicate"]).strip(),
            "unit": str(claim.get("unit", "")).strip(),
            "value": claim["value"],
            "source_id": str(claim["source_id"]).strip(),
        })

    conflicts: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = groups[key]
        values = [item["value"] for item in group]
        numeric_values = [_decimal_or_none(value) for value in values]
        if all(value is not None for value in numeric_values):
            distinct_conflict = max(numeric_values) - min(numeric_values) > Decimal(str(numeric_tolerance))
            reason = "numeric_values_exceed_tolerance"
            rendered_values = sorted({format(value, "f") for value in numeric_values})
        else:
            rendered_values = sorted({" ".join(str(value).split()).casefold() for value in values})
            distinct_conflict = len(rendered_values) > 1
            reason = "categorical_values_disagree"
        if distinct_conflict:
            conflicts.append({
                "subject": group[0]["subject"],
                "predicate": group[0]["predicate"],
                "unit": group[0]["unit"],
                "values": rendered_values,
                "source_ids": sorted({item["source_id"] for item in group}),
                "claim_count": len(group),
                "status": "conflict",
                "reason": reason,
            })
    return conflicts
