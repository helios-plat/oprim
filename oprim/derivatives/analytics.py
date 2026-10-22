"""Deterministic options-chain indicators.

Gloomberb (https://github.com/gloom-sh/gloomberb, MIT) was reviewed as the
requested reference; no Gloomberb source code is copied here. Black-Scholes
delta/gamma fallbacks follow Black & Scholes (1973), *The Pricing of Options
and Corporate Liabilities*, Journal of Political Economy 81(3), 637-654.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _expiry_label(option: dict[str, Any]) -> str:
    timestamp = _number(option.get("expiration_timestamp"))
    if timestamp and timestamp > 0:
        return datetime.fromtimestamp(timestamp / 1000.0, UTC).strftime("%Y-%m-%d")
    expiry = str(option.get("expiry", "unknown"))
    if len(expiry) == 6 and expiry.isdigit():
        try:
            return datetime.strptime(expiry, "%y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            pass
    return expiry


def _time_to_expiry(option: dict[str, Any], as_of: datetime) -> float:
    timestamp = _number(option.get("expiration_timestamp"))
    if timestamp is None or timestamp <= 0:
        return 30.0 / 365.0
    seconds = timestamp / 1000.0 - as_of.timestamp()
    return max(seconds / (365.0 * 24.0 * 60.0 * 60.0), 1.0 / (365.0 * 24.0))


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _bs_delta_gamma(
    spot: float, strike: float, volatility: float, time_to_expiry: float, option_type: str
) -> tuple[float, float] | None:
    if spot <= 0.0 or strike <= 0.0 or volatility <= 0.0 or time_to_expiry <= 0.0:
        return None
    sigma_sqrt_t = volatility * math.sqrt(time_to_expiry)
    d1 = (math.log(spot / strike) + 0.5 * volatility * volatility * time_to_expiry) / sigma_sqrt_t
    delta = _normal_cdf(d1)
    if option_type == "put":
        delta -= 1.0
    gamma = math.exp(-0.5 * d1 * d1) / (math.sqrt(2.0 * math.pi) * spot * sigma_sqrt_t)
    return delta, gamma


def compute_max_pain(options_chain: list[dict[str, Any]]) -> float | None:
    """Return the settlement strike minimizing aggregate intrinsic OI payout."""
    valid: list[tuple[float, float, str]] = []
    for option in options_chain:
        strike = _number(option.get("strike"))
        interest = _number(option.get("open_interest"))
        option_type = str(option.get("option_type", option.get("type", ""))).lower()
        if strike is None or strike <= 0 or interest is None or interest < 0:
            continue
        if option_type not in ("call", "put"):
            continue
        valid.append((strike, interest, option_type))
    if not valid:
        return None
    strikes = sorted({strike for strike, _, _ in valid})
    pain_by_strike = {
        settle: sum(
            max(settle - strike, 0.0) * interest
            if option_type == "call"
            else max(strike - settle, 0.0) * interest
            for strike, interest, option_type in valid
        )
        for settle in strikes
    }
    return min(strikes, key=lambda strike: (pain_by_strike[strike], strike))


def compute_put_call_ratio(summary: dict[str, Any]) -> dict[str, float | None]:
    """Compute put/call ratio from explicit volumes or normalized chain OI.

    No market values are defaulted. A zero call denominator is represented by
    ``None`` rather than a misleading zero ratio.
    """
    options = summary.get("options")
    if isinstance(options, list):
        puts = sum(
            _number(row.get("open_interest")) or 0.0
            for row in options
            if str(row.get("option_type", row.get("type", ""))).lower() == "put"
        )
        calls = sum(
            _number(row.get("open_interest")) or 0.0
            for row in options
            if str(row.get("option_type", row.get("type", ""))).lower() == "call"
        )
        numerator_name, denominator_name = "puts_open_interest", "calls_open_interest"
    else:
        puts = _number(summary.get("puts_volume"))
        calls = _number(summary.get("calls_volume"))
        if puts is None or calls is None or puts < 0 or calls < 0:
            raise ValueError("provide non-negative puts_volume and calls_volume or an options list")
        numerator_name, denominator_name = "puts_volume", "calls_volume"
    ratio = puts / calls if calls > 0 else None
    return {
        numerator_name: puts,
        denominator_name: calls,
        "put_call_ratio": ratio,
    }


def analyze_option_chain(
    snapshot: dict[str, Any], *, as_of: datetime | None = None
) -> dict[str, Any]:
    """Compute expiry OI, max pain, OI PCR, ATM IV, 25-delta skew, and approximate GEX.

    The GEX sign convention treats call OI as positive and put OI as negative,
    then estimates one-percent spot-move dollar gamma. Open interest alone does
    not identify dealer positioning, so the result is explicitly an exposure
    proxy, not a dealer-position fact.
    """
    options = snapshot.get("options")
    if not isinstance(options, list) or not options:
        raise ValueError("snapshot must contain a non-empty options list")
    now = as_of or datetime.now(UTC)
    spot = _number(snapshot.get("spot"))
    if spot is None or spot <= 0:
        spot_values = [_number(row.get("underlying_price")) for row in options]
        valid_spots = [value for value in spot_values if value is not None and value > 0]
        if valid_spots:
            spot = sum(valid_spots) / len(valid_spots)

    by_expiry: dict[str, list[dict[str, Any]]] = {}
    for option in options:
        strike = _number(option.get("strike"))
        open_interest = _number(option.get("open_interest"))
        option_type = str(option.get("option_type", option.get("type", ""))).lower()
        if strike is None or strike <= 0 or open_interest is None or open_interest < 0:
            continue
        if option_type not in ("call", "put"):
            continue
        by_expiry.setdefault(_expiry_label(option), []).append(option)
    if not by_expiry:
        raise ValueError("no valid option contracts in snapshot")

    oi_by_expiry: dict[str, dict[str, float]] = {}
    max_pain_by_expiry: dict[str, float | None] = {}
    atm_iv_by_expiry: dict[str, dict[str, float] | None] = {}
    skew_by_expiry: dict[str, dict[str, float | None]] = {}
    gex_by_expiry: dict[str, float | None] = {}
    total_puts = 0.0
    total_calls = 0.0

    for expiry, contracts in sorted(by_expiry.items()):
        puts = sum(
            _number(row.get("open_interest")) or 0.0
            for row in contracts
            if str(row.get("option_type", row.get("type", ""))).lower() == "put"
        )
        calls = sum(
            _number(row.get("open_interest")) or 0.0
            for row in contracts
            if str(row.get("option_type", row.get("type", ""))).lower() == "call"
        )
        total_puts += puts
        total_calls += calls
        oi_by_expiry[expiry] = {"calls": calls, "puts": puts, "total": calls + puts}
        max_pain_by_expiry[expiry] = compute_max_pain(contracts)

        if spot is None:
            atm_iv_by_expiry[expiry] = None
            skew_by_expiry[expiry] = {"call_iv": None, "put_iv": None, "put_minus_call": None}
            gex_by_expiry[expiry] = None
            continue

        with_iv = [row for row in contracts if (_number(row.get("mark_iv")) or 0.0) > 0.0]
        if with_iv:
            closest_strike = min(with_iv, key=lambda row: abs(_number(row["strike"]) - spot))
            atm_strike = _number(closest_strike["strike"])
            atm_rows = [
                row for row in with_iv
                if _number(row.get("strike")) == atm_strike
            ]
            atm_iv_by_expiry[expiry] = {
                "strike": float(atm_strike),
                "iv": sum(float(_number(row["mark_iv"])) for row in atm_rows) / len(atm_rows),
            }
        else:
            atm_iv_by_expiry[expiry] = None

        delta_rows: list[tuple[dict[str, Any], float]] = []
        for row in with_iv:
            option_type = str(row.get("option_type", row.get("type", ""))).lower()
            delta = _number(row.get("delta"))
            gamma = _number(row.get("gamma"))
            calculated = _bs_delta_gamma(
                spot,
                float(_number(row["strike"])),
                float(_number(row["mark_iv"])),
                _time_to_expiry(row, now),
                option_type,
            )
            if delta is None and calculated is not None:
                delta = calculated[0]
            if delta is not None:
                delta_rows.append((row, delta))

        call_candidates = [
            (row, delta) for row, delta in delta_rows
            if str(row.get("option_type", row.get("type", ""))).lower() == "call"
        ]
        put_candidates = [
            (row, delta) for row, delta in delta_rows
            if str(row.get("option_type", row.get("type", ""))).lower() == "put"
        ]
        call_25 = (
            min(call_candidates, key=lambda pair: abs(pair[1] - 0.25))[0]
            if call_candidates
            else None
        )
        put_25 = (
            min(put_candidates, key=lambda pair: abs(pair[1] + 0.25))[0]
            if put_candidates
            else None
        )
        call_iv = _number(call_25.get("mark_iv")) if call_25 else None
        put_iv = _number(put_25.get("mark_iv")) if put_25 else None
        skew_by_expiry[expiry] = {
            "call_iv": call_iv,
            "put_iv": put_iv,
            "put_minus_call": (
                put_iv - call_iv if put_iv is not None and call_iv is not None else None
            ),
        }

        gex_value = 0.0
        has_gamma = False
        for row in contracts:
            option_type = str(row.get("option_type", row.get("type", ""))).lower()
            iv = _number(row.get("mark_iv"))
            if iv is None or iv <= 0:
                continue
            calculated = _bs_delta_gamma(
                spot,
                float(_number(row["strike"])),
                iv,
                _time_to_expiry(row, now),
                option_type,
            )
            gamma = _number(row.get("gamma"))
            if gamma is None and calculated is not None:
                gamma = calculated[1]
            if gamma is None or gamma < 0:
                continue
            oi = float(_number(row.get("open_interest")) or 0.0)
            multiplier = float(_number(row.get("contract_multiplier")) or 1.0)
            sign = 1.0 if option_type == "call" else -1.0
            gex_value += sign * gamma * oi * multiplier * spot * spot * 0.01
            has_gamma = True
        gex_by_expiry[expiry] = gex_value if has_gamma else None

    ratio = total_puts / total_calls if total_calls > 0 else None
    gex_values = [value for value in gex_by_expiry.values() if value is not None]
    return {
        "exchange": snapshot.get("exchange"),
        "underlying": snapshot.get("underlying"),
        "as_of": now.astimezone(UTC).isoformat(),
        "spot": spot,
        "open_interest_by_expiry": oi_by_expiry,
        "max_pain_by_expiry": max_pain_by_expiry,
        "put_call_ratio_oi": ratio,
        "atm_iv_by_expiry": atm_iv_by_expiry,
        "skew_25d_by_expiry": skew_by_expiry,
        "gex_1pct_by_expiry": gex_by_expiry,
        "gex_1pct": sum(gex_values) if gex_values else None,
        "gex_assumption": (
            "call OI positive, put OI negative; gamma*OI*contract_multiplier*spot^2*0.01; "
            "OI does not reveal dealer-side positioning"
        ),
    }
