"""Opt-in public-network smoke test; set GLOOM3O_LIVE=1 to execute."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from obase.options_marketdata import OptionsMarketDataAdapter

from oprim.derivatives.analytics import analyze_option_chain


def _evidence_path() -> Path:
    configured = os.environ.get("GLOOM3O_EVIDENCE_DIR")
    if configured:
        return Path(configured) / "options_live.json"
    return Path(__file__).resolve().parents[3] / "evidence" / "options_live.json"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_public_deribit_and_okx_options_and_write_real_evidence() -> None:
    if os.environ.get("GLOOM3O_LIVE") != "1":
        pytest.skip("set GLOOM3O_LIVE=1 to call public exchange APIs")

    adapter = OptionsMarketDataAdapter(timeout=20.0, max_retries=2, cache_ttl=0)
    snapshots = await asyncio.gather(
        adapter.fetch_options("deribit", underlying="BTC"),
        adapter.fetch_options("okx", underlying="BTC"),
    )
    exchanges: dict[str, Any] = {}
    for snapshot in snapshots:
        metrics = analyze_option_chain(snapshot)
        exchanges[snapshot["exchange"]] = {
            "source": (
                "Deribit /api/v2/public/get_instruments + "
                "/api/v2/public/get_book_summary_by_currency + /api/v2/public/get_time"
                if snapshot["exchange"] == "deribit"
                else "OKX /api/v5/public/instruments + /public/open-interest + "
                "/public/opt-summary + /market/index-tickers + /public/time"
            ),
            "provider_timestamp_ms": snapshot["provider_timestamp"],
            "fetched_at": snapshot["fetched_at"],
            "underlying": snapshot["underlying"],
            "spot": snapshot["spot"],
            "instrument_count": len(snapshot["options"]),
            "expiry_count": len(metrics["open_interest_by_expiry"]),
            "open_interest_by_expiry": metrics["open_interest_by_expiry"],
            "max_pain_by_expiry": metrics["max_pain_by_expiry"],
            "put_call_ratio_oi": metrics["put_call_ratio_oi"],
            "atm_iv_by_expiry": metrics["atm_iv_by_expiry"],
            "skew_25d_by_expiry": metrics["skew_25d_by_expiry"],
            "gex_1pct_by_expiry": metrics["gex_1pct_by_expiry"],
            "gex_1pct": metrics["gex_1pct"],
            "gex_assumption": metrics["gex_assumption"],
        }

    assert set(exchanges) == {"deribit", "okx"}
    for summary in exchanges.values():
        assert summary["instrument_count"] > 0
        assert summary["expiry_count"] > 0
        assert summary["provider_timestamp_ms"] > 0
        assert summary["spot"] > 0

    evidence = {
        "captured_at": datetime.now(UTC).isoformat(),
        "capture_method": "live public HTTPS API calls through HTTPS_PROXY",
        "exchanges": exchanges,
    }
    path = _evidence_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    assert path.is_file()
