# SELF_CHECK P9-B2 — 4 Alipay oprim

**Date:** 2026-05-31
**Status:** ✅ PASS

---

## §1 — 5 Red Lines

### pytest — 25/25 passed

```
tests/test_alipay_create_qr_order.py      7 passed
tests/test_alipay_query_order.py          6 passed
tests/test_alipay_refund_order.py         6 passed
tests/test_alipay_verify_notify_signature.py  6 passed
Total: 25 passed in 2.12s
```

### Coverage

```
oprim/alipay_create_qr_order.py         95%   (miss: line 77 — edge path)
oprim/alipay_query_order.py            100%
oprim/alipay_refund_order.py           100%
oprim/alipay_verify_notify_signature.py 100%
```

All ≥90% (spec requirement).

### mypy --strict

```
Success: no issues found in 4 source files
```

Note: `[[tool.mypy.overrides]] module = ["alipay"] ignore_missing_imports = true` added to
pyproject.toml (python-alipay-sdk ships no type stubs).

### ruff check

```
All checks passed!
```

### git tag v2.21.0

```
v2.21.0  (pushed to github.com/helios-plat/oprim, shared with P9-B3)
```

---

## §2 — Modules

### `oprim/alipay_create_qr_order.py`

- `AlipayConfig(app_id, app_private_key, alipay_public_key, notify_url, sandbox=False)`
- `AlipayQRCode(qr_code_url, out_trade_no)`
- `AlipayError` / `AlipayAPIError`
- `async alipay_create_qr_order(*, config, out_trade_no, total_amount, subject, body=None) → AlipayQRCode`
- Uses python-alipay-sdk `AliPay` + `asyncio.to_thread`; `sub_code` → `AlipayAPIError`

### `oprim/alipay_query_order.py`

- `AlipayOrderStatus(out_trade_no, trade_status, total_amount, trade_no)` — imports `AlipayConfig` from create
- `async alipay_query_order(*, config, out_trade_no) → AlipayOrderStatus`
- trade_status: WAIT_BUYER_PAY / TRADE_SUCCESS / TRADE_CLOSED / TRADE_FINISHED

### `oprim/alipay_refund_order.py`

- `async alipay_refund_order(*, config, out_trade_no, refund_amount, refund_reason="") → bool`
- Full + partial refund; `sub_code` → `AlipayAPIError`

### `oprim/alipay_verify_notify_signature.py`

- `AlipayInvalidSignatureError(AlipayError)`
- `def alipay_verify_notify_signature(*, config, notify_data) → bool` (SYNC)
- Strips sign/sign_type → `client.verify()` from python-alipay-sdk; SDK exception → `AlipayInvalidSignatureError`; missing `sign` key → `AlipayInvalidSignatureError`

---

## §3 — Commits

```
fc5393d chore: uv.lock update for alipay-sdk + stripe deps (oprim v2.21.0)
3fd0699 feat(alipay+stripe): 8 payment oprim — alipay (4) + stripe (4) — v2.21.0
```

Dependency added: `python-alipay-sdk>=3.3`
