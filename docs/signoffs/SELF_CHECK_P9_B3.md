# SELF_CHECK P9-B3 — 4 Stripe oprim

**Date:** 2026-05-31
**Status:** ✅ PASS

---

## §1 — 5 Red Lines

### pytest — 20/20 passed

```
tests/test_stripe_create_payment_intent.py    5 passed
tests/test_stripe_retrieve_payment_intent.py  5 passed
tests/test_stripe_refund_payment.py           5 passed
tests/test_stripe_verify_webhook_signature.py 5 passed
Total: 20 passed in 2.12s (combined with alipay: 45 passed)
```

### Coverage

```
oprim/stripe_create_payment_intent.py    100%
oprim/stripe_retrieve_payment_intent.py  100%
oprim/stripe_refund_payment.py           100%
oprim/stripe_verify_webhook_signature.py 100%
```

### mypy --strict

```
Success: no issues found in 4 source files
```

Notes:
- `stripe_create_payment_intent.py` / `stripe_retrieve_payment_intent.py`: metadata extracted via `hasattr(_m, "to_dict")` guard for `UntypedStripeObject` compatibility
- `stripe_verify_webhook_signature.py`: `# type: ignore[no-untyped-call]` on `stripe.Webhook.construct_event`

### ruff check

```
All checks passed!
```

### git tag v2.21.0

```
v2.21.0  (shared with P9-B2, pushed to github.com/helios-plat/oprim)
```

---

## §2 — Modules

### `oprim/stripe_create_payment_intent.py`

- `StripeConfig(api_key, webhook_secret=None)`
- `StripePaymentIntent(intent_id, client_secret, amount, currency, status, metadata={})`
- `StripeError` / `StripeAPIError`
- `async stripe_create_payment_intent(*, config, amount, currency="usd", metadata=None) → StripePaymentIntent`
- Uses `asyncio.to_thread(stripe.PaymentIntent.create, ...)`; `stripe.error.StripeError` → `StripeAPIError`

### `oprim/stripe_retrieve_payment_intent.py`

- `async stripe_retrieve_payment_intent(*, config, intent_id) → StripePaymentIntent`
- `asyncio.to_thread(stripe.PaymentIntent.retrieve, intent_id, api_key=...)`

### `oprim/stripe_refund_payment.py`

- `async stripe_refund_payment(*, config, intent_id, amount=None, reason=None) → bool`
- `asyncio.to_thread(stripe.Refund.create, payment_intent=intent_id, ...)`
- reason ∈ {"duplicate", "fraudulent", "requested_by_customer"}

### `oprim/stripe_verify_webhook_signature.py`

- `StripeInvalidSignatureError(StripeError)`
- `def stripe_verify_webhook_signature(*, config, payload, signature) → dict` (SYNC)
- Uses `stripe.Webhook.construct_event` (official SDK — not reimplemented)
- `webhook_secret=None` → `ValueError`; `SignatureVerificationError` → `StripeInvalidSignatureError`

---

## §3 — Commits

```
fc5393d chore: uv.lock update for alipay-sdk + stripe deps (oprim v2.21.0)
3fd0699 feat(alipay+stripe): 8 payment oprim — alipay (4) + stripe (4) — v2.21.0
```

Dependency added: `stripe>=11.0`
