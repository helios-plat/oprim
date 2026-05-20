# Aegis-Ops Batch 1 — Self-Test Report
Date: 2026-05-20
Branch: feat/phase-10-translation (will merge as aegis-batch-1-oprim)

## Summary

| Metric | Result |
|--------|--------|
| Tests total | 42 |
| Tests passed | 42 |
| Tests failed | 0 |
| Statement coverage (6 new modules) | 99.57% |
| Branch coverage (6 new modules) | 32/32 fully covered; 1 partial (postgres line 104→96) |
| mypy strict | 0 errors |
| ruff | 0 errors |

## Modules Delivered

| Module | Stmts | Miss | Cover | Notes |
|--------|-------|------|-------|-------|
| oprim/_exceptions.py | 4 | 0 | 100% | |
| oprim/docker_logs.py | 33 | 0 | 100% | |
| oprim/loki_query.py | 44 | 0 | 100% | |
| oprim/postgres_pool_status.py | 45 | 0 | 98% | 1 branch partial (104→96, unreachable in test) |
| oprim/prometheus_instant_query.py | 39 | 0 | 100% | |
| oprim/rabbitmq_queue_status.py | 36 | 0 | 100% | |

## Test Distribution

| File | Tests |
|------|-------|
| test_rabbitmq_queue_status.py | 9 |
| test_docker_logs.py | 7 |
| test_postgres_pool_status.py | 6 |
| test_prometheus_instant_query.py | 10 |
| test_loki_query.py | 10 |

## Infrastructure Notes

### numpy 2.4.4 + pytest-cov incompatibility — RESOLVED
Root cause: `oprim/__init__.py` imports `oprim.behavioral` → `scipy` → `numpy.fft._pocketfft_umath`
(a numpy C extension). Coverage's `sys.settrace` hook is attached before conftest.py loads.
When the C extension is first loaded under an active trace, numpy 2.4.4 raises:
`ImportError: cannot load module more than once per process`.

Fix: `sitecustomize.py` in venv site-packages pre-loads numpy + numpy.fft + scipy + scipy.stats +
scipy.sparse before Python starts pytest-cov. Python executes sitecustomize.py at interpreter
startup, before any import hooks are modified.

File: `.venv/lib/python3.12/site-packages/sitecustomize.py`

### respx (not responses) for httpx mocking
`oprim.rabbitmq_queue_status`, `oprim.prometheus_instant_query`, and `oprim.loki_query` all use
`httpx.Client`. The `responses` library only intercepts `urllib3`/`requests` — it does not hook
httpx transports. `respx` (v0.23.1) is the correct library for httpx mocking.

### Exception pattern
All except blocks use `raise X(...) from exc` (ruff B904). TimeoutError is used as the builtin
(not asyncio.TimeoutError — ruff UP041). postgres_pool_status uses asyncio.run() wrapping an
async inner `_query()` function.

## Raw Commands Run

```
.venv/bin/python -m mypy oprim/_exceptions.py oprim/rabbitmq_queue_status.py oprim/docker_logs.py \
  oprim/postgres_pool_status.py oprim/prometheus_instant_query.py oprim/loki_query.py --config-file mypy.ini
# → Success: no issues found in 6 source files

.venv/bin/python -m ruff check oprim/_exceptions.py oprim/rabbitmq_queue_status.py oprim/docker_logs.py \
  oprim/postgres_pool_status.py oprim/prometheus_instant_query.py oprim/loki_query.py \
  tests/test_rabbitmq_queue_status.py tests/test_docker_logs.py \
  tests/test_postgres_pool_status.py tests/test_prometheus_instant_query.py \
  tests/test_loki_query.py tests/conftest.py
# → All checks passed!

.venv/bin/python -m pytest tests/test_rabbitmq_queue_status.py tests/test_docker_logs.py \
  tests/test_postgres_pool_status.py tests/test_prometheus_instant_query.py \
  tests/test_loki_query.py \
  --cov=oprim.rabbitmq_queue_status --cov=oprim.docker_logs \
  --cov=oprim.postgres_pool_status --cov=oprim.prometheus_instant_query \
  --cov=oprim.loki_query --cov=oprim._exceptions \
  --cov-report=term-missing -v
# → 42 passed, total coverage 99.57%
```
