# SELF_CHECK — oprim B2: url_safety_check

**Date**: 2026-05-28
**Branch**: feat/c1-url-safety-check
**Target version**: 2.16.0 (SPEC specified 2.13.0; actual current was 2.15.0, next MINOR = 2.16.0)
**Files added/modified**:
- `oprim/url_safety_check.py` — url_safety_check / URLSafetyResult / URLSafetyError
- `oprim/__init__.py` — export added
- `oprim/_manifest.py` — ELEMENTS + CATEGORIES["security"] updated
- `oprim/_version.py` — bumped 2.15.0 → 2.16.0
- `pyproject.toml` — version bumped to 2.16.0
- `CHANGELOG.md` — entry added under [Unreleased]
- `tests/test_url_safety_check.py` — 20 tests

---

## Test count

| File | Tests |
|------|-------|
| test_url_safety_check.py | 20 |
| **Total new** | **20** |

---

## Coverage (new file only)

```
Name                        Stmts   Miss Branch BrPart  Cover   Missing
-----------------------------------------------------------------------
oprim/url_safety_check.py      49      0     18      0   100%
-----------------------------------------------------------------------
TOTAL                          49      0     18      0   100%
Required test coverage of 95.0% reached. Total coverage: 100.00%
20 passed in 1.39s
```

---

## mypy --strict

```
Success: no issues found in 1 source file
```

---

## ruff

```
All checks passed!
```

---

## Implementation notes

- `_CGN_NETWORK = ipaddress.ip_network("100.64.0.0/10")` added at module level — Python 3.11+ changed `is_reserved` semantics and 100.64/10 (CGN, RFC 6598) no longer returns `True` for `is_reserved`. Explicit check preserves SPEC intent.
- Check order: `is_link_local` before `is_private` — `169.254/16` is both in Python 3.12; link_local label is more informative for SSRF diagnostics.
- DNS-rebinding residual risk documented in docstring per SPEC §2.1.

---

## Gate summary

| Gate | Result |
|------|--------|
| Coverage ≥95% | ✓ 100% |
| Tests ≥10 | ✓ 20 |
| mypy --strict 0 errors | ✓ |
| ruff 0 errors | ✓ |
| CHANGELOG entry | ✓ |
| __all__ + _manifest.py updated | ✓ |
| Version bump | ✓ 2.15.0 → 2.16.0 |
