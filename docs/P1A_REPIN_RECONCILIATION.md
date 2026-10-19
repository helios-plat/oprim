# P1-A — HEVI re-pin reconciliation

**Status**: complete. Zero push, zero force. Produces a canonical release commit
that is safe to fast-forward.

```
5d992f2  origin tip (= HEVI's current pin)
   │
   └── 7eab94e  canonical release commit   ← ready to push, NOT pushed
```

---

## 1. The conflict, resolved

`5d992f2` ("feat(manifest): restore root __manifest__ for HEVI consumer
contract", hevi-qualification, 2026-09-24) added 48 lines to
`oprim/__init__.py` and touched nothing else.

The P0 media line independently restored the same contract as a *derived* view.
Both sides wrote `__manifest__` at the end of the file, so the merge had one
real conflict:

```
oprim/__init__.py  UU  (only conflicted file)
```

Resolved in favour of the derived implementation, because origin's literal is
the same class of defect D5 removed elsewhere in this branch — a second,
hand-maintained source of truth that had already drifted from reality:

| origin's entry | problem |
|---|---|
| `video_generate` signature `'(request, /, *, provider, output_path) -> Path'` | `video_generate` has no `request` parameter and never has |
| `edge_tts_synthesize` signature `'(request, /, *, output_path) -> Path'` | same |
| `avatar_generate` signature `'(request, /, *, provider, output_path) -> Path'` | same |
| `hevi_types` | a module namespace, not a definition → `dangling` under the audit script's own `entity_exists` rule |
| `pillars: ['cost','fingerprint','trail','report']` on every atom | omodul concepts asserted on Layer 1 atoms, where they mean nothing |

Supersession is deliberate and total for everything resolvable. The derived
manifest is a **strict superset** of origin's four resolvable element names
(`video_generate`, `edge_tts_synthesize`, `avatar_generate`,
`style_marker_prompt`) at 25 entries, each asserted to resolve to a real
definition. Origin's fifth entry, `hevi_types`, is replaced by the five
concrete types HEVI actually imports from that namespace: `CanvasEdge`,
`CanvasNode`, `ProviderCapability`, `Subject`, `VideoQuality`.

Two further differences recorded rather than hidden:

- `depends_on` now lists intra-package modules derived from each element's real
  imports; origin listed external `obase.*` paths. Nothing consumes this field
  for oprim — HEVI asserts `len(depends_on) >= 2` only for **oskill** — and the
  intra-package form is what the dependency-direction rule can be checked
  against.
- `signature` is computed on demand by `element_signature()` rather than
  stored, because introspecting every element at import time would import the
  whole package. The audit script reads it as `entry.get("signature")`, so
  absence is already tolerated.

## 2. Merge, not rebase — and why it matters

HEVI pins oprim by exact SHA. A rebase would have orphaned `5d992f2`, breaking
HEVI's existing `uv.lock` and venv until the re-pin landed. Merging keeps it
reachable:

```
git merge-base --is-ancestor 5d992f2 HEAD  →  YES
```

`5d992f2` remains a valid pin until HEVI is re-pinned, and remains valid
afterwards for anyone still resolving it.

## 3. Push shape — fast-forward, no force

```
origin tip : 5d992f2
local HEAD : 7eab94e
ahead/behind: 0 / 13        (0 behind = nothing on origin is missing locally)

git push --dry-run origin HEAD:refs/heads/feat/v3.23.0-commerce-atoms
  →  5d992f2..7eab94e  HEAD -> feat/v3.23.0-commerce-atoms
```

Ordinary fast-forward. Git accepts it with no `--force`, no `--force-with-lease`.

## 4. Change surface — 13 commits, 1 file of conflict

```
7eab94e merge(manifest): reconcile origin's literal __manifest__ with the derived one
2731a8e chore: stop the veya runtime from dirtying the tree
39bd8b9 refactor: declare the 7 provable shared bases as infra
a33d2e9 fix: restore the release manifest HEVI's gate requires
2c1c887 chore: clear the pre-existing ruff findings
84e8698 docs: P0 media inventory, collision map and DAG audit
9353a1d test: pin the media DAG contract and the debt closure
4798238 fix: close the media quality debt
069be88 refactor: declare the layer contract, close the lint blind spot
1bc504a feat: canonical media primitives
86fa205 / 82033ed / 2852d2e  feat: add Veya runtime 3O element  (pre-existing local line)
```

Net content delta vs origin: **86 files, +4909 / −970**, confined to
`oprim/`, `tests/`, `docs/`, and one line of `pyproject.toml`.

## 5. Public surface — audited, not assumed

`__all__`: **1461 → 1448**. Thirteen names removed. Every one was audited for
consumers across all five 3O layers plus HEVI:

| name | HEVI consumers | 3O consumers | import style |
|---|---|---|---|
| `SdClient` `SearxngClient` `TtsClient` `WebSearchResult` `WhisperClient` `WhisperSegment` | **0** | oskill `knowledge/*` (3 files) + oprim tests | module path only |
| `probe_json` `probe_text` `require_ffprobe` `parse_rational` `coerce_float` `coerce_int` | **0** | oprim only | `from oprim._ffprobe import …` |
| `encode_frames_to_mp4` | **0** | oprim only | `from oprim._encode_frames import …` |

**Not one top-level import exists for any of the 13.** Every reference is
`from oprim._<module> import …` or `from oprim.external.clients.<x> import …`,
both of which bypass `__all__`. So the removals cannot break an importer.

They were all introduced by this branch's own unreleased work, or are empty
placeholder classes: the six infra helpers came from P0-C's `_ffprobe` /
`_encode_frames`, and the six clients are `class X: pass` shells that oskill
imports by path (deleting the files would break oskill at import time, which
is why only the export was dropped).

## 6. Version and packaging

| item | origin | local | action |
|---|---|---|---|
| `pyproject.toml` version | 3.23.0 | 3.23.0 | unchanged |
| `oprim/_version.py` | 3.23.0 | 3.23.0 | unchanged |
| `pyproject.toml` delta | — | one line | `oprim/_manifest.py` removed from `coverage.run.omit` (the file was deleted) |
| hatch version source | `oprim/_version.py` | same | consistent |

No version bump is required for resolution: `oskill` requires `oprim>=3.0.0`
and `omodul` requires `oprim>=2.11`, both satisfied by 3.23.0. A bump to
3.24.0 is defensible for a release of this size and would also satisfy both
ranges, but that is a release decision, not a technical requirement — see
Open Decisions.

## 7. Downstream consumers — one finding

HEVI pins oprim in **two** places, and both must change together:

- `pyproject.toml:21` — `[project].dependencies`
- `pyproject.toml:187` — `[tool.uv].override-dependencies`

Missing the second would let uv resolve two different oprim sources in one
environment. That override exists because, per its own comment, "oskill/omodul/
oservi hard-pin oprim as a git URL in their metadata". **That comment is
stale**: the local checkouts and the installed dist-info metadata all use
version ranges (`oprim>=3.0.0`, `oprim>=2.11`, bare `oprim`) — no git URLs.
The override is harmless belt-and-braces; the comment should be corrected
when the pin is updated.

Consequence for non-HEVI consumers of oskill/omodul: they resolve oprim by
range, so they may pick up 3.23.0 from PyPI or elsewhere rather than this
commit. That is expected for a SHA-pinned release and is out of scope here.

## 8. Verification of the merged tree

| gate | result |
|---|---|
| `ruff check oprim/ tests/` | **All checks passed** |
| `3o_lint/runner.py` (all 5 layers) | **ALL GREEN** |
| media DAG element→element edges | **0** |
| non-media backlog | 142 (pinned by test, unchanged) |
| oprim full suite | `273 failed / 22 errors`, **test-ID set identical to the pre-P0 baseline**; passed 4972 → 5198 |
| `oprim/__init__.py` vs pre-merge | byte-identical |
| `__all__` vs pre-merge | unchanged at 1448 |
| HEVI `test_three_o_contracts` + `test_oprim_prims` + `test_media_workflows_3o` | **15 passed** |
| HEVI full suite | **6599 passed / 7 failed** |
| the 7 HEVI failures | identical set under pinned and merged-local oprim: 2 pre-existing in `test_history_arc_adapter`, 5 `asyncpg.TooManyConnections` in `test_library.py` that **pass in isolation under both** |
| `git status` | clean |
| tracked `.veya/` files | 0 |
| tracked `out.tar.gz` | 0 |

HEVI backward compatibility is therefore demonstrated, not assumed: same
failure set before and after, with the release-contract test going from
failing to green.

## 9. Executable migration plan

Nothing below has been executed. Steps 1–2 are the owner's call.

```bash
# ── owner gate ────────────────────────────────────────────────────────────
# 1. PUSH the canonical release commit (fast-forward, no force)
cd /data/soffy/projects/platform/3O/oprim
git push origin HEAD:refs/heads/feat/v3.23.0-commerce-atoms
#    expected: 5d992f2..7eab94e

# 2. RE-PIN HEVI — both occurrences, together
cd /data/soffy/projects/hevi
NEW=7eab94e7e773622996f757cabc0486eaa4dd8e5a
OLD=5d992f2cbb8b838e8a04d743caf778b8a1be894a
sed -i "s|oprim.git@$OLD|oprim.git@$NEW|g" pyproject.toml
grep -n "helios-plat/oprim.git" pyproject.toml      # expect 2 lines, both @ $NEW
#    also correct the now-stale override comment at pyproject.toml:180-182

# 3. RELOCK
uv lock                       # obase/oskill/omodul/oservi pins unchanged

# 4. CONSUMER CONTRACT
.venv/bin/python -m pytest tests/test_three_o_contracts.py tests/test_oprim_prims.py -q
#    expect: all pass — this is the gate that was broken before

# 5. FULL SUITE
.venv/bin/python -m pytest -q
#    expect: 7 failures, identical to the pre-re-pin set
#            (2 test_history_arc_adapter, 5 asyncpg pool exhaustion)

# 6. PRODUCTION GATE — see Open Decisions
```

Rollback is one command and needs no force, because the merge commit keeps
`5d992f2` reachable:

```bash
git -C /data/soffy/projects/platform/3O/oprim push origin 5d992f2:refs/heads/feat/v3.23.0-commerce-atoms
cd /data/soffy/projects/hevi && git checkout pyproject.toml uv.lock && uv sync
```

Backup refs in the local repo, should the push need to be unwound locally:

```
backup/p0-checkpoint-history      7552d42   pre-rebuild msg history
backup/p0-checkpoint-history-2    4b13753   pre-merge
backup/pre-merge                  2731a8e   pre-merge, same content as HEAD^1
```

## 10. Open decisions

1. **Is the push authorised?** `helios-plat/oprim` is a shared remote with
   other consumers. Steps 1–2 are withheld pending explicit approval.
2. **Version bump or not?** 3.24.0 is justified by the surface added and is
   satisfied by every downstream range; 3.23.0 also works. If bumping, both
   `pyproject.toml` and `oprim/_version.py` must move together (hatch reads
   the latter).
3. **Stale override comment** at `pyproject.toml:180-182` — correct it as part
   of the re-pin, or separately.
4. **P1-B / P1-C ordering.** The 142 non-media edges and the 7 missing source
   repositories are independent of this re-pin and can proceed in parallel.
   Nothing in this plan depends on them, and nothing in them depends on the
   push.
