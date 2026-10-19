# P0 Media Canonical Inventory & Collision Map

Scope: `platform/3O` → `oprim` (Layer 1). Reference: `P0 Media 3O Canonicalization SPEC v1.0`.
Status: P0-A complete · P0-B complete · P0-C complete · P0-D/P0-E deferred (blocked).

---

## 1. Scope resolution

The SPEC names scope `platform/3O`. That is `/data/soffy/projects/platform/3O`, five
independent git repos consumed by HEVI through exact-SHA pins
(`hevi/pyproject.toml:20-26`):

| repo | branch at time of work | HEVI pin |
|---|---|---|
| `obase` | `feat/v0.20.0-commerce-batch-schema` | `210fb151` |
| `oprim` | `feat/v3.23.0-commerce-atoms` | `5d992f2c` |
| `oskill` | `feat/v4.12.0-resolve-display-batch` | `e265469a` |
| `omodul` | `feat/v1.39.0-create-inventory-batch` | `c37519a4` |
| `oservi` | `main` | `d8dcea6b` |

All five working trees were already dirty on arrival. `oprim` local HEAD (`86fa205`)
has **diverged** from the HEVI pin (`git merge-base --is-ancestor 5d992f2c HEAD` → false),
and two other 3O checkouts exist with different pins
(`veya/platform/3O/CANONICAL_PINS.json`, `platform/helios-platform/3O`).
Nothing in this round re-pins HEVI; that is a separate consumer-contract gate.

## 2. Source-project availability (blocking finding)

The SPEC's premise is extraction from 8 projects. Exhaustive search of
`/data`, `/home/soffy`, `/mnt/d`, `/opt`, `/srv` (including enumerating every
`.git` remote on the host) found **1 of 8**:

| project | status |
|---|---|
| `video-use` | present only as a vendored skill inside the `hevi/docs/ref-repos/rnskill` submodule (`Pluviobyte/rnskill` @ `fed3eef`) — not an independent clone |
| `LuxTTS` | adapter only: `hevi/services/voicebox/backend/backends/luxtts_backend.py` |
| `LongLive` | provider boundary only; `LONGLIVE_BASE_URL` unset, Stage 1 BLOCKED |
| `creatorhub` | already internalized into HEVI (`hevi/layer4_legacy/platforms/oskill_impl/`) |
| `image-blaster`, `stickman-video-director`, `qisi-video-remix`, `aid-studio` | zero hits anywhere on the host |

Consequence: SPEC §5.1 (`来源: qisi-video-remix`), §5.2, §5.4, §5.5, §5.6 carry
the highest design judgment in the SPEC and have no source to extract from.
**P0-D (OSkill) and P0-E (OModul) are deferred** rather than invented.

## 3. Collision map

The SPEC §8 forbids duplicate canonical implementations but did not enumerate
collisions with already-public oprim elements. Four found, all resolved per the
ruling "keep existing names canonical; SPEC names become aliases":

| SPEC §4 name | pre-existing public element | resolution |
|---|---|---|
| `probe_media` | `media_probe` (`_media_probe.py`) | `media_probe` stays canonical; `probe_media` = same-module alias. Canonical element **enriched** with `fps` / `bitrate` / `video_codec` / `audio_codec` / `audio_sample_rate` / `audio_channels`, plus per-stream `fps`/`bitrate`/`sample_rate`/`channels`/`channel_layout`/`duration_seconds`. All new fields default `None` → additive. |
| `transcribe_media` | `transcribe_audio` (`_transcribe_audio.py`) | `transcribe_audio` canonical; `transcribe_media` alias. |
| `mix_audio_tracks` | `audio_mix` (`_audio_mix.py`) | `audio_mix` canonical, **enriched** with `offsets` / `fade_in_s` / `fade_out_s` / `duration` policy; `mix_audio_tracks` alias. |
| `burn_subtitles` | `subtitle_burn` (`_subtitle_burn.py`) | `subtitle_burn` canonical; `burn_subtitles` alias. |
| `render_media` | `render_html_to_mp4` (`_render_html_to_mp4.py`) | not a collision — `render_media` is the renderer-agnostic path, `render_html_to_mp4` stays the HTML specialization. Both encode through one shared `_encode_frames`. |

`probe_duration` → compatibility wrapper over the single `_ffprobe` executor
(SPEC §4.1/§8.1). `ProbeDurationError` and its two message contracts are preserved.

## 4. Duplicate-implementation audit (before → after)

Five independent ffprobe invocations existed. Now one:

| before | after |
|---|---|
| `_probe_duration.py` own `subprocess.run` | delegates to `_ffprobe.probe_text` |
| `_media_probe.py` own `_run_ffprobe` | delegates to `_ffprobe.probe_json` |
| `_video_concat.py:_probe_durations` | untouched (private, concat-internal) |
| `_video_recompose.py:_probe_dimensions` | untouched (private, recompose-internal) |
| `_video_quality_metrics.py` own subprocess | untouched (separate concern) |

`obase.ffmpeg` public API is `run` / `FFmpegError` / `FFmpegNotFoundError` only.
`run()` is async, executes `ffmpeg` (not `ffprobe`), and **discards stdout**
(`_run.py:73`). Two consequences, both recorded rather than worked around:

- `_ffprobe.py` is a private infra module, not a second public wrapper — SPEC §8.4
  forbids `video_ffmpeg` / `media_ffmpeg` / `hevi_ffmpeg`, and a test now asserts
  no such module exists.
- `_extract_audio_waveform.py` spawns `ffmpeg` directly because it must read
  decoded PCM off stdout, which `run()` throws away. Declared explicitly in the
  single-source test rather than left as a silent exception.

## 5. Capability inventory (canonical set)

### 5.1 OPrim — canonical media atoms

| capability | canonical element | module | status |
|---|---|---|---|
| media inspection | `media_probe` (+ `probe_media` alias) | `_media_probe.py` | enriched |
| duration scalar | `probe_duration` | `_probe_duration.py` | compat wrapper |
| transcription | `transcribe_audio` (+ `transcribe_media`) | `_transcribe_audio.py` | alias added |
| frame extraction | `extract_video_frames` | `_extract_video_frames.py` | **new** |
| audio waveform | `extract_audio_waveform` | `_extract_audio_waveform.py` | **new** |
| physical segmentation | `segment_media` | `_segment_media.py` | **new** |
| segment extraction | `extract_media_segment` | `_extract_media_segment.py` | **new** |
| general render | `render_media` | `_render_media.py` | **new** |
| audio mixing | `audio_mix` (+ `mix_audio_tracks`) | `_audio_mix.py` | enriched |
| audio normalize | `audio_normalize` | `_audio_normalize.py` | unchanged |
| audio/video merge | `audio_video_merge` | `_audio_video_merge.py` | unchanged |
| subtitle burn-in | `subtitle_burn` (+ `burn_subtitles`) | `_subtitle_burn.py` | alias added |
| TTS | `tts_synthesize` | `tts_synthesize.py` | unchanged |
| video generation | `video_generate` | `_video_generate.py` | orphan closed |
| voice reference | `encode_voice_reference` | `_encode_voice_reference.py` | **new** |

Private infra: `_ffprobe.py` (sole ffprobe executor), `_encode_frames.py` (sole
frame-sequence encoder).

### 5.2 Notable defect found and fixed

`ltx2_cloud_generate` was in `oprim.__all__` but unreachable from `video_generate` —
a vendor capability that existed as a public export with no canonical path to it.
`video_generate(provider="ltx2_cloud")` now dispatches with signature adaptation
(`mode` t2v/i2v, `resolution` tuple).

### 5.3 Not changed, flagged

- `audio_lufs` on `VideoQualityMetrics` is declared and **never populated** — dead field.
- `render_html_to_mp4` calls `validate_html`; `video_generate` calls 4 vendor
  generators + 2 provider modules. 7 sibling edges, see §6.
- `video_generate` drops `fps` / `bitrate_kbps` at the fal dispatch stage
  (`_video_generate.py:93-110`) while the wan stage receives them. Pre-existing
  inconsistency; not touched, since changing it alters fal payload behaviour.
- `_manifest.py` is orphaned: 454 hardcoded names, zero media elements,
  `VERSION = "3.0.0"` vs actual `3.23.0`, not imported anywhere.
- `oprim/external/clients/{whisper,tts,sd,searxng}_client.py` are `class X: pass`
  stubs yet exported in `__all__`.
- `_vibevoice_synthesize.py:141` holds a mutable `_spk_map` closure — cross-call state
  in an element that must be stateless.
- `transcribe_audio` reloads the Whisper model on every call (`_transcribe_audio.py:96`).

## 6. Acceptance gate status (SPEC §12)

| gate | status | evidence |
|---|---|---|
| A. no duplicate canonical implementation | **PASS for this round** | `test_exactly_one_defining_module` (19 parametrized cases); mutation-verified |
| B. vendor names absent from capability naming | **PASS** | `test_no_vendor_names_in_new_canonical_set` |
| C. no OPrim sibling bare calls | **NOT MET (pre-existing)** | see below |
| D. no OSkill persistence | N/A | P0-D deferred |
| E. OModul transaction semantics | N/A | P0-E deferred |
| F. every new element independently testable | **PASS** | 125 tests, 8 against real ffmpeg |
| G. existing media API backward compatible | **PASS** | all 4 pre-existing media test files keep their exact pass/fail counts; new fields additive |
| H. HEVI unmodified | **PASS** | no file under `hevi/` touched |
| I. 3O lint PASS | **PASS** | `3o_lint/runner.py` ALL GREEN, now covering 16 media modules instead of 0 |
| J. full regression PASS | **ZERO REGRESSION** | 295 baseline FAILED/ERROR IDs before and after, identical set |

### Gate C — the one real gap

`check_no_sibling_call.py:63-64` skips `_`-prefixed files, and oprim puts every
implementation in `_foo.py`. The rule was therefore **vacuously green**: the
7-element video-gen DAG was invisible to it.

Adding the media set to `3o_lint`'s allowlist made the check real and immediately
surfaced 9 violations — **all pre-existing, none from new code**:

```
_render_html_to_mp4.py → oprim._validate_html
_video_generate.py     → oprim._providers.wan_cloud, _fal_queue_generate,
                         _hailuo_generate, _kling_v2_generate,
                         _veo3_generate, _ltx2_cloud_generate
```

Closing this is the §12.C refactor (move provider dispatch behind an injected
resolver; move `validate_html` to infra) and is **not** in the SPEC §11
implementation order. This round deliberately keeps those two modules out of the
allowlist so gate I stays meaningful and green, and pins the exact debt in
`test_known_sibling_call_debt_is_pinned` so it cannot grow silently. That test
and `test_new_media_atoms_do_not_call_sibling_atoms` are both mutation-verified.

## 7. Test evidence

`tests/test_media_canonical_prims.py` — 125 tests, all passing.

Per SPEC §9, each new OPrim covers happy path, invalid input, provider failure,
missing artifact, timeout, and output contract. `TestRealFfmpeg` (8 tests)
executes real ffmpeg/ffprobe against a generated `testsrc`+`sine` MP4 — these run
(not skipped) on this host.

Invariant tests are mutation-verified: adding a sibling import to a new atom,
growing the pre-existing debt set, and duplicating `media_probe` into a second
module each produce a failure.

Regression: `4972 passed → 5097 passed` (+125). Failures unchanged at
`273 failed / 22 errors`, identical test-ID set to the pre-work baseline.

Ruff: 2 errors, both pre-existing (`_controller_generation.py`, `_execution_lineage.py`);
the 5 further errors reported repo-wide come from untracked `.veya/worktrees/` copies.

## 8. Blocked / not done

- **P0-D (OSkill) and P0-E (OModul)** — blocked on the 7 absent source projects.
  Specifically `extract_reference_evidence`, `extract_film_grammar`,
  `segment_video_semantically`, `build_shot_continuity`,
  `generate_standalone_prompt`, `validate_generation_prompt`, `visual_cut_qc`,
  `audio_cut_qc`, `artifact_qualification` and all four OModuls.
- **§12.C refactor** — 9 pinned sibling edges (§6).
- **HEVI re-pin** — HEVI consumes oprim by exact SHA; this round changed the
  working tree only. Re-pin + `uv lock` + the 29/29 consumer contract gate is a
  separate step, and oprim local HEAD has already diverged from the pinned line.
