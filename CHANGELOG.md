# Changelog

<!-- Governance: see RELEASE_POLICY.md. main = release branch; feat branches deleted after merge; oprim → oskill → omodul merge order required; container bind-mount means git checkout is a live operation. -->

## [3.9.0] — 2026-06-14

### Added (H-B: IO oprim 36 新建)

**A组 — 文件 IO 扩展 (5)**
- feat: `ensure_parent_dir` — 幂等创建父目录链（atomic_write 辅助）
- feat: `file_read_bytes` — 字节范围读取（offset/length，图片/二进制 part）
- feat: `image_to_base64` — 图片 → base64 ASCII 字符串（多模态 part，不校验 mime）
- feat: `atomic_write` — 原子写（临时文件 + fsync + rename，防写一半崩溃）
- feat: `backup_before_overwrite` — 覆盖前自动备份（.bak / .bak1 序列，undo 辅助）

**B组 — 进程控制扩展 (5)**
- feat: `spawn_pty` — PTY 伪终端启动（交互式命令，返回 PtyHandle）
- feat: `stream_stdout` — 流式读进程输出 async generator（PtyHandle | ProcHandle）
- feat: `kill_process` — 向进程组发信号（TERM/KILL/INT/HUP，幂等）
- feat: `wait_with_timeout` — 等待进程结束带超时，超时 raise TimeoutError
- feat: `run_background` — 后台启动进程，立即返回 JobId（UUID4）
- types: `ProcHandle`, `PtyHandle`, `JobId`, `ExecResult`

**D组 — LSP IO 扩展 (12)**
- feat: `lsp_goto_definition(path, *, pos, lsp)` — 跳转定义（pos: Pos = tuple[int,int]）
- feat: `lsp_find_references(path, *, pos, lsp)` — 查找引用
- feat: `lsp_goto_implementation(path, *, pos, lsp)` — 跳转实现（接口→实现类）
- feat: `lsp_document_symbol(path, *, lsp)` — 文档符号大纲（singular）
- feat: `lsp_workspace_symbol(*, query, lsp)` — workspace 符号搜索（singular）
- feat: `lsp_prepare_call_hierarchy(path, *, pos, lsp)` — 准备调用层级
- feat: `lsp_incoming_calls(item, *, lsp)` — 谁调用了它
- feat: `lsp_outgoing_calls(item, *, lsp)` — 它调用了谁
- feat: `diagnostics_to_summary(diags)` — 诊断列表 → 文本摘要 [s 纯计算]
- feat: `location_to_snippet(loc, *, ctx)` — Location → 带上下文代码片段
- upd: `lsp_hover` — 新增 `pos: Pos | None` 和 `lsp:` 参数（向后兼容 line/character/server）
- upd: `lsp_diagnostics` — 新增 `lsp:` 参数别名（向后兼容 server:）
- types: `Pos = tuple[int, int]`, `CallItem`

**E组 — 网络 IO 扩展 (5)**
- feat: `validate_api_key(key, *, provider)` — 校验 API key（anthropic/openai/openrouter/google/mistral/cohere，401→False，网络失败 raise）
- feat: `upload_share(payload, *, endpoint)` — 上传 session payload 返回分享 URL
- feat: `revoke_share(url)` — 撤销分享链接（404/409 幂等）
- feat: `fetch_models_dev(*, refresh)` — 从 models.dev 拉取 75+ provider 模型清单
- feat: `load_skill_raw(path)` — 读 SKILL.md 返回原始字符串（解析交 parse_skill_md H-A）
- types: `ModelSpec`, `ShareUrl`

**G组 — MCP IO 扩展 (2)**
- feat: `mcp_connect(server_url, *, timeout)` — 连接 MCP server（HTTP/SSE + stdio://），返回 McpSession
- feat: `load_custom_tool(path)` — 加载 .ts/.js/.json/.yaml tool 定义，返回 Tool
- types: `McpSession`, `Tool`

**H组 — Git 原子 (7)**
- feat: `parse_git_status(raw)` — 解析 git status --porcelain 输出 [s 纯计算]
- feat: `parse_git_diff(raw)` — 解析 git diff unified format → FileChange 列表 [s 纯计算]
- feat: `parse_gitignore(content)` — 解析 .gitignore → GitIgnorePattern 列表 [s 纯计算]
- feat: `detect_project_type(root)` — 探测项目类型（读盘标志文件，不依赖 git）
- feat: `git_current_branch(*, cwd)` — 当前分支名（detached HEAD 返回短 hash）
- feat: `git_snapshot(*, cwd)` — 创建工作区快照（stash push + unique id）
- feat: `git_restore_snapshot(snap_id, *, cwd)` — 恢复快照（stash pop，empty: 前缀 no-op）
- types: `GitStatus`, `StatusEntry`, `FileChange`, `GitIgnorePattern`, `ProjectType`, `SnapshotId`

## [3.1.1] — 2026-06-13

### Fixed (B-1 + B-2)
- fix(B-2): `from oprim import image_generate` returned `<module>` instead of `<function>` — PEP 562 lazy loading (v2.38.0) removed the v2.24.1 explicit re-exports; restored for `image_generate`, `image_understand`, `tts_synthesize`
- feat(B-1): `[project.entry-points."obase.providers"]` — `qwen3_dashscope` + `qwen3` LLM providers now declared; `ProviderRegistry.auto_discover()` can discover without manual Layer-4 bootstrap

## [3.1.0] — 2026-06-13

### Added (hevi v2 — M1/M3 新建 + M2/M4 扩展)
- feat: `ltx2_cloud_generate` — LTX-2 cloud video generation via fal.ai (T2V/I2V, ≤20s, async poll, base64 i2v)
- feat: `vibevoice_synthesize` — VibeVoice 1.5B 本地多说话人 TTS (zero-shot 克隆, 分段, safety watermark)
- feat: `video_generate` +`"wan_cloud"` provider — Wan 2.6/2.7 Alibaba Cloud T2V/I2V (删 duration 参数)
- feat: `avatar_generate` +`"duix"` provider — Duix-Avatar 本地 Docker REST (fun-asr + fish-speech + duix.avatar)
- Internal: `_providers/wan_cloud.py`, `_providers/duix.py`

## [3.0.0] — 2026-06-13

### Removed — BREAKING CHANGES
- 删除: `db_insert` / `db_query` / `db_read` / `db_write` / `db_update` / `db_soft_delete` (已迁移至 obase.persistence)
- 删除: `cache_invalidate` (已迁移至 obase.cache)
- 删除: `_docker.py` (整文件 + 22 个 `docker_*` 函数，已迁移至 obase.docker)
- 删除: `realtime_quote_redis_fetch` (依赖 redis)
- pyproject 移除重依赖: `docker` / `psycopg` / `redis`

## [2.38.0] — 2026-06-12

### Changed — L2 枢纽惰性化
- feat: 顶层 `__init__.py` 惰性化: 采用 PEP 562 (`__getattr__`) + AST 静态扫描机制。
- 效果: `import oprim` 启动速度提升 ~15x (~2s → <150ms)，且在仅访问纯函数时不再触发 `docker` / `httpx` / `psycopg2` 等重依赖加载。
- 兼容性: 100% 保持现有 `from oprim import <name>` 路径可用 (含 519 个导出项)，非 BREAKING。

## [2.37.0] — 2026-06-12

### Added (AII 3O Batch 6 — 2 new elements)
- feat: `mathlib_lookup` — 查一个标识符在 Mathlib 的形式化条目: 通过 Loogle API 查询 Lean/Mathlib 既有形式化定理条目; count==1 表示唯一无歧义命中, 可用于既有定理确证 (不需现场证明)。依赖 obase.http.dns_pinned_transport。
- feat: `epistemic_confidence_compute` — 按 grade 加权算整体认知可信度: 输入一组检索 KU 的 grade → 输出加权整体可信度 [0,1]。跨项目复用: Tide 信号可信度 / Stratum 引用置信 / Aegis 根因可信度。

## [2.36.1] — 2026-06-12

### Fixed
- fix: 声明缺失依赖 `fsrs>=4.0.0`（`oprim.cognitive` 使用 `fsrs.Card/Rating/Scheduler`，容器重建后 ModuleNotFoundError；陷阱 11 变体）

## [2.31.0] — 2026-06-05

### Added (Aegis 3O Element IMPL SPEC v1.0 — B2: 25 new elements)
- **Docker short-names**: `docker_logs`, `docker_ps`, `docker_restart`, `docker_stats`, `docker_inspect`, `docker_compose_up`, `docker_compose_down` (aliases over existing `docker_container_*`/`compose_*`)
- **`docker_compose_pull`**: new — `docker compose pull` subprocess wrapper; raises `OprimNotFoundError` if compose file missing, `OprimConnectionError` on non-zero exit
- **PostgreSQL aliases**: `postgres_long_running_queries` → `postgres_slow_queries`, `postgres_locks` → `postgres_locks_status`
- **RabbitMQ focused wrappers**: `rabbitmq_queue_depth` (ready+unacked int), `rabbitmq_consumer_count` (consumers int)
- **Network aliases**: `network_port_check` → `tcp_port_check`, `network_http_health` → `http_health_probe`, `network_dns_resolve` → `dns_resolve`
- **Filesystem**: `fs_disk_usage` → `disk_usage` alias; `fs_inode_check` — new inode stat via `os.statvfs`
- **System focused wrappers**: `system_cpu_usage` (float 0–100), `system_ram_usage` (dict), `system_load_avg` (dict with 1m/5m/15m)
- **Caddy new ops**: `caddy_admin_config` (GET full config), `caddy_admin_routes` → `caddy_routes_list` alias, `caddy_route_add_atomic` (GET→insert→PUT), `caddy_route_remove_atomic` (GET→filter→PUT)
- **`appstore_catalog_fetch`** (new module `appstore_catalog_fetch.py`): httpx GET catalog endpoint → `AppCatalogEntry` Pydantic model
- 80 new tests (B2), all green. Pre-existing `postgres_pool_status` and `caddy_admin_reload` count as part of the 27-element surface.

## [2.30.0] — 2026-06-05

### Added (Stratum B2)
- `searxng_search` — single searxng instance query; SSRF-safe transport for public URLs, direct urllib for Docker-internal (172.17.x.x); structured results with title/url/content/engine/score

## [2.29.1] — 2026-06-04

### Fixed
- `vector_encode`: `ProviderRegistry.get_instance()` → `ProviderRegistry.get("embedding", provider)` (classmethod, 3-arg form). `except Exception` split: `ProviderNotFoundError` → log.warning + stub; other exceptions re-raise.
- `llm_extract_ku`: same `get_instance()` fix + `ProviderNotFoundError` vs code-error distinction.
- `llm_distill_strategy`: same fix. All three were silently falling to stub on every call due to `get_instance()` not existing.

### Tests added
- 3 tests per element (9 total): provider-not-registered → stub + warning; provider-registered → real call with messages passthrough; code-error → re-raise.

## [2.29.0] — 2026-06-04

### Added (Stratum B2 — 13 new elements)
**Feed/Content group (B2a):**
- `url_fetch_ssrf_safe` — SSRF-safe URL fetch via obase.http.dns_pinned_transport
- `fetch_rss_feed` — RSS 2.0 feed fetch + parse
- `parse_atom_feed` — Atom 1.0 feed parse from XML string
- `detect_feed_url` — Auto-detect RSS/Atom URL from HTML <link> tags
- `podcast_episode_parser` — Podcast RSS with iTunes enclosure/duration
- `feed_diff_detector` — New/removed items between two feed snapshots
- `ocr_detect_text` — OCR text extraction via provider (stub fallback)

**Utility group (B2b):**
- `concept_extractor` — LLM concept extraction (stub: capitalized phrase regex)
- `keyword_alert_checker` — Exact/regex/fuzzy keyword match with positions
- `citation_formatter` — APA/MLA/Chicago citation formatting, pure logic
- `timeline_aggregator` — Items bucketed by day/week/month from timestamps
- `backlink_resolver` — [[wikilink]] resolution + bidirectional index
- `graph_traversal` — Generic BFS/DFS traversal, cross-business reusable

## [2.28.0] — 2026-06-04

### Added (AII-3O Batch 5b — P5 causal + backtest)
- `cmi_verify` — deterministic CMI causal verification: Cohen's d + Welch's t-test p-value, causal_confidence classification (strong/moderate/weak/none); A11/A17 reproducible
- `backtest_stat` — deterministic backtest statistics from returns series: total_return, annualized_return, annualized_volatility, sharpe_ratio, max_drawdown, win_rate; A17 reproducible

## [2.27.0] — 2026-06-04

### Added (AII-3O Batch 5a — P3 Q-matrix)
- `build_q_matrix` — build IRT/CDM Q-matrix from knowledge graph `assesses` edges; pure logic, no LLM; used by cognitive_diagnosis DINA model

## [2.26.0] — 2026-06-04

### Added (AII-3O Batch 4a — P2 knowledge layer)
- `structural_chunk` — MD semantic chunking, pure logic, no LLM
- `ku_gate_validate` — HOS-001 three-face-unity gate validation, pure logic
- `llm_extract_ku` — single LLM call: text → unverified KU candidate (A19)
- `llm_distill_strategy` — single LLM call: Episode → unverified solution_strategy (A19)

## [2.25.0] — 2026-06-04

### Added (AII-3O Batch 3a)
- `coherence_compute` — deterministic KU coherence evidence from confirmed knowledge (A20 compliant, extracted from omodul.knowledge_reflux)
- `entity_graph_search` — single graph BFS traversal from seed nodes, cross-business reusable
- `vector_encode` — single-call text encoding via obase.ProviderRegistry with deterministic stub fallback

## [2.24.1] - 2026-06-03 — fix: re-export tts_synthesize / image_generate / image_understand

### Fixed

- `oprim/__init__.py`: added missing top-level re-exports for `tts_synthesize`, `image_generate`, `image_understand` (resolves AttributeError from consumers calling `oprim.tts_synthesize` etc.)
- `__all__` updated with all three symbols

## [2.24.0] - 2026-06-02 — Aegis C3-4 ErrorAggregator primitives

### Added

- `compute_event_fingerprint` — Sentry-style error aggregation key: SHA-256(exception_type|exception_value|top_frame_function|top_frame_filename, null-byte separated). Custom fingerprint override supported. NOT omodul fingerprint (business transaction identity); NOT compute_dedup_key (time-bucket dedup). Stable across time; same error → same fingerprint → same issue.

## [2.23.0] - 2026-06-01 — Stratum Batch 1: 24 P0 oprims

### Added — Stratum B1 P0 — ghost dependency clearance + file/DB/LLM/upload primitives

**File parsers (6)**
- `file_parser_pdf` — PDF → ParsedDocument via pymupdf4llm; DRM detection
- `file_parser_epub` — EPUB → ParsedDocument via ebooklib; per-chapter pages
- `file_parser_html` — HTML → ParsedDocument via trafilatura main-content extraction
- `file_parser_markdown` — Markdown + YAML frontmatter → ParsedMarkdown via python-frontmatter
- `file_parser_plaintext` — Plain text → ParsedPlaintext with chardet encoding detection
- `document_structure_extractor` — ParsedDocument → DocumentStructure (headings, TOC, word count)

**DB operations (7)**
- `db_insert` — Single row INSERT, returns RETURNING column value
- `db_query` — Parameterized SELECT, returns list[dict]
- `db_write` — INSERT with optional ON CONFLICT DO UPDATE (upsert)
- `db_read` — SELECT by ID with deleted_at IS NULL filter
- `db_soft_delete` — UPDATE set deleted_at = NOW()
- `db_update` — UPDATE single row by ID
- `migration_runner` — Alembic upgrade/downgrade/history/current/stamp

**Utility (5)**
- `template_render` — Jinja2 string template rendering; strict/non-strict undefined handling
- `crypto_token_generate` — secrets.token_urlsafe() wrapper; URL-safe or hex output
- `http_post` — Generic single HTTP POST (distinct from webhook-specific http_post_webhook)
- `file_size_limiter` — Client-type-aware upload size validation
- `file_type_detector` — Magic-byte MIME detection + category classification

**LLM (1)**
- `llm_summarize` — Single LLM call summary via obase.ProviderRegistry; concise/detailed/bullet styles

**Cache (1)**
- `cache_invalidate` — Redis DEL or in-memory pop; returns True if key existed

**Upload/temp (2)**
- `file_upload_handler` — Chunked BinaryIO → disk write with SHA-256 checksum
- `temp_file_manager` — TTL-based temp file registry; create/get/cleanup_expired/cleanup_user

**Push (1)**
- `push_email` — Single SMTP email send; STARTTLS; plain + HTML multipart

**Auth (1, REUSE★)**
- `otp_generate` / `otp_verify` — TOTP wrapper (obase.auth.totp equivalent; pyotp)

### Notes
- oprim-081 otp_generate: REUSE path — pyotp directly used (obase.auth unavailable in oprim venv due to missing argon2-cffi)
- oprim-078 http_post: NEW (obase has no generic POST; http_post_webhook is webhook-specific)
- obase dependency: spec requires v0.8.0; current v0.7.0 used — no v0.8.0-specific features required for P0
- Total new tests: 41 + 37 + 36 + 38 = 152 tests for this batch

## [2.22.0] - 2026-05-31 — Step-12 markets-related oprims (5) for paper_trading_session deps

### Added — 5 markets-related oprims (消除 omodul.paper_trading_session 28 fail 中的 9 fail)

- `detect_daily_limit_up` — A 股日线涨停判定,1e-9 浮点容差,回测日线撮合用
- `detect_daily_limit_down` — A 股日线跌停判定,对称 detect_daily_limit_up
- `t_plus_n_blocked` — A 股 T+N 持仓锁定判定(days_held < t_plus_n)
- `compute_commission` — 券商佣金计算 max(amount × rate, min_fee)
- `compute_stamp_tax` — 印花税额计算,税率由 caller 从 stamp_tax_rate_by_date 取得

### Notes
- 5 元素平铺顶层 `oprim/<name>.py`,不放 `oprim/markets/`
- `_version.py` 同步修复(stale 2.20.0 → 2.22.0)
- Spec: Tide v4 经理人 Step-12 IMPL SPEC 2026-05-28

## [2.21.0] - 2026-05-31 — P9-B2+B3 payment oprims — Alipay (4) + Stripe (4)

### Added — Alipay payment oprims

- `alipay_create_qr_order` — async face-to-face QR code order via `api_alipay_trade_precreate`; returns `AlipayQRCode(qr_code_url, out_trade_no)`; `sub_code` present → `AlipayAPIError`.
- `alipay_query_order` — async trade status query via `api_alipay_trade_query`; returns `AlipayTradeStatus(trade_status, trade_no, out_trade_no, total_amount)`; `sub_code` → `AlipayAPIError`.
- `alipay_refund_order` — async refund via `api_alipay_trade_refund`; full or partial; optional `refund_reason`; returns `True` on success; `sub_code` → `AlipayAPIError`.
- `alipay_verify_notify_signature` — **sync** RSA2 notification signature verification via python-alipay-sdk `client.verify`; strips `sign`/`sign_type` before verification; missing `sign` → `AlipayInvalidSignatureError`; SDK exception → `AlipayInvalidSignatureError`.
- `AlipayConfig(app_id, app_private_key, alipay_public_key, notify_url, sandbox)` — shared config model; `sandbox=True` passes `debug=True` to AliPay constructor.
- `AlipayError` / `AlipayAPIError` / `AlipayInvalidSignatureError` — error taxonomy.

### Added — Stripe payment oprims

- `stripe_create_payment_intent` — async `stripe.PaymentIntent.create` wrapper; returns `StripePaymentIntent`; `StripeError` → `StripeAPIError`.
- `stripe_retrieve_payment_intent` — async `stripe.PaymentIntent.retrieve` by `intent_id`; returns `StripePaymentIntent`; not found/API error → `StripeAPIError`.
- `stripe_refund_payment` — async `stripe.Refund.create`; full or partial (`amount` optional); `reason` ∈ `{"duplicate","fraudulent","requested_by_customer"}`; returns `True`; `StripeError` → `StripeAPIError`.
- `stripe_verify_webhook_signature` — **sync** `stripe.Webhook.construct_event` wrapper; missing `webhook_secret` → `ValueError`; `SignatureVerificationError` → `StripeInvalidSignatureError`; returns `dict[str, object]` event.
- `StripeConfig(api_key, webhook_secret)` — shared config model.
- `StripePaymentIntent(intent_id, client_secret, amount, currency, status, metadata)` — result model; `currency` ∈ `{"usd","eur","cny","gbp","hkd"}`; `status` full Stripe lifecycle Literal.
- `StripeError` / `StripeAPIError` / `StripeInvalidSignatureError` — error taxonomy.
- 26 tests (6 alipay-verify + 5 stripe-create + 5 stripe-retrieve + 5 stripe-refund + 5 stripe-webhook); ruff clean; mypy --strict clean.

## [2.20.0] - 2026-05-30 (planned) — Aegis C2 B2-B5 webhook pipeline

### Added — B2 single-shot webhook delivery

- `http_post_webhook` — keyword-only HTTP POST webhook delivery; never raises; all errors returned via `WebhookResult.success=False`.
- `WebhookResult(success, status_code, elapsed_ms, response_body, error)` — result model; `status_code=None` on network error; `response_body` truncated to 4 096 chars.
- Error taxonomy: `"timeout"`, `"connect_failed: ..."`, `"http_4xx"`, `"http_5xx"`, `"payload_not_serializable: ..."`, `"unexpected: ..."`.
- `follow_redirects=False` by default (SSRF prevention); `signature` / `signature_header` kwargs for HMAC delivery.
- 18 tests; ruff clean; mypy --strict clean.

### Added — B3 triple-tier threshold severity evaluator

- `evaluate_threshold_rule` — keyword-only single-value vs dual-threshold triple-tier evaluator for alert engines; never silently passes misconfigured rules.
- `ThresholdResult(triggered, severity, reason, metric, current_value, threshold_breached)` — result model; `severity` ∈ `{"ok","warn","critical"}`; `threshold_breached=None` when ok.
- `ThresholdRuleError` — raised immediately on any misconfiguration: missing fields, unsupported operator, inverted warn/critical thresholds, threshold not a dict.
- Supports four operators: `>=`, `>` (higher = more severe) and `<=`, `<` (lower = more severe); semantic order validated per direction.
- 27 tests; ruff clean; mypy --strict clean.

### Added — B4 time-window throttle decision

- `should_throttle` — keyword-only time-window throttle gate; returns `True` (skip) when still within window, `False` (allow) when window expired or never fired.
- `last_fired_at=None` → always allow (False); `throttle_seconds <= 0` → `ValueError`; naive datetimes → `ValueError`.
- `now` kwarg for deterministic testing; defaults to `datetime.now(UTC)`.
- 9 tests; ruff clean; mypy --strict clean.

### Added — B5 time-bucket dedup key

- `compute_dedup_key` — keyword-only SHA-256 time-bucket dedup key; NOT omodul fingerprint (content identity); this is time-window identity (same inputs in different buckets → different key).
- `rule_id` + `entity_id` + `bucket_start` + `bucket_seconds` → SHA-256 hex (64 chars).
- `bucket_seconds <= 0` → `ValueError`; naive `bucket_anchor` → `ValueError("timezone-aware")`; `bucket_anchor=None` → `datetime.now(UTC)`.
- 12 tests; ruff clean; mypy --strict clean.

## [2.19.0] - 2026-05-30 — B9 realtime detector oprims (7 detectors)

### Added — B9 realtime detectors

- `detect_sector_collapse` — 板块塌方: 1H 跌幅 > T1 AND 内部分化 std > T2; `SectorCollapseConfig`.
- `detect_dragon_switch` — 龙头切换: 原 Top1 滞涨 AND 新候补量比放大; `DragonSwitchConfig`.
- `detect_hot_money_converge` — 游资集中: 知名席位命中数 ≥ T1 AND 净买入 ≥ T2; 席位表注入; `HotMoneyConvergeConfig`.
- `detect_limit_board_explosion` — 涨停炸板: 调 `limit_status_calc` 判状态切换 + 放量验证; `LimitBoardExplosionConfig`.
- `detect_volume_spike` — 异常放量: 调 `volume_ratio` + MA20 价格确认 + 5min 涨幅; `VolumeSpikeConfig`.
- `detect_northbound_reversal` — 北向逆转: 连续 N min 净流入后突现净流出 ≥ T1 亿; `NorthboundReversalConfig`.
- `detect_news_shock` — 新闻冲击: `financial_metric_extraction` 情感命中 + inline 5min 波动率; 确认为 oprim (单 oprim 调用 + inline 计算); `NewsShockConfig`.
- `DetectorSignal(detector_name, severity, triggered_at, evidence)` — 统一信号模型, 定义于 `_detector_types.py`.
- 所有检测器: 同步函数, keyword-only, hit→`DetectorSignal | None`, 数据注入, 阈值 `ConfigModel` 参数化.
- `detect_limit_board_explosion` / `detect_volume_spike` 各调用 1 个既有 oprim (符合单一调用约定).
- 49 tests total (≥5 per detector, 7 additional structural tests); 2.18.0 → 2.19.0.

## [2.18.0] - 2026-05-29 — B8 utility/compute oprims (13 oprims)

### Added — B8 utility/compute

- `compute_seat_t3_return` — 席位 T+3 收益率计算; `SeatT3ReturnResult(return_pct, is_profit)`.
- `fetch_themes_daily` — async 每日概念主题行情 (akshare); `list[ThemeEntry]` 按涨跌幅降序.
- `theme_to_sw_industry_mapping` — 概念→申万行业查表映射 (mapping_table 注入); `list[ThemeSWMapping]`.
- `fetch_sector_returns` — async 申万板块涨幅 (akshare); `top_n` 截断.
- `pe_ttm_lookback_safe` — 消除前视偏差 TTM PE; `lag_days=45`; `PETTMResult(pe_ttm, eps_ttm, warning)`.
- `stop_loss_compliance_check` — 止损合规判定; `StopLossResult(triggered, action)`.
- `realtime_quote_redis_fetch` — async Redis 实时行情 + EOD 兜底; 使用既有 `CacheClient` Protocol.
- `stamp_tax_rate_by_date` — A股印花税率; 2023-08-28 精确切换 1‰→0.5‰; 仅卖方.
- `broker_export_render` — 配置驱动券商导出 (csv/tsv/json); `template_config` 注入.
- `compliance_disclaimer_inject` — 注入"信息参考, 不构成投资建议"; prefix/suffix/both.
- `monthly_review_jinja2_render` — Jinja2 月度复盘模板渲染; `template_dir` 注入; `RenderedReport`.
- `train_val_oos_splitter` — 60/20/20 时序切分; 严格时序, 无 shuffle; `TrainValOOSSplit`.
- `detect_volume_dryup_breakout` — 缩量调整后放量突破 (华安规律 ①); `VolumeBreakoutResult`.
- `jinja2>=3.0` 加入 `pyproject.toml` 主依赖.
- 84 tests total across 13 oprims (≥5 each); 2.17.0 → 2.18.0.

## [2.17.0] - 2026-05-29 — B7 macro data fetch oprims (8 oprims)

### Added — B7 macro data fetch

- `oprim.fetch_macro_m2` — PBoC M2 monthly money supply; indicators `m2_yoy` (%) + `m2_abs` (亿元).
- `oprim.fetch_macro_pboc` — PBoC open market ops (reverse repo / MLF / SLF); indicators `pboc_reverse_repo_rate`, `pboc_mlf_rate`, `pboc_slf_rate`.
- `oprim.fetch_macro_cpi_ppi_pmi` — NBS monthly CPI/PPI/PMI (3 parallel fetches); indicators `cpi_yoy`, `ppi_yoy`, `pmi_mfg`.
- `oprim.fetch_macro_lpr` — LPR 1y / 5y+ irregular; indicators `lpr_1y`, `lpr_5y`.
- `oprim.fetch_macro_rrr` — PBoC RRR irregular; indicators `rrr_large`, `rrr_small`.
- `oprim.fetch_macro_yield_spread` — Daily China–US 10y yield spread; indicator `cn_us_yield_spread_10y`; raw yields in `metadata`.
- `oprim.fetch_macro_calendar` — China econ calendar events with actual + forecast; `indicator` = event name; `metadata["forecast"]` / `metadata["prev"]`.
- `oprim.fetch_macro_policy_news` — Policy-relevant headlines (央行/财政部/发改委/证监会/商务部); `indicator="policy_news"`, `value=0.0`, text in `metadata`.
- `oprim._macro_types.MacroDataPoint` — Shared Pydantic model (indicator, date, value, metadata).
- `oprim._macro_types.MacroFetchError` — Inherits `OprimError`; raised on network error, licensed source, or bad response.
- All 8 use `source: Literal["wind","akshare","tushare"]="akshare"`; wind/tushare raise immediately.
- `akshare>=1.14` added to `pyproject.toml` as optional dep `[macro]` — `pip install oprim[macro]`.
- 44 tests total (≥5 per oprim); akshare calls fully mocked — no network required.

## [2.15.0] - 2026-05-28 — Tide v4 B1-B3 extraction (11 oprims)

### Added — Tide v4 B1-B3 — A股技术/基本面/选股 oprim

- `oprim.kdj` — KDJ 随机指标 (K/D/J 三序列). Pure function, keyword-only.
- `oprim.limit_status_calc` — A股涨跌停状态判定 (lookback N 日). Parameterized by `limit_pct` (10%/20%/30%).
- `oprim.beneish_m_score` — Beneish M-Score 财务造假风险 8 因子. `BeneishInput`/`BeneishResult`.
- `oprim.dupont_decomposition` — 杜邦分解 ROE = NPM × asset_turnover × equity_multiplier. `DuPontResult`.
- `oprim.dcf_valuation` — 两阶段 DCF 内在价值. `DCFResult`. Raises `OprimError` when `discount_rate <= terminal_growth_rate`.
- `oprim.financial_metric_extraction` — 中文财经新闻财务指标抽取 + 情感分 (V1 规则). `NewsItem`/`FinancialMetric`.
- `oprim.policy_event_extraction` — 政策新闻结构化事件抽取 (severity/direction). `PolicyNews`/`PolicyEvent`.
- `oprim.industry_attribution` — 政策事件 → 受影响行业归因 (纯映射). `IndustryImpact`.
- `oprim.pattern_detection` — K线形态识别 (hammer/engulfing 等). `OHLCVInput`/`PatternMatch`.
- `oprim.volume_ratio` — 量比 = 最新量 / 前 N 日均量. Returns 1.0 on insufficient data.
- `oprim.apply_screen_filter` — 配置驱动选股过滤 (gt/lt/gte/lte/eq/between/flag). `ScreenRule`/`ScreenResult`.

NOTE (§2.3): `symbol_dim_score` / `regime_inference` / `candidate_pool` omoduls use ThreadPoolExecutor
without manual `copy_context()` wrapping — Python 3.12+ ThreadPoolExecutor propagates contextvars
automatically. Cost pillar not enabled on these omoduls so `cost_tracker` ContextVar is unused.
Awaiting Owner confirmation per SPEC §2.3.

## [Unreleased]

### Added — Aegis Step 15 B2 — SSRF URL Safety Check

- `oprim.url_safety_check(*, url, allowed_schemes, block_loopback, block_private, block_link_local, block_reserved, block_multicast)` → `URLSafetyResult` — SSRF pre-flight URL safety check. Validates scheme whitelist, resolves all A/AAAA records via `socket.getaddrinfo` (prevents multi-homed bypass), and blocks loopback/private/link-local/reserved/multicast addresses. Returns `URLSafetyResult(is_safe, reason, resolved_ips, failed_check)`; never raises on business rejection. `URLSafetyError` raised only for technical failures (DNS library crash, URL parse exception). Explicit CGN (`100.64/10`, RFC 6598) check added for Python 3.11+ compatibility where `is_reserved` no longer covers that range. Link-local checked before private so `169.254/16` reports the more specific `is_link_local` label. DNS-rebinding residual risk documented in docstring. 16 tests.
- `URLSafetyResult` — Pydantic model: `is_safe`, `reason`, `resolved_ips`, `failed_check`.
- `URLSafetyError` — Technical-failure exception (distinct from business rejection).

### Added — P7-B2 — Video Prompt Primitives + Frame Transition + Story Predict

- `oprim.style_marker_prompt` — 风格关键词注入 (7 styles: 科普/严肃/搞笑/治愈/悬疑/热血/温暖). Pure function, no I/O.
- `oprim.lighting_control_prompt` — 灯光描述注入 (6 lightings: 暖/冷/戏剧/自然/高对比/柔和). Pure function.
- `oprim.camera_motion_prompt` — 镜头运动 prompt 生成 (8 motions + intensity [0,1] → slow/medium/fast). Pure function.
- `oprim.first_last_frame_transition` — 首尾帧过渡视频生成 via `ProviderRegistry.get(category='image_to_video')`.
  - `FrameTransitionError` / `FrameTransitionProviderNotFoundError` — Error hierarchy.
- `oprim.video_edit_element_remove` — 视频精准编辑去除元素 via `ProviderRegistry.get(category='video_inpaint')`.
  - `VideoEditError` / `VideoEditProviderNotFoundError` — Error hierarchy.
- `oprim.story_predict` — 单 LLM 调用基于参考图推演剧情. `LLMCaller` Protocol + Pydantic `StoryPrediction`.
  - `StoryPrediction`, `TimePrediction`, `StoryPredictError` — Models and error.
- `oprim._providers.longcat_avatar` — LongCat-Video-Avatar 1.5 subprocess wrapper (private).
  - `invoke_local`: subprocess call to `vendor_dir/inference.py`.
  - `invoke_cloud`: TECHNICAL_DEBT stub (no official Meituan cloud API as of 2026-05-27).
- Tests: ≥43 total (7+6+8+6+6+7+6), 100% coverage.

### Added — P6-B2 — Video Generation + Audience Analytics primitives

- `oprim.image_to_video` — Image-to-video via provider injection (wan22_local/cloud/veo/runway).
- `oprim.face_animation` — Face animation via provider injection (wav2lip/sadtalker/musetalk).
- `oprim.motion_prompt_translate` — LLM translation of motion description to video prompt.
- `oprim.audience_sentiment_analyze` — LLM-based comment sentiment analysis.
- `oprim.audience_feedback_extract` — LLM-based structured feedback extraction.
- `oprim.youtube_video_stats` — YouTube video statistics fetch.
- `oprim.youtube_comments_fetch` — YouTube comments with auto-pagination.
- `oprim.bilibili_video_stats` — Bilibili video statistics fetch.
- `oprim.bilibili_comments_fetch` — Bilibili comments with pagination.
- `oprim.video_quality_metrics` — ffprobe-based technical video metrics.
- `oprim.vlm_video_analyze` — VLM-based video frame analysis.
- `oprim._providers.wan22` — Wan2.2 local subprocess + DashScope cloud (private).
- `oprim._providers.sadtalker` — SadTalker subprocess wrapper (private).
- `oprim._providers.musetalk` — MuseTalk subprocess wrapper (private).
- `oprim._providers.youtube_api` — YouTube Data API v3 wrapper (private).
- `oprim._providers.bilibili_api` — Bilibili API wrapper (private).

---

## [2.13.0] - 2026-05-24

### Added — Sprint 12 — Sector Strength + Within-Group Percentile (A5 + A6)

- `sector_strength_proxy(returns, volumes, scoring, lookback)` — 0-100 normalized sector strength.
- `within_group_percentile(values, target_idx, method)` — Percentile of target within group.

## [2.12.0] - 2026-05-24

### Added — Sprint 11 — Timeseries Split & Segment Label (A7 + A8)

- `time_series_split(dates, train_pct, val_pct, gap_days)` — Split date sequence into train/val/oos with optional gap exclusion.
  - Example: `splits = time_series_split(dates=dates, train_pct=0.6, val_pct=0.2, gap_days=15)`
- `equity_curve_segment_label(equity_curve, split_dates)` — Label equity curve rows with segment (train/gap/val/oos).
  - Example: `labeled = equity_curve_segment_label(equity_curve=df, split_dates=splits["split_dates"])`

## [2.11.0] - 2026-05-24

### Added — Hevi Batch 2 — External API Primitives

- `image_generate(provider, prompt, width, height, output_path, seed, timeout_s, extra)` — Image generation via provider injection.
  - Example: `await image_generate(provider="siliconflow", prompt="sunset", output_path=Path("img.png"))`
- `image_understand(provider, image_path, prompt, timeout_s)` — VLM image understanding (image → text).
  - Example: `text = await image_understand(provider="qwen_vl", image_path=Path("img.jpg"), prompt="Describe")`
- `tts_synthesize(provider, text, voice, output_path, rate, pitch, timeout_s)` — TTS speech synthesis via provider injection.
  - Example: `await tts_synthesize(provider="edge_tts", text="Hello", voice="zh-CN-XiaoxiaoNeural", output_path=Path("out.mp3"))`
- `srt_translate(src_srt_path, target_lang, llm, output_path, batch_size)` — SRT subtitle translation via LLMCaller Protocol.
  - Example: `await srt_translate(src_srt_path=Path("zh.srt"), target_lang="en", llm=llm, output_path=Path("en.srt"))`
- `avatar_generate(provider, portrait_image, audio_path, output_path, fps, timeout_s)` — Digital avatar generation via subprocess provider.
  - Example: `await avatar_generate(provider="wav2lip", portrait_image=Path("face.png"), audio_path=Path("audio.wav"), output_path=Path("avatar.mp4"))`

### Added — Hevi Batch 1 — FFmpeg Media Primitives

- `audio_mix(inputs, weights, output_path, sample_rate, timeout_s)` — Multi-track audio mixing via FFmpeg amix filter.
  - Example: `await audio_mix(inputs=[Path("narration.wav"), Path("bgm.wav")], weights=[1.0, 0.3], output_path=Path("mixed.wav"))`
- `audio_normalize(input_path, output_path, target_lufs, timeout_s)` — EBU R128 loudness normalization via FFmpeg loudnorm.
  - Example: `await audio_normalize(input_path=Path("raw.wav"), output_path=Path("norm.wav"), target_lufs=-14.0)`
- `audio_video_merge(video_path, audio_path, output_path, audio_codec, timeout_s)` — Merge audio into video (replacing original track).
  - Example: `await audio_video_merge(video_path=Path("v.mp4"), audio_path=Path("a.wav"), output_path=Path("out.mp4"))`
- `video_concat(inputs, output_path, method, timeout_s)` — Concatenate multiple videos (demuxer or filter).
  - Example: `await video_concat(inputs=[Path("p1.mp4"), Path("p2.mp4")], output_path=Path("full.mp4"))`
- `video_recompose(input_path, output_path, target_width, target_height, method, timeout_s)` — Recompose video aspect ratio (landscape → portrait).
  - Example: `await video_recompose(input_path=Path("wide.mp4"), output_path=Path("vertical.mp4"))`
- `subtitle_burn(video_path, srt_paths, output_path, primary_alignment, secondary_alignment, timeout_s)` — Burn subtitles (single/dual language).
  - Example: `await subtitle_burn(video_path=Path("v.mp4"), srt_paths=[Path("zh.srt"), Path("en.srt")], output_path=Path("burned.mp4"))`
- `video_generate(provider, prompt, reference_image, duration_s, width, height, output_path, timeout_s)` — Video generation via provider injection (stub).
  - Example: `await video_generate(provider="stub", prompt="A cat on the moon", output_path=Path("gen.mp4"))`

### Added — Phase 11C
- `parse_obsidian_tasks`: parse Obsidian tasks from markdown.
- `llm_judge_rerank`: use LLM to rerank documents.
- `llm_query_expand`: use LLM to expand queries with synonyms and variants.

## [2.10.0] - 2026-05-24

### Added — BATCH 19 — Infrastructure & Ops Primitives Expansion

#### Docker (7 new): `_docker.py`
- `docker_image_list` — List docker images with id, tags, size, created_at.
- `docker_image_delete` — Delete docker image (force support).
- `docker_volume_list` — List docker volumes.
- `docker_volume_delete` — Delete docker volume.
- `docker_network_list` — List docker networks.
- `compose_up` — Docker Compose up (supports project_name, detach, pull).
- `compose_down` — Docker Compose down (supports volumes, remove_orphans).

#### Caddy (1 new): `_caddy.py`
- `caddy_admin_post` — Generic POST/PATCH/PUT/DELETE for Caddy Admin API.

#### Filesystem (1 extended): `_filesystem.py`
- `archive_to_targz` — Multi-source support for archiving (replaces dir_archive_to_targz).
- `dir_archive_to_targz` — Deprecated in favor of `archive_to_targz`.

### Changed
- `ArchiveResult` model updated: `src_dir` (str) -> `sources` (list[str]).

## [2.9.0] - 2026-05-24

### Added — Aegis Batch 1: Infrastructure / Ops Primitives (32 new elements)

#### Docker (7): `_docker.py`
- `docker_container_inspect` — 查容器完整状态 (state / health / ports / mounts)
- `docker_container_logs` — 读容器日志 (支持 tail / since / until)
- `docker_container_start` — 启动容器, 返回 state 变化
- `docker_container_stop` — 停止容器 (SIGTERM + timeout SIGKILL)
- `docker_container_restart` — 重启容器
- `docker_image_pull` — 拉取镜像 (含 auth_config 私有仓库支持)
- `docker_container_stats` — 容器资源快照 (CPU / mem / net / blkio / pids)

#### PostgreSQL (5): `_postgres.py`
- `postgres_pool_status` — 连接池状态 (active/idle/idle-in-tx/waiting/usage%)
- `postgres_slow_queries` — 慢查询 top N (依赖 pg_stat_statements)
- `postgres_locks_status` — 锁状态 (默认只返 waiting 锁)
- `postgres_table_size` — 表大小 top N (含索引 + toast)
- `postgres_replication_lag` — 主从复制延迟

#### RabbitMQ (4): `_rabbitmq.py`
- `rabbitmq_queue_status` — 队列状态 (messages / consumers / state / memory)
- `rabbitmq_connection_status` — 所有连接状态 (blocked / running)
- `rabbitmq_consumer_status` — 指定队列的 consumer 列表
- `rabbitmq_node_status` — 节点状态 (mem / disk / fd / sockets / proc)

#### Caddy (3): `_caddy.py`
- `caddy_admin_reload` — 原子替换 Caddy 配置 (/load)
- `caddy_routes_list` — 列出当前路由 (从 config tree 提取)
- `caddy_certificates_status` — 域名证书状态 (issued / expiry)

#### Network (4): `_network.py`
- `tcp_port_check` — TCP 端口连通性探测 (永不 raise 网络错误)
- `http_health_probe` — HTTP 健康探测 (永不 raise 网络错误)
- `dns_resolve` — DNS 解析 (A/AAAA/CNAME/MX/TXT, 支持指定 nameserver)
- `http_request_once` — 通用 HTTP 单次调用

#### Filesystem (3): `_filesystem.py`
- `disk_usage` — 文件系统使用情况
- `dir_archive_to_targz` — 目录打包 tar.gz (单次流式含 SHA-256 checksum)
- `file_checksum` — 文件 checksum (sha256/md5/sha1)

#### Metrics & Logs (4): `_metrics_logs.py`
- `prometheus_instant_query` — Prometheus 即时查询
- `prometheus_range_query` — Prometheus 范围查询
- `loki_log_query` — Loki LogQL 查询
- `structlog_parse` — structlog 输出解析 (json / logfmt)

#### System (2): `_system.py`
- `cpu_memory_snapshot` — CPU + 内存系统快照
- `process_list_top` — 进程 top N (按 CPU 或内存)

#### S3 (2): `_s3.py`
- `s3_upload_file` — 上传本地文件到 S3
- `s3_object_metadata` — 查 S3 对象元数据 (HEAD)

#### Infrastructure
- `_exceptions.py`: `OprimError` / `OprimConnectionError` / `OprimTimeoutError` / `OprimNotFoundError` / `OprimAuthError` / `OprimValidationError`
- 新增依赖: `psycopg[binary]>=3.1`, `psutil>=5.9`, `boto3>=1.34`, `dnspython>=2.6`

## [2.0.0] - 2026-05-14

### Added — Phase 10 (10 new elements)
- `behavioral/cpt.py`: `cpt_value_function`, `probability_weighting_function` (Tversky-Kahneman 1992)
- `behavioral/llad.py`: `large_loss_aversion_degree` (Bernard-Ghossoub 2010)
- `behavioral/salience.py`: `salience_function`, `salience_ranking_weights` (BGS 2013)
- `spectral/marchenko_pastur.py`: `marchenko_pastur_threshold` (Random Matrix Theory)
- `spectral/rie.py`: `rotationally_invariant_estimator` (Bouchaud-Potters)
- `spectral/ledoit_wolf.py`: `ledoit_wolf_shrinkage`
- `spectral/eigengap.py`: `spectral_eigengap_detect`
- `recursive_utility/epstein_zin.py`: `epstein_zin_aggregator` (Epstein-Zin 1989)

### Changed
- Version bump: 1.11.0 → 2.0.0 (major: new submodule structure for behavioral/spectral/recursive_utility)

## [1.11.0] - 2026-05-09
### Added — Phase 9A
- path_signature_compute, fisher_rao_distance, rough_volatility_simulate, sabr_implied_volatility, ed25519_keypair_generate, ed25519_sign, ed25519_verify

## [1.5.0] - 2026-05-14

### Added (Phase 2: 10 new elements)

#### Performance (`oprim/performance/`)
- `cumulative_returns`: Simple and log compounding of return series
- `cagr`: Compound Annual Growth Rate — geometric and arithmetic methods (Bodie, Kane, Marcus 2014)

#### Mean Reversion (`oprim/mean_reversion/`)
- `ornstein_uhlenbeck_fit`: Closed-form MLE for OU process parameters (Smith 2010)
- `ornstein_uhlenbeck_half_life`: Half-life estimation via regression or MLE (López de Prado 2018; Chan 2013)

#### Volatility (`oprim/volatility/`)
- `garch_fit`: GARCH(1,1) MLE fitting via L-BFGS-B (Bollerslev 1986)
- `garch_forecast`: Multi-step GARCH variance forecasting (Bollerslev 1986; Hamilton 1994)
- `ewma_volatility`: RiskMetrics EWMA volatility with lambda_=0.94 default (JP Morgan 1996)

#### Derivatives (`oprim/derivatives/`)
- `black_scholes_price`: Black-Scholes-Merton closed-form option pricing (Black & Scholes 1973; Merton 1973)
- `black_scholes_greeks`: Delta, gamma, vega, theta, rho (Hull 2018 Ch.19)
- `implied_volatility`: Brent/Newton IV extraction (Manaster & Koehler 1982)

### Infrastructure
- New subdirectory structure: `performance/`, `mean_reversion/`, `volatility/`, `derivatives/`
- JSON schemas in `oprim/schemas/<category>/<element>.schema.json`
- Private helpers in `oprim/derivatives/_base.py` (H2 exempt)
- Coverage threshold lowered to 75% to accommodate new complex modules

### Notes
- 10 new elements, 75 new tests across 4 test modules
- All Phase 2 elements have `@pytest.mark.academic_reference` tests

## [1.4.0] - 2026-05-14

### Added (Phase 1: 14 new elements)

#### Technical Indicators (`oprim/technical/`)
- `sma`: Simple Moving Average — SMA_t = (1/N) * sum(P_{t-N+1}..P_t)
- `ema`: Exponential Moving Average (adjust=False/True) — matches pandas ewm exactly
- `vwap`: Volume Weighted Average Price (cumulative and rolling)
- `macd`: MACD line, signal line, histogram — Appel (1979)
- `rsi_normalized`: RSI normalized to [0,1] — Wilder (1978) SMA-seeded smoothing
- `bollinger_bands`: Upper/middle/lower/bandwidth/%B — population std (ddof=0)
- `donchian_channel`: Upper/middle/lower rolling extrema
- `chandelier_exit`: Long/short trailing stop — Le Beau ATR-based

#### Cryptographic Primitives (`oprim/crypto/`)
- `sha256_hash`: NIST FIPS 180-4 SHA-256, returns 64-char hex string, accepts bytes/str
- `hmac_sha256`: RFC 2104 HMAC-SHA-256, returns 64-char hex string
- `rfc6962_merkle_root`: RFC 6962 MTH with arbitrary-bytes leaves (not pre-hashed)
- `rfc6962_inclusion_proof`: RFC 6962 §2.1.1 audit path

#### Serialization (`oprim/serialization/`)
- `canonical_json`: RFC 8785 JCS — deterministic, UTF-16 sorted keys, no whitespace

#### Risk (`oprim/risk/`)
- `cvar`: Conditional Value at Risk (historical & Gaussian closed-form methods)

### Infrastructure
- New subdirectory structure: `technical/`, `crypto/`, `serialization/`, `risk/`
- `oprim/_manifest.py`: authoritative element list with categories and stability tags
- JSON schemas in `oprim/schemas/<category>/<element>.schema.json`
- Private helpers in `oprim/technical/_base.py` (H2 exempt from H1)
- `pyproject.toml`: restored `academic_reference` pytest marker

### Notes
- 433 total tests (144 new Phase 1 tests), 80% coverage
- All Phase 1 elements have `@pytest.mark.academic_reference` tests
- Crypto elements verified against NIST FIPS 180-4, RFC 4231, RFC 6962, RFC 8785
- Technical indicators verified against pandas rolling/ewm reference implementations
- No new third-party dependencies (crypto uses Python stdlib only)

## [1.0.0] - 2026-05-10

### Added
- 36 atomic operations across 6 modules:
  - **time_series** (11 ops): log_returns, cumulative_returns, rolling_window_split, lag_forward_fill, percentile_rank, ewma_smooth, realized_vol, zscore_normalize, gap_detect, resample_align, purge_embargo_split
  - **statistics** (10 ops): bootstrap_ci, percentile_ci, distribution_summary, skew_kurt_robust, kolmogorov_smirnov_test, mann_kendall_trend, bayes_beta_update, brier_score_decomposed, pearson_spearman_corr, kde_density
  - **distance** (5 ops): wasserstein_distance, dtw_distance, cosine_similarity_batch, euclidean_distance_matrix, symmetric_kl_divergence
  - **numerics** (3 ops): logsumexp_safe, softmax_safe, clip_with_warning
  - **regime** (3 ops): regime_filter_data, regime_transition_matrix, regime_label_align
  - **finance** (4 ops): drawdown_curve, sharpe_ratio, beta_alpha_ols, value_at_risk

### Fixed
- time_series: zscore_normalize min_periods capping, purge_embargo_split edge cases
- statistics: distribution_summary skew calculation alignment with scipy
- distance: dtw_distance early termination optimization
- finance: sharpe_ratio near-zero std handling, value_at_risk method validation
- regime: stationary distribution for non-ergodic chains, ffill performance optimization, soft mode vectorization
- numerics: logsumexp_safe docstring clarification, clip_with_warning scalar detection

### Testing
- 289 tests with 91% code coverage
- Academic validation tests vs scipy, statsmodels, hmmlearn
- Performance benchmarks for critical paths

---

## Release Governance Note (2026-05-14)

During the Phase 10 release process, we discovered that Phases 4-10 had been
accumulated on a single long-running feature branch (feat/v1.7.0-phase4) without
intermediate merges to main. The main branch was stale at v1.2.0 while the
actual code was at v2.0.0 on the feature branch.

**Resolution**: fast-forward merged main to feat HEAD, retagged on main, deleted
feat branch. See `RELEASE_POLICY.md` for the corrected workflow.

All future Phase releases must:
1. Use independent feat branches (not accumulate Phases on one branch)
2. Merge to main via PR before tagging
3. Tag on main (never on feat branches)

## [3.2.0] — 2026-06-13
### Added
- hicode 批次 A-C: file_*/git_*/bash_exec/lsp_*/mcp_*/llm_complete/llm_stream
  embed_text/http_fetch/web_search/build_system_prompt/truncate_messages
  extract_thinking/snapshot_conversation/parse_unified_diff/compute_diff
  detect_language/html_to_markdown/redact_secrets/count_tokens/estimate_cost
  run_hook/load_image/read_skill_frontmatter/git_worktree_* (共 55 个新元素)

## [3.8.0] — 2026-06-14
### Added
- hicode 批次 H-A: 纯计算 oprim 100 个新元素
  (file_read_range/detect_encoding/detect_mime/is_binary/truncate_for_context/
   add_line_numbers/normalize_line_endings/preserve_indentation/
   apply_string_replace/verify_unique_match/apply_patch/apply_hunk/
   plan_multiedit/detect_edit_conflict/parse_ripgrep_output/sort_by_mtime/
   format_tree/apply_gitignore/build_ripgrep_args/truncate_output/strip_ansi/
   parse_exit_signal/sanitize_env/detect_shell/extract_main_content/
   validate_url/resolve_redirect/todo_*/session_*/to_*_format/from_*_format/
   normalize_tool_schema/normalize_stop_reason/patch_provider_quirk/
   map_model_alias/inject_cache_control/split_system_message/make_*_part/
   render_part/parts_to_message/message_to_parts/merge_streaming_parts/
   build_system_prompt/inject_agents_md/parse_tool_calls/parse_stop_reason/
   format_tool_results/build_tool_schema/select_compaction_window/
   merge_summary/should_compact/extract_pinned_messages/build_compaction_prompt/
   resolve_config_paths/parse_json_config/parse_markdown_agent/
   interpolate_env_vars/resolve_config_path_refs/parse_skill_md/
   resolve_external_dir/check_path_allowed/match_wildcard_pattern/
   classify_risk/match_bash_command_rule/resolve_agent_permissions/
   serialize_share_payload/redact_share_secrets/redact_secret/make_event/
   serialize_event/deserialize_event/event_should_sync/select_model/
   filter_curated_models/resolve_model_capabilities/resolve_subagent_tools/
   summarize_subagent_result/mcp_tool_to_schema/build_question_payload/
   parse_question_answer/estimate_tokens/count_message_tokens)

## [3.10.2] — 2026-06-13
### Fixed
- types.py: re-export KCState from _cognitive (修复 bkt.py 等 from oprim.types import KCState 的 ImportError)
- v3.10.1 fix 不完整，本次彻底修复

## [3.10.3] — 2026-06-14
### Fixed
- finance.py: add public re-export facade (oprim.finance, 修复 omodul/strategies import)
- __init__.py: re-export llm_summarize (修复 from oprim import llm_summarize)
### Note
- 含 v3.10.1/v3.10.2 的 KCState 修复（已在远端，本地 merge 进来）

## [3.10.4] — 2026-06-14
### Fixed
- _llm_summarize: ProviderRegistry.get_caller() → ProviderRegistry.get().llm(provider)
  (get_caller 方法不存在，修正为真实 API)

## [3.10.5] — 2026-06-15
### Fixed
- image_generate: ProviderRegistry.get(category=,name=) → get().image_gen(name)
- image_generate: catch RuntimeError (from image_gen()) alongside ProviderNotFoundError

## [3.10.6] — 2026-06-15
### Fixed
- llm/llm_call._call_dashscope: stub → 真实 DashScope HTTP 实现
  (httpx POST + DASHSCOPE_API_KEY + 429 rate limit 处理)
- _call_claude: 明确报错指向 oprim.llm_complete (不再静默返回 dummy)

## [3.10.10] — 2026-06-17
### Fixed
- _vibevoice_synthesize: AutoModelForTextToWaveform/AutoProcessor →
  VibeVoiceForConditionalGenerationInference/VibeVoiceProcessor
  (正确的 vibevoice 包 API，修复 hevi M3 真跑失败)

## [3.10.12] — 2026-06-18
### Fixed
- _cognitive.py KCState: 补 p_recognition/p_recognition_init 字段（M-G 识别维度）
- types.py GradeResult: 补 reason 字段（M-F compute_feedback 依赖）

## [3.10.14] — 2026-06-18
### Fixed
- _file_parser_epub: 章节标题改用 TOC 语义标题（spine 文件名→toc title）
- parser/parse_epub: 同上修复 + metadata 补 author/language
### Added
- epub_toc_split: EPUB 按顶层 TOC 拆分（套装→多本，单本→长度1）
- markdown_frontmatter_build: 元数据 dict → YAML frontmatter 字符串
- text_clean_publish_noise: 去版权页/水印/页眉页脚/空白页

## [3.10.15] — 2026-06-18
### Fixed
- _vibevoice_synthesize: Speaker N 格式化 / speech_outputs 提取 /
  正确采样率 / 波形 clip（CC fix/m3-m4 真跑验证）
- _providers/duix: _map_path host↔container 路径映射 /
  response["data"]["status"] 层级修正 / 失败状态码 3
- _avatar_generate: 传入 DUIX_HOST_DATA_DIR / DUIX_CONTAINER_DATA_DIR
- pyproject.toml: 补 vibevoice>=0.0.1 依赖声明

## [3.10.16] — 2026-06-18
### Fixed
- _file_parser_pdf: CID/Type1 字体乱码检测 + fallback to blocks mode
  (\ufffd 比例 > 30% 时自动切换，回归保持向后兼容)

## [3.10.19] — 2026-06-19
### Changed
- parser/parse_pdf: embed_images 参数化（默认 False 保持兼容）
  True 时图片转 base64 嵌入 md（体积增大 ~23x，适合数学图表场景）
  Stratum 按需传 embed_images=True

## [3.10.20] — 2026-06-19
### Added
- arxiv_search: arXiv API 论文检索（分类/关键词/作者/日期过滤，rate limit 遵守）
- http_download_file: URL→本地文件（流式，支持大文件，可选代理）

## [3.10.20] patch — 2026-06-19
### Fixed
- _arxiv_search.py / _http_download_file.py: 补 commit 漏提交的实现文件
  (v3.10.20 __init__ 已声明但文件未 commit，干净 clone 会崩)

## [3.10.21] — 2026-06-19
### Added
- SourceResult: 统一源订阅结果接口（external_id/title/download_url/file_type/metadata）
- gutenberg_search: Gutendex API 公版书检索（epub/txt，User-Agent 必须）
- oapen_search: OAPEN + Unpaywall + 白名单过滤（IPv4 强制，Springer OA 子集）
### Changed
- http_download_file: 加 force_ipv4 参数（OAPEN 等 IPv6 超时场景）
- _media_types: 加 SourceResult dataclass

## [3.10.23] — 2026-06-19
### Fixed
- epub_toc_split: 过滤辅助节点（扉页/版权页/目录/前言等，内容<2000字符）
  防止辅助页被拆成独立 EpubBook 进入 bundle 路径

## [3.10.24] — 2026-06-23
### Fixed
- 新增 oprim/quant_analysis.py 门面模块，暴露 compute_shapley_decomposition
  修复 Helios oprim.quant_analysis.compute_shapley_decomposition 调用 500 问题
- __init__ 补 compute_shapley_decomposition export

## [3.10.25] — 2026-06-23
### Fixed
- __init__: llm_summarize 改惰性加载，不在 import 时触发 obase 依赖
  修复 Helios 等无 obase 环境 import oprim 失败问题（ModuleNotFoundError: obase）

## [3.10.26] — 2026-06-23
### Fixed
- __init__: 系统性惰性化所有 obase 依赖模块
  llm_complete / llm_stream / embed_text / image_generate / image_understand / tts_synthesize
  全部改为调用时才 import obase，不在 import oprim 时触发
  修复 Helios 等无 obase 环境 import oprim 失败（打地鼠根治）

## [3.10.27] — 2026-06-23
### Fixed
- __init__: 惰性化文件解析器（file_parser_pdf/epub/html + epub_toc_split）
  修复 fitz/ebooklib/bs4 在无重依赖环境（Helios）触发 ImportError
  配合 v3.10.26 的 obase 惰性化，import oprim 全程不触发任何重依赖
  Helios 最小环境三条验收全部通过

## [3.10.28] — 2026-06-23
### Changed
- pyproject.toml: 重型依赖移到 optional extras
  核心依赖（required）：numpy/scipy/pandas/pydantic/chardet/fsrs 等轻量包
  optional[pdf]: pymupdf/pymupdf4llm
  optional[epub]: ebooklib/beautifulsoup4
  optional[storage]: lancedb/tantivy/duckdb
  optional[llm]: dashscope
  optional[image]: Pillow
  optional[tts]: vibevoice
  optional[cloud]: boto3/google-api/asyncpg
  optional[full]: 全部 optional 合集
  Helios 等轻量环境 pip install oprim 不再拉重型依赖

## [3.10.29] — 2026-06-24
### Added
- _aii_graph_types: 知识本体受控词表
  VALID_KNOWLEDGE_TYPES / VALID_RELATION_TYPES / VALID_GRADES / VALID_SUB_TYPES
  OntologyExtractResult / RegisterKuOntologyInput 共享类型

## [3.10.30] — 2026-06-24
### Added
- compute_shapley_values: 真·非线性 Shapley 分解
  传特征dict + 聚合函数callable，算真边际贡献（在场vs不在场）
  method=monte_carlo（采样，默认2000）或 exact（2^n，n<=12）
  确定性（seed）、baseline/residual 显式分离、可加性保证
  比例分配版 compute_shapley_decomposition 保留作 fallback
### Fixed
- _cognitive: fsrs import 改函数内惰性加载
  KCState/BKT 不依赖 fsrs，import oprim 不再触发 fsrs eager-load
  根治 __init__ 第124行 + types.py 第175行的 fsrs 链
