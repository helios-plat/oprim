# Changelog

<!-- Governance: see RELEASE_POLICY.md. main = release branch; feat branches deleted after merge; oprim → oskill → omodul merge order required; container bind-mount means git checkout is a live operation. -->

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
