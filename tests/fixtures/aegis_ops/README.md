# Aegis Ops Fixtures

Real-world fixture samples from Aegis (运维诊断) oprim element tests.

Source: 脱敏后的 Helios/Helixa 生产数据 + 合成的 error case。

## Structure

- `rabbitmq/` — RabbitMQ Management API 响应样本
- `docker/` — Docker daemon logs / inspect 输出
- `postgres/` — pg_stat_activity / pg_locks 查询结果
- `prometheus/` — Prometheus /api/v1/query 响应
- `loki/` — Loki /loki/api/v1/query_range 响应

## Adding new fixtures

1. 从生产环境抓取真实响应 (curl / 日志)
2. 用 sanitize 脚本脱敏 (替换 user_id / token / IP / 内部 hostname)
3. 命名: `<scenario>.json` 或 `<scenario>.txt` (logs 用 .txt)
4. 在对应 oprim 测试文件加引用

## Naming conventions

- `<scenario>_normal.json` — 正常状态
- `<scenario>_error.json` — 错误状态
- `<scenario>_<bug_name>.json` — 特定真实 bug 场景 (e.g. `queue_heartbeat_loss.json`)
