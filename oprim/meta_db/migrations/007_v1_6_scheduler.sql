-- Phase 11A: Scheduled job tables
CREATE TABLE IF NOT EXISTS scheduled_jobs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    agent_params TEXT NOT NULL DEFAULT '{}',   -- JSON
    cron_expression TEXT NOT NULL,
    timezone TEXT NOT NULL DEFAULT 'Asia/Shanghai',
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    notify_on_completion BOOLEAN NOT NULL DEFAULT TRUE,
    notify_on_failure BOOLEAN NOT NULL DEFAULT TRUE,
    max_runtime_seconds INTEGER NOT NULL DEFAULT 1800,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS scheduled_job_runs (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    agent_run_id TEXT,
    status TEXT NOT NULL,     -- 'running'|'completed'|'failed'|'timeout'
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_scheduled_jobs_user
    ON scheduled_jobs(user_id);

CREATE INDEX IF NOT EXISTS idx_scheduled_job_runs_job
    ON scheduled_job_runs(job_id, started_at)
