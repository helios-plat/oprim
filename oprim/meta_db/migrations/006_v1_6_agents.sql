-- Phase 11A: Agent run trace table
CREATE TABLE IF NOT EXISTS agent_runs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    params TEXT NOT NULL,                   -- JSON
    status TEXT NOT NULL,                   -- 'running' | 'completed' | 'failed' | 'timeout'
    trace TEXT,                             -- JSON array of AgentStep dicts
    citations TEXT,                         -- JSON array of Citation dicts
    output TEXT,                            -- JSON
    total_input_tokens INTEGER DEFAULT 0,
    total_output_tokens INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0.0,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_agent_runs_user_agent
    ON agent_runs(user_id, agent_name, started_at);

CREATE INDEX IF NOT EXISTS idx_agent_runs_status
    ON agent_runs(user_id, status, started_at)
