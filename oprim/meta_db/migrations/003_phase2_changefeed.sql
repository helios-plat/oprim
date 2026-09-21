-- Phase 2 changefeed: per-device event log + snapshot markers
-- Append-only: INSERT only — never UPDATE or DELETE existing events

CREATE SEQUENCE IF NOT EXISTS changefeed_event_id_seq START 1;

CREATE TABLE IF NOT EXISTS changefeed_events (
    id         BIGINT DEFAULT nextval('changefeed_event_id_seq') PRIMARY KEY,
    device_id  TEXT NOT NULL,
    user_id    TEXT NOT NULL,
    event_type TEXT NOT NULL,
    aggregate_id TEXT,
    payload    TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    seq        BIGINT NOT NULL,
    UNIQUE (user_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_cfe_user_seq
    ON changefeed_events(user_id, seq);
CREATE INDEX IF NOT EXISTS idx_cfe_user_agg
    ON changefeed_events(user_id, aggregate_id);
CREATE INDEX IF NOT EXISTS idx_cfe_user_created
    ON changefeed_events(user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS changefeed_snapshots (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    device_id  TEXT NOT NULL,
    seq_at     BIGINT NOT NULL,
    file_id    TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_cfsnap_user_seq
    ON changefeed_snapshots(user_id, seq_at DESC);
