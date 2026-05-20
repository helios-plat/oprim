-- Phase 4 browser extension URL deduplication index
CREATE TABLE IF NOT EXISTS browser_ext_url_index (
    id             TEXT PRIMARY KEY,
    url            TEXT NOT NULL,
    normalized_url TEXT NOT NULL,
    substrate_id   TEXT NOT NULL,
    ingested_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(normalized_url)
);

CREATE INDEX IF NOT EXISTS idx_browser_ext_normalized
    ON browser_ext_url_index(normalized_url);
