-- Monitoring schema (SPEC §12). Mounted into the postgres container's
-- docker-entrypoint-initdb.d and also created idempotently by monitoring.db.

CREATE TABLE IF NOT EXISTS conversations (
    id              TEXT PRIMARY KEY,
    ts              TIMESTAMPTZ NOT NULL DEFAULT now(),
    question        TEXT NOT NULL,
    rewritten_query TEXT NOT NULL,
    answer          TEXT NOT NULL,
    model           TEXT NOT NULL,
    prompt_version  TEXT NOT NULL,
    retrieval_mode  TEXT NOT NULL,
    chunk_ids       JSONB NOT NULL,
    tokens_in       INTEGER NOT NULL,
    tokens_out      INTEGER NOT NULL,
    cost_usd        DOUBLE PRECISION NOT NULL,
    latency_ms      INTEGER NOT NULL,
    judge_relevance TEXT,
    citations_grounded   BOOLEAN,
    ungrounded_citations JSONB,
    -- 'live' = a real /ask call, 'eval' = seeded by evaluation/eval_rag.py.
    -- The usage/cost/latency dashboards filter to 'live' so the evaluation's
    -- hundreds of synthetic conversations don't read as user traffic.
    source               TEXT NOT NULL DEFAULT 'live',
    -- The answer hit the output-token cap and is cut off mid-sentence.
    answer_truncated     BOOLEAN
);

CREATE TABLE IF NOT EXISTS feedback (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    thumbs          TEXT NOT NULL,
    comment         TEXT,
    ts              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_conversations_ts ON conversations (ts);
-- Every usage panel filters on source, so index it alongside ts.
CREATE INDEX IF NOT EXISTS idx_conversations_source_ts ON conversations (source, ts);
CREATE INDEX IF NOT EXISTS idx_feedback_conversation ON feedback (conversation_id);
