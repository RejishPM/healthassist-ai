CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    file_name TEXT NOT NULL,
    title TEXT,
    version TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_id TEXT NOT NULL UNIQUE,
    text TEXT NOT NULL,
    page INTEGER,
    section TEXT,
    content_type TEXT,
    safety_critical BOOLEAN DEFAULT FALSE,
    word_count INTEGER,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ollama_embeddings (
    id BIGSERIAL PRIMARY KEY,
    document_chunk_id BIGINT NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
    provider TEXT NOT NULL DEFAULT 'ollama',
    model TEXT NOT NULL,
    embedding VECTOR(768) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (document_chunk_id, provider, model)
);

CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id
    ON document_chunks(document_id);

CREATE INDEX IF NOT EXISTS idx_document_chunks_metadata
    ON document_chunks USING GIN(metadata);