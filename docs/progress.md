# HealthAssist Development Progress

## Current Phase

Local RAG MVP implemented; real-model demo validation is blocked by model-registry network access.
See `README.md` for usage and `docs/mvp-validation.md` for actual results.
The sections below record the earlier foundation phase.

## Completed / Reviewed

- Project structure
- PDF extraction design
- Chunking implementation
- Embedding abstraction
- Ollama embedding implementation
- PostgreSQL setup
- pgvector extension
- documents table
- document_chunks table
- ollama_embeddings table
- Database connection layer
- Vector persistence layer
- Naming cleanup for document_chunk_id
- Reviewed responsibilities of schema.sql
- Reviewed responsibilities of connection.py
- Reviewed responsibilities of vector_store.py

## Current Validation

Before moving forward we are validating:

- schema.sql
- connection.py
- vector_store.py
- ingestion flow
- embedding persistence

## Next Stage

Semantic Retrieval

Planned flow:

User Question
    ↓
Generate Query Embedding
    ↓
pgvector Similarity Search
    ↓
Top-K Chunks
    ↓
Return Text + Source + Page + Similarity

## After Semantic Retrieval

- Hybrid retrieval
- Prompt construction
- LLM generation
- Citation formatting
- LangGraph workflow
- Guardrails
- Human-in-the-loop
- Evaluation
- API layer
- Containerized deployment