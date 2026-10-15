# HealthAssist Architecture

## High-Level Architecture

Clinical Documents
        |
        v
PDF Extractor
        |
        v
Chunker
        |
        v
Embedding Provider
   /            \
Ollama        Bedrock
   \            /
        |
        v
PostgreSQL + pgvector
        |
        v
Retrieval Layer
        |
        v
LLM Generation
        |
        v
LangGraph Workflow
        |
        v
Guardrails / HITL
        |
        v
Answer + Citations

## Current Implementation Layers

### Ingestion Layer

Location:

app/ingestion/

Responsibilities:

- Extract text from PDFs
- Preserve useful metadata
- Split documents into retrievable chunks

### Embedding Layer

Location:

app/embedding/

Responsibilities:

- Convert text into vector embeddings
- Abstract embedding providers
- Support Ollama for development
- Support Bedrock for cloud environments

### Database Layer

Location:

app/db/

Responsibilities:

- Define database schema
- Manage PostgreSQL connections
- Persist documents
- Persist chunks
- Persist embeddings

### Retrieval Layer

Status: Implemented for the local RAG MVP (see `README.md`).

Responsibilities will include:

- Generate embedding for user query
- Perform vector similarity search
- Retrieve Top-K document chunks
- Return source and page metadata
- Later support hybrid semantic + keyword retrieval

## RAG MVP API

`app/main.py` manages the PostgreSQL pool and validates HTTP inputs.
`POST /documents` calls `app/rag.py`, which extracts/chunks uploaded PDFs and uses
the existing transactional vector store. `POST /questions` validates the question,
retrieves compatible active-document vectors, and calls local Ollama through
`app/generation/ollama.py`. The model selects quotations; the server verifies
them and builds citations from retrieved metadata. No context bypasses generation.
This MVP does not require LangGraph, paid providers, or a new storage abstraction.
