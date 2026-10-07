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

Status: Next implementation stage

Responsibilities will include:

- Generate embedding for user query
- Perform vector similarity search
- Retrieve Top-K document chunks
- Return source and page metadata
- Later support hybrid semantic + keyword retrieval