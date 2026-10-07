# Database Design

## Overview

HealthAssist currently uses PostgreSQL with the pgvector extension.

The initial schema contains three main tables:

documents
    |
    | 1-to-many
    v
document_chunks
    |
    | referenced by
    v
ollama_embeddings

## documents

Represents one source document.

Important fields:

- id
- file_name
- title
- version
- created_at

The document table allows HealthAssist to track document provenance and versions.

## document_chunks

Represents smaller pieces extracted from a document.

Important fields:

- id
- document_id
- chunk_id
- text
- page
- section
- content_type
- safety_critical
- word_count
- metadata

### document_chunks.id

Internal PostgreSQL primary key.

Example:

101

### document_chunks.chunk_id

Application-level logical chunk identifier.

Example:

living_diabetes_p12_c03

These two identifiers have different purposes and must not be confused.

## ollama_embeddings

Stores vector representations of document chunks.

Important fields:

- id
- document_chunk_id
- provider
- model
- embedding
- created_at

### document_chunk_id

Foreign key referencing:

document_chunks.id

Example relationship:

document_chunks.id = 101

ollama_embeddings.document_chunk_id = 101

## Embedding Dimension

Current Ollama embedding model produces vectors with dimension:

768

Therefore:

embedding VECTOR(768)

## Cascading Deletes

Deleting a document removes its chunks.

Deleting the chunks removes their embeddings.

Relationship:

documents
    ↓
document_chunks
    ↓
ollama_embeddings

This prevents orphaned data.

## Metadata

Flexible metadata is stored using PostgreSQL JSONB.

Potential metadata includes:

- source
- document title
- chunk index
- country
- hospital
- department
- guideline type

A GIN index is used to support efficient metadata filtering.