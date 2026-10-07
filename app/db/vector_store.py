import asyncio
from typing import Any, Sequence

from pgvector.psycopg import register_vector_async
from psycopg.types.json import Jsonb

from app.embedding.ollama_embedder import OllamaEmbedder
from app.query.retrieval_result import RetrievalResult

from .connection import pool


async def insert_document(
    connection: Any,
    *,
    file_name: str,
    title: str | None = None,
    version: str | None = None,
) -> int:
    cursor = await connection.execute(
        """
        INSERT INTO documents (file_name, title, version)
        VALUES (%s, %s, %s)
        RETURNING id
        """,
        (file_name, title, version),
    )
    row = await cursor.fetchone()

    if row is None:
        raise RuntimeError("Document insert did not return an id")

    return row[0]


async def insert_document_chunks(
    connection: Any,
    *,
    document_id: int,
    chunks: list[dict],
) -> list[int]:
    document_chunk_ids: list[int] = []

    for chunk in chunks:
        metadata = dict(chunk.get("metadata") or {})

        for key in ("source", "doc_title", "chunk_index"):
            if key in chunk:
                metadata[key] = chunk[key]

        cursor = await connection.execute(
            """
            INSERT INTO document_chunks (
                document_id,
                chunk_id,
                text,
                page,
                section,
                content_type,
                safety_critical,
                word_count,
                metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                document_id,
                chunk["chunk_id"],
                chunk["text"],
                chunk.get("page"),
                chunk.get("section"),
                chunk.get("content_type"),
                chunk.get("safety_critical", False),
                chunk.get("word_count"),
                Jsonb(metadata),
            ),
        )
        row = await cursor.fetchone()

        if row is None:
            raise RuntimeError("Chunk insert did not return an id")

        document_chunk_ids.append(row[0])

    return document_chunk_ids


async def insert_ollama_embeddings(
    connection: Any,
    *,
    document_chunk_ids: list[int],
    embeddings: list[list[float]],
    model: str,
) -> None:
    if len(document_chunk_ids) != len(embeddings):
        raise ValueError("Each chunk must have exactly one embedding")

    for document_chunk_id, embedding in zip(document_chunk_ids, embeddings):
        await connection.execute(
            """
            INSERT INTO ollama_embeddings (
                document_chunk_id,
                provider,
                model,
                embedding
            )
            VALUES (%s, 'ollama', %s, %s)
            """,
            (document_chunk_id, model, embedding),
        )


async def persist_pdf_chunks(
    *,
    file_name: str,
    title: str | None,
    chunks: list[dict],
    embedder: OllamaEmbedder,
    version: str | None = None,
) -> int:
    embeddings = [
        await asyncio.to_thread(
            embedder.get_embedding,
            chunk["text"],
        )
        for chunk in chunks
    ]

    async with pool.connection() as connection:
        await register_vector_async(connection)

        async with connection.transaction():
            document_id = await insert_document(
                connection,
                file_name=file_name,
                title=title,
                version=version,
            )
            chunk_ids = await insert_document_chunks(
                connection,
                document_id=document_id,
                chunks=chunks,
            )
            await insert_ollama_embeddings(
                connection,
                document_chunk_ids=document_chunk_ids,
                embeddings=embeddings,
                model=embedder.model,
            )

    return document_id


async def search_similar_chunks(
    *,
    query: str,
    embedder: OllamaEmbedder,
    top_k: int = 5,
    min_similarity: float = 0.0,
) -> list[RetrievalResult]:
    if not query or not query.strip():
        raise ValueError("Query must not be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    query_embedding = await asyncio.to_thread(
        embedder.get_embedding,
        query.strip(),
    )

    async with pool.connection() as connection:
        await register_vector_async(connection)

        cursor = await connection.execute(
            """
            SELECT
                dc.chunk_id,
                dc.text,
                oe.embedding <=> %s::vector AS distance,
                1 - (oe.embedding <=> %s::vector) AS similarity,
                d.id AS document_id,
                d.title AS document_title,
                d.version AS document_version,
                dc.page,
                dc.section,
                dc.content_type,
                dc.safety_critical,
                dc.metadata
            FROM ollama_embeddings oe
            INNER JOIN document_chunks dc
                ON dc.id = oe.document_chunk_id
            INNER JOIN documents d
                ON d.id = dc.document_id
            WHERE
                d.is_active = TRUE
                AND oe.provider = 'ollama'
                AND oe.model = %s
                AND (1 - (oe.embedding <=> %s::vector)) >= %s
            ORDER BY oe.embedding <=> %s::vector
            LIMIT %s
            """,
            (
                query_embedding,
                query_embedding,
                embedder.model,
                query_embedding,
                min_similarity,
                query_embedding,
                top_k,
            ),
        )

        rows = await cursor.fetchall()

    return [
        RetrievalResult(
            chunk_id=row[0],
            text=row[1],
            distance=float(row[2]),
            similarity=float(row[3]),
            document_id=row[4],
            document_title=row[5],
            document_version=row[6],
            page=row[7],
            section=row[8],
            content_type=row[9],
            safety_critical=row[10],
            metadata=row[11] or {},
        )
        for row in rows
    ]
