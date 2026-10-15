"""Application flow composed from the existing ingestion and vector-store modules."""

import asyncio
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from uuid import uuid4
from typing import Literal

import pymupdf
from pydantic import BaseModel

from app.db import vector_store
from app.embedding.ollama_embedder import OllamaEmbedder
from app.generation.ollama import OllamaGenerator
from app.ingestion.chunker import chunk_pdf
from app.query.query_validator import validate_query
from app.settings import Settings


class InvalidDocument(ValueError):
    pass


class InvalidGeneration(ValueError):
    pass


class Citation(BaseModel):
    source_id: int
    document_id: int
    chunk_id: str
    file_name: str
    title: str | None
    version: str | None
    page: int | None
    section: str | None
    safety_critical: bool
    similarity: float
    quote: str


class Answer(BaseModel):
    status: Literal["answered", "no_relevant_context", "insufficient_evidence"]
    answer: str
    citations: list[Citation]
    retrieved_count: int


def extract_chunks(data: bytes, file_name: str, settings: Settings) -> list[dict]:
    if (
        not file_name
        or file_name != Path(file_name).name
        or "\\" in file_name
        or any(ord(char) < 32 for char in file_name)
    ):
        raise InvalidDocument("Invalid PDF basename")
    if not data.startswith(b"%PDF-"):
        raise InvalidDocument("Expected a PDF document")
    with TemporaryDirectory(prefix="healthassist-") as directory:
        path = Path(directory) / file_name
        path.write_bytes(data)
        try:
            with pymupdf.open(path) as document:
                if document.needs_pass:
                    raise InvalidDocument("Encrypted PDFs are not supported")
                if len(document) > settings.max_pdf_pages:
                    raise InvalidDocument("PDF exceeds the page limit")
            chunks = chunk_pdf(str(path))
        except (pymupdf.FileDataError, RuntimeError) as exc:
            raise InvalidDocument("Unable to read PDF") from exc
    if not chunks:
        raise InvalidDocument("PDF contains no usable text; scanned PDFs require OCR")
    if len(chunks) > settings.max_chunks:
        raise InvalidDocument("PDF exceeds the chunk limit")
    # Logical IDs from the chunker are filename based. Namespace each upload so
    # two uploads of the same filename cannot collide; preserve the actual source.
    upload_id = uuid4().hex
    for chunk in chunks:
        chunk["chunk_id"] = f"{upload_id}_{chunk['chunk_id']}"
    return chunks


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


class RAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedder = OllamaEmbedder(
            model=settings.embedding_model, base_url=settings.ollama_base_url
        )
        self.generator = OllamaGenerator(
            model=settings.chat_model, base_url=settings.ollama_base_url
        )

    async def ingest(
        self, data: bytes, file_name: str, title: str | None, version: str | None
    ) -> dict:
        chunks = await asyncio.to_thread(extract_chunks, data, file_name, self.settings)
        document_id = await vector_store.persist_pdf_chunks(
            file_name=file_name,
            title=title or chunks[0].get("doc_title"),
            version=version,
            chunks=chunks,
            embedder=self.embedder,
        )
        return {
            "document_id": document_id,
            "file_name": file_name,
            "chunk_count": len(chunks),
        }

    async def ask(self, question: str) -> Answer:
        question = validate_query(question)
        results = await vector_store.search_similar_chunks(
            query=question,
            embedder=self.embedder,
            top_k=self.settings.top_k,
            min_similarity=self.settings.min_similarity,
        )
        if not results:
            return Answer(
                status="no_relevant_context",
                answer="I do not have sufficient source information to answer this question.",
                citations=[],
                retrieved_count=0,
            )
        # Bound the prompt. Citation checking must use precisely the text sent.
        per_source_limit = min(4000, 12000 // len(results))
        sources = [
            {"source_id": index, "text": result.text[:per_source_limit]}
            for index, result in enumerate(results, 1)
        ]
        selection = await asyncio.to_thread(self.generator.select, question, sources)
        if not selection.quotes:
            return Answer(
                status="insufficient_evidence",
                answer="The retrieved sources do not provide sufficient evidence to answer this question.",
                citations=[],
                retrieved_count=len(results),
            )
        citations = []
        for item in selection.quotes:
            if item.source_id > len(results):
                raise InvalidGeneration("Unknown citation")
            quote = normalize_whitespace(item.quote)
            if len(quote) < 15 or quote not in normalize_whitespace(
                sources[item.source_id - 1]["text"]
            ):
                raise InvalidGeneration("Quotation does not occur in cited source")
            result = results[item.source_id - 1]
            citations.append(
                Citation(
                    source_id=item.source_id,
                    document_id=result.document_id,
                    chunk_id=result.chunk_id,
                    file_name=result.metadata.get("source")
                    or result.document_title
                    or "Unknown source",
                    title=result.document_title,
                    version=result.document_version,
                    page=result.page,
                    section=result.section,
                    safety_critical=result.safety_critical,
                    similarity=result.similarity,
                    quote=quote,
                )
            )
        return Answer(
            status="answered",
            answer="\n\n".join(
                f"“{citation.quote}” [{citation.source_id}]" for citation in citations
            ),
            citations=citations,
            retrieved_count=len(results),
        )
