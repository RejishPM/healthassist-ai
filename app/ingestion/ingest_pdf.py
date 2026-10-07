import argparse
import asyncio
from pathlib import Path

from app.db.connection import close_pool, open_pool
from app.db.vector_store import persist_pdf_chunks
from app.embedding.ollama_embedder import OllamaEmbedder
from app.ingestion.chunker import chunk_pdf


async def ingest_pdf(pdf_path: Path) -> tuple[int, int]:
    chunks = await asyncio.to_thread(chunk_pdf, str(pdf_path))
    title = chunks[0].get("doc_title") if chunks else pdf_path.stem
    embedder = OllamaEmbedder()

    await open_pool()

    try:
        document_id = await persist_pdf_chunks(
            file_name=pdf_path.name,
            title=title,
            chunks=chunks,
            embedder=embedder,
        )
    finally:
        await close_pool()

    return document_id, len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest one local PDF into PostgreSQL with Ollama embeddings."
    )
    parser.add_argument("pdf_path", type=Path)
    args = parser.parse_args()

    document_id, chunk_count = asyncio.run(ingest_pdf(args.pdf_path))

    print(
        f"Ingested document {document_id} with {chunk_count} chunks."
    )


if __name__ == "__main__":
    main()
