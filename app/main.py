"""Local development API. Run with uvicorn app.main:app --host 127.0.0.1."""

from contextlib import asynccontextmanager
import re

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from psycopg import Error as DatabaseError
from psycopg_pool import PoolTimeout
import requests

from app.db import connection as database
from app.query.query_validator import QueryValidationError
from app.rag import Answer, InvalidDocument, InvalidGeneration, RAGService
from app.settings import Settings


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=2000)


class IngestedDocument(BaseModel):
    document_id: int
    file_name: str
    chunk_count: int


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        try:
            await database.open_pool()
            await database.pool.wait(timeout=10)
            async with database.pool.connection() as connection:
                # Fail startup on a missing/unmigrated schema, without changing it.
                await connection.execute(
                    "SELECT d.is_active, dc.metadata, oe.embedding FROM documents d "
                    "JOIN document_chunks dc ON dc.document_id = d.id "
                    "JOIN ollama_embeddings oe ON oe.document_chunk_id = dc.id LIMIT 0"
                )
            yield
        finally:
            await database.close_pool()

    application = FastAPI(
        title="HealthAssist RAG MVP", version="0.1.0", lifespan=lifespan
    )
    application.state.rag = RAGService(settings)

    @application.get("/health")
    async def health():
        # Check the database on each request; this is not Ollama/model readiness.
        try:
            async with database.pool.connection() as connection:
                await connection.execute("SELECT 1")
        except (DatabaseError, PoolTimeout):
            raise HTTPException(503, "Database unavailable") from None
        return {"status": "ok"}

    @application.post(
        "/documents",
        status_code=201,
        response_model=IngestedDocument,
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/pdf": {
                        "schema": {"type": "string", "format": "binary"}
                    }
                },
            }
        },
    )
    async def ingest(
        request: Request,
        file_name: str = Query(default="document.pdf", min_length=5, max_length=200),
        title: str | None = Query(default=None, max_length=200),
        version: str | None = Query(default=None, max_length=80),
    ):
        """Upload raw PDF bytes (Content-Type: application/pdf), not a server path."""
        if not re.fullmatch(r"[^\x00-\x1f\x7f/\\]+\.pdf", file_name, re.IGNORECASE):
            raise HTTPException(
                422, "file_name must be a PDF basename without path separators"
            )
        if (
            request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            != "application/pdf"
        ):
            raise HTTPException(
                415, "Use Content-Type: application/pdf with raw PDF bytes"
            )
        data = bytearray()
        async for block in request.stream():
            if len(data) + len(block) > settings.max_upload_bytes:
                raise HTTPException(413, "PDF exceeds the upload limit")
            data.extend(block)
        try:
            return await request.app.state.rag.ingest(
                bytes(data), file_name, title, version
            )
        except InvalidDocument as exc:
            raise HTTPException(422, str(exc)) from None
        except (DatabaseError, PoolTimeout):
            raise HTTPException(
                503, "Database unavailable; ingestion was not completed"
            ) from None
        except (requests.RequestException, ValueError, KeyError):
            raise HTTPException(
                502, "Embedding provider failed; ingestion was not completed"
            ) from None

    @application.post("/questions", response_model=Answer)
    async def ask(payload: Question, request: Request):
        try:
            return await request.app.state.rag.ask(payload.question)
        except QueryValidationError as exc:
            raise HTTPException(422, str(exc)) from None
        except (DatabaseError, PoolTimeout):
            raise HTTPException(503, "Database unavailable") from None
        except (
            InvalidGeneration,
            ValidationError,
            requests.RequestException,
            ValueError,
            KeyError,
        ):
            # Never return an unchecked model response or provider error body.
            raise HTTPException(
                502, "Model response unavailable or failed evidence validation"
            ) from None

    return application


app = create_app()
