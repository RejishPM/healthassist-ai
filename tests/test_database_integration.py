"""Opt-in real PostgreSQL/pgvector tests; embeddings stay local and deterministic."""
import os
from pathlib import Path
import unittest
from uuid import uuid4
from unittest.mock import patch

import psycopg

from tests.test_vector_store import AsyncContext, FakeEmbedder
from app.db import vector_store


DATABASE_URL = os.getenv("HEALTHASSIST_TEST_DATABASE_URL")
SCHEMA_SQL = (Path(__file__).resolve().parents[1] / "app/db/schema.sql").read_text()


class ConnectionPool:
    def __init__(self, connection):
        self.value = connection

    def connection(self):
        return AsyncContext(self.value)


@unittest.skipUnless(DATABASE_URL, "HEALTHASSIST_TEST_DATABASE_URL not configured")
class DatabaseIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.connection = await psycopg.AsyncConnection.connect(DATABASE_URL)
        self.addAsyncCleanup(self.connection.close)
        # Extension creation and every schema/write are rolled back on close.
        await self.connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        schema = "healthassist_test_" + uuid4().hex
        await self.connection.execute(
            psycopg.sql.SQL("CREATE SCHEMA {}").format(psycopg.sql.Identifier(schema))
        )
        await self.connection.execute(
            psycopg.sql.SQL("SET LOCAL search_path TO {}, public").format(psycopg.sql.Identifier(schema))
        )

    async def active_contract(self):
        cursor = await self.connection.execute(
            "SELECT is_nullable, column_default FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name = 'documents' "
            "AND column_name = 'is_active'"
        )
        self.assertEqual(await cursor.fetchone(), ("NO", "true"))

    async def test_fresh_schema_persistence_retrieval_and_inactive_filter(self):
        await self.connection.execute(SCHEMA_SQL)
        await self.active_contract()
        embedder = FakeEmbedder()
        with patch.object(vector_store, "pool", ConnectionPool(self.connection)):
            document_id = await vector_store.persist_pdf_chunks(
                file_name="report.pdf", title="Report", version="v1",
                chunks=[{"chunk_id": "report_p1", "text": "care", "page": 1,
                         "metadata": {"source": "report.pdf"}}], embedder=embedder,
            )
            results = await vector_store.search_similar_chunks(
                query="care", embedder=embedder, min_similarity=0.9,
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].document_id, document_id)
            self.assertEqual(results[0].metadata, {"source": "report.pdf"})
            self.assertAlmostEqual(results[0].similarity, 1.0, places=5)
            await self.connection.execute(
                "UPDATE documents SET is_active = FALSE WHERE id = %s", (document_id,)
            )
            self.assertEqual(await vector_store.search_similar_chunks(
                query="care", embedder=embedder), [])

    async def test_duplicate_chunk_rolls_back_document_and_embeddings(self):
        await self.connection.execute(SCHEMA_SQL)
        with patch.object(vector_store, "pool", ConnectionPool(self.connection)):
            with self.assertRaises(psycopg.errors.UniqueViolation):
                await vector_store.persist_pdf_chunks(
                    file_name="duplicate.pdf", title=None,
                    chunks=[{"chunk_id": "duplicate", "text": "one"},
                            {"chunk_id": "duplicate", "text": "two"}],
                    embedder=FakeEmbedder(),
                )
        for table in ("documents", "document_chunks", "ollama_embeddings"):
            cursor = await self.connection.execute(
                psycopg.sql.SQL("SELECT COUNT(*) FROM {}").format(psycopg.sql.Identifier(table))
            )
            self.assertEqual(await cursor.fetchone(), (0,))

    async def test_pdf_api_to_database_to_cited_answer(self):
        import json
        from unittest.mock import Mock
        import httpx
        from app.main import create_app
        from app.settings import Settings
        from tests.test_rag import sample_pdf

        await self.connection.execute(SCHEMA_SQL)
        app = create_app(Settings())

        def ollama_response(url, **kwargs):
            response = Mock()
            if url.endswith("/api/embeddings"):
                # Only the known-answer query shares the document vector.
                if "unknown planet" in kwargs["json"]["prompt"]:
                    vector = [0.0, 1.0] + [0.0] * 766
                else:
                    vector = [1.0] + [0.0] * 767
                response.json.return_value = {"embedding": vector}
            else:
                payload = json.loads(kwargs["json"]["messages"][1]["content"])
                response.json.return_value = {
                    "message": {
                        "content": json.dumps(
                            {
                                "quotes": [
                                    {
                                        "source_id": 1,
                                        "quote": payload["sources"][0]["text"],
                                    }
                                ]
                            }
                        )
                    }
                }
            return response

        with patch.object(vector_store, "pool", ConnectionPool(self.connection)), patch(
            "requests.post", side_effect=ollama_response
        ) as provider:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                ids = []
                for _ in range(2):
                    upload = await client.post(
                        "/documents?file_name=guide.pdf&version=v1",
                        content=sample_pdf(),
                        headers={"content-type": "application/pdf"},
                    )
                    self.assertEqual(upload.status_code, 201, upload.text)
                    ids.append(upload.json()["document_id"])
                self.assertNotEqual(*ids)
                answer = await client.post(
                    "/questions", json={"question": "When does the Cedar clinic open?"}
                )
                self.assertEqual(answer.status_code, 200, answer.text)
                body = answer.json()
                self.assertEqual(body["status"], "answered")
                self.assertEqual(body["citations"][0]["file_name"], "guide.pdf")
                self.assertEqual(body["citations"][0]["page"], 1)
                self.assertIn(body["citations"][0]["document_id"], ids)
                self.assertIn("nine", body["answer"])
                previous_chat_calls = len(
                    [
                        call
                        for call in provider.call_args_list
                        if call.args[0].endswith("/api/chat")
                    ]
                )
                absent = await client.post(
                    "/questions", json={"question": "Describe an unknown planet"}
                )
                self.assertEqual(absent.json()["status"], "no_relevant_context")
                self.assertEqual(
                    previous_chat_calls,
                    len(
                        [
                            call
                            for call in provider.call_args_list
                            if call.args[0].endswith("/api/chat")
                        ]
                    ),
                )
                await self.connection.execute("UPDATE documents SET is_active = FALSE")
                inactive = await client.post(
                    "/questions", json={"question": "When does the Cedar clinic open?"}
                )
                self.assertEqual(inactive.json()["citations"], [])
                self.assertEqual(inactive.json()["status"], "no_relevant_context")

    async def test_upgrade_preserves_documents_and_is_idempotent(self):
        await self.connection.execute(
            "CREATE TABLE documents (id BIGSERIAL PRIMARY KEY, file_name TEXT NOT NULL, "
            "title TEXT, version TEXT, created_at TIMESTAMPTZ DEFAULT NOW())"
        )
        await self.connection.execute("INSERT INTO documents (file_name) VALUES ('legacy.pdf')")
        await self.connection.execute(SCHEMA_SQL)
        await self.connection.execute(SCHEMA_SQL)
        await self.active_contract()
        cursor = await self.connection.execute("SELECT file_name, is_active FROM documents")
        self.assertEqual(await cursor.fetchall(), [("legacy.pdf", True)])
        with self.assertRaises(psycopg.errors.NotNullViolation):
            async with self.connection.transaction():
                await self.connection.execute("UPDATE documents SET is_active = NULL")
