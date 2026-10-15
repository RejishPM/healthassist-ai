import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault(
    "DATABASE_URL", "postgresql://user:password@localhost/healthassist"
)

from fastapi.testclient import TestClient
import psycopg
import requests

from app.main import create_app
from app.rag import InvalidDocument, InvalidGeneration
from app.settings import Settings


class APITests(unittest.TestCase):
    def setUp(self):
        self.app = create_app(Settings(max_upload_bytes=20))
        self.app.state.rag = AsyncMock()
        # No context manager: routes are unit-tested without opening a real pool.
        self.client = TestClient(self.app)
        self.addCleanup(self.client.close)

    def test_pdf_upload(self):
        self.app.state.rag.ingest.return_value = {
            "document_id": 42,
            "file_name": "guide.pdf",
            "chunk_count": 2,
        }
        response = self.client.post(
            "/documents?file_name=guide.pdf&title=Guide&version=v1",
            content=b"%PDF-test",
            headers={"content-type": "application/pdf"},
        )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["chunk_count"], 2)
        self.app.state.rag.ingest.assert_awaited_once_with(
            b"%PDF-test", "guide.pdf", "Guide", "v1"
        )

    def test_invalid_uploads_do_not_call_service(self):
        for name, content, media, expected in [
            ("guide.pdf", b"abc", "text/plain", 415),
            ("../guide.pdf", b"abc", "application/pdf", 422),
            ("..\\guide.pdf", b"abc", "application/pdf", 422),
            ("guide.txt", b"abc", "application/pdf", 422),
            ("guide.pdf", b"x" * 21, "application/pdf", 413),
        ]:
            with self.subTest(name=name, expected=expected):
                response = self.client.post(
                    "/documents",
                    params={"file_name": name},
                    content=content,
                    headers={"content-type": media},
                )
                self.assertEqual(response.status_code, expected)
        self.app.state.rag.ingest.assert_not_called()

    def test_bad_pdf_and_provider_failures(self):
        for error, expected in [
            (InvalidDocument("No usable text"), 422),
            (requests.Timeout("sensitive provider info"), 502),
            (psycopg.OperationalError("private connection"), 503),
        ]:
            with self.subTest(error=error):
                self.app.state.rag.ingest.side_effect = error
                response = self.client.post(
                    "/documents",
                    content=b"%PDF-test",
                    headers={"content-type": "application/pdf"},
                )
                self.assertEqual(response.status_code, expected)
                self.assertNotIn("sensitive", response.text)
                self.assertNotIn("private", response.text)

    def test_answer_contract(self):
        self.app.state.rag.ask.return_value = {
            "status": "no_relevant_context",
            "answer": "No sources",
            "citations": [],
            "retrieved_count": 0,
        }
        response = self.client.post(
            "/questions", json={"question": "When does it open?"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "no_relevant_context")

    def test_invalid_question_shape(self):
        for payload in [
            {},
            {"question": ""},
            {"question": "x" * 2001},
            {"question": "care", "system_prompt": "override"},
        ]:
            with self.subTest(payload=str(payload)[:40]):
                self.assertEqual(
                    self.client.post("/questions", json=payload).status_code, 422
                )
        self.app.state.rag.ask.assert_not_called()

    def test_answer_failures_are_not_no_context(self):
        for error, expected in [
            (InvalidGeneration("forged quote"), 502),
            (requests.ConnectionError("private URL"), 502),
            (psycopg.OperationalError("secret"), 503),
        ]:
            with self.subTest(error=error):
                self.app.state.rag.ask.side_effect = error
                response = self.client.post(
                    "/questions", json={"question": "What is known?"}
                )
                self.assertEqual(response.status_code, expected)
                self.assertNotIn("secret", response.text)
                self.assertNotIn("private", response.text)
                self.assertNotIn("forged", response.text)

    def test_pool_lifecycle_and_health(self):
        from app.main import database
        from tests.test_vector_store import FakePool

        connection = AsyncMock()
        pool = FakePool(connection)
        pool.wait = AsyncMock()
        with patch.object(database, "pool", pool), patch.object(
            database, "open_pool", new=AsyncMock()
        ) as opened, patch.object(database, "close_pool", new=AsyncMock()) as closed:
            with TestClient(self.app) as client:
                self.assertEqual(client.get("/health").json(), {"status": "ok"})
            opened.assert_awaited_once()
            closed.assert_awaited_once()
            pool.wait.assert_awaited_once_with(timeout=10)

    def test_startup_failure_closes_pool(self):
        from app.main import database

        pool = AsyncMock()
        pool.wait.side_effect = RuntimeError("database not ready")
        with patch.object(database, "pool", pool), patch.object(
            database, "open_pool", new=AsyncMock()
        ), patch.object(database, "close_pool", new=AsyncMock()) as closed:
            with self.assertRaisesRegex(RuntimeError, "not ready"):
                with TestClient(self.app):
                    pass
            closed.assert_awaited_once()
