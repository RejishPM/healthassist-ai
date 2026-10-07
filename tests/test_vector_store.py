import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql://user:password@localhost/healthassist",
)

from app.db import vector_store


class AsyncContext:
    def __init__(self, value=None):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class FakeCursor:
    def __init__(self, row=None):
        self.row = row

    async def fetchone(self):
        return self.row


class FakeConnection:
    def __init__(self):
        self.calls = []
        self.next_chunk_id = 100

    def transaction(self):
        return AsyncContext()

    async def execute(self, query, parameters):
        normalized_query = " ".join(query.split())
        self.calls.append((normalized_query, parameters))

        if "INSERT INTO documents" in normalized_query:
            return FakeCursor((42,))

        if "INSERT INTO document_chunks" in normalized_query:
            self.next_chunk_id += 1
            return FakeCursor((self.next_chunk_id,))

        return FakeCursor()


class FakePool:
    def __init__(self, connection):
        self.connection_value = connection

    def connection(self):
        return AsyncContext(self.connection_value)


class FakeEmbedder:
    model = "nomic-embed-text"

    def __init__(self):
        self.texts = []

    def get_embedding(self, text):
        self.texts.append(text)
        return [float(len(self.texts))] * 768


class PersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_persists_document_chunks_and_ollama_embeddings(self):
        chunks = [
            {
                "chunk_id": "report.pdf_p1_c0",
                "text": "first chunk",
                "page": 1,
                "section": "Introduction",
                "content_type": "prose",
                "safety_critical": False,
                "word_count": 2,
                "source": "report.pdf",
                "doc_title": "Report",
                "chunk_index": 0,
            },
            {
                "chunk_id": "report.pdf_p2_c1",
                "text": "second chunk",
                "page": 2,
                "section": "Care",
                "content_type": "prose",
                "safety_critical": True,
                "word_count": 2,
                "source": "report.pdf",
                "doc_title": "Report",
                "chunk_index": 1,
            },
        ]
        connection = FakeConnection()
        embedder = FakeEmbedder()

        with (
            patch.object(
                vector_store,
                "pool",
                FakePool(connection),
            ),
            patch.object(
                vector_store,
                "register_vector_async",
                new=AsyncMock(),
            ) as register_vector,
        ):
            document_id = await vector_store.persist_pdf_chunks(
                file_name="report.pdf",
                title="Report",
                chunks=chunks,
                embedder=embedder,
            )

        self.assertEqual(document_id, 42)
        self.assertEqual(embedder.texts, ["first chunk", "second chunk"])
        register_vector.assert_awaited_once_with(connection)

        document_calls = [
            call
            for call in connection.calls
            if "INSERT INTO documents" in call[0]
        ]
        chunk_calls = [
            call
            for call in connection.calls
            if "INSERT INTO document_chunks" in call[0]
        ]
        embedding_calls = [
            call
            for call in connection.calls
            if "INSERT INTO ollama_embeddings" in call[0]
        ]

        self.assertEqual(len(document_calls), 1)
        self.assertEqual(len(chunk_calls), 2)
        self.assertEqual(len(embedding_calls), 2)
        self.assertEqual(document_calls[0][1], ("report.pdf", "Report", None))
        self.assertEqual(
            [call[1][0] for call in embedding_calls],
            [101, 102],
        )
        self.assertTrue(
            all(
                call[1][1] == "nomic-embed-text"
                for call in embedding_calls
            )
        )

    async def test_embedding_count_must_match_chunk_count(self):
        with self.assertRaisesRegex(
            ValueError,
            "Each chunk must have exactly one embedding",
        ):
            await vector_store.insert_ollama_embeddings(
                FakeConnection(),
                chunk_ids=[1],
                embeddings=[],
                model="nomic-embed-text",
            )


if __name__ == "__main__":
    unittest.main()
