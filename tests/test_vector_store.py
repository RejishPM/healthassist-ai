import os
import unittest
from unittest.mock import AsyncMock, Mock, patch

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

    async def fetchall(self):
        return self.row or []

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
                document_chunk_ids=[1],
                embeddings=[],
                model="nomic-embed-text",
            )


class RegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_embedding_failure_precedes_database_writes(self):
        embedder = FakeEmbedder()
        embedder.get_embedding = Mock(
            side_effect=RuntimeError("Ollama unavailable")
        )
        with patch.object(vector_store, "pool") as pool:
            with self.assertRaisesRegex(RuntimeError, "Ollama unavailable"):
                await vector_store.persist_pdf_chunks(
                    file_name="report.pdf", title=None,
                    chunks=[{"text": "care"}], embedder=embedder,
                )
        pool.connection.assert_not_called()

    async def test_chunk_metadata_preserved_without_mutating_input(self):
        connection = FakeConnection()
        chunk = {"chunk_id": "one", "text": "care", "source": "report.pdf",
                 "metadata": {"custom": "value"}}
        ids = await vector_store.insert_document_chunks(
            connection, document_id=42, chunks=[chunk],
        )
        self.assertEqual(ids, [101])
        self.assertEqual(connection.calls[0][1][-1].obj,
                         {"custom": "value", "source": "report.pdf"})
        self.assertEqual(chunk["metadata"], {"custom": "value"})

    async def test_missing_insert_id_fails(self):
        connection = AsyncMock()
        connection.execute.return_value.fetchone.return_value = None
        with self.assertRaisesRegex(RuntimeError, "Document insert"):
            await vector_store.insert_document(connection, file_name="report.pdf")
        with self.assertRaisesRegex(RuntimeError, "Chunk insert"):
            await vector_store.insert_document_chunks(
                connection, document_id=42, chunks=[{"chunk_id": "one", "text": "care"}],
            )

    async def test_retrieval_parameters_and_result_mapping(self):
        connection = AsyncMock()
        connection.execute.return_value.fetchall.return_value = [
            ("one", "care", 0.2, 0.8, 42, "Report", "v1", 2, "Care", "prose", True, None)
        ]
        embedder = FakeEmbedder()
        with patch.object(vector_store, "pool", FakePool(connection)), patch.object(
            vector_store, "register_vector_async", new=AsyncMock()
        ) as register:
            results = await vector_store.search_similar_chunks(
                query="  care  ", embedder=embedder, top_k=3, min_similarity=0.7,
            )
        register.assert_awaited_once_with(connection)
        self.assertEqual(embedder.texts, ["care"])
        query, params = connection.execute.call_args.args
        normalized = " ".join(query.split())
        self.assertIn("d.is_active = TRUE", normalized)
        self.assertIn("oe.provider = 'ollama'", normalized)
        self.assertIn("oe.model = %s", normalized)
        self.assertIn("ORDER BY oe.embedding <=> %s::vector LIMIT %s", normalized)
        self.assertEqual(params, ([1.0] * 768, [1.0] * 768, embedder.model,
                                  [1.0] * 768, 0.7, [1.0] * 768, 3))
        result = results[0]
        self.assertEqual((result.chunk_id, result.text, result.document_id), ("one", "care", 42))
        self.assertEqual((result.distance, result.similarity), (0.2, 0.8))
        self.assertEqual((result.document_title, result.document_version, result.page,
                          result.section, result.content_type, result.safety_critical,
                          result.metadata), ("Report", "v1", 2, "Care", "prose", True, {}))

    async def test_invalid_retrieval_does_not_call_services(self):
        with patch.object(vector_store, "pool") as pool:
            embedder = FakeEmbedder()
            for query, top_k in [("", 5), ("  ", 5), ("care", 0), ("care", -1)]:
                with self.subTest(query=query, top_k=top_k), self.assertRaises(ValueError):
                    await vector_store.search_similar_chunks(query=query, top_k=top_k, embedder=embedder)
            self.assertEqual(embedder.texts, [])
            pool.connection.assert_not_called()

    async def test_database_failure_exits_transaction_with_error(self):
        connection = FakeConnection()
        transaction = AsyncMock()
        connection.transaction = Mock(return_value=transaction)
        connection.execute = AsyncMock(side_effect=RuntimeError("database write failed"))
        with patch.object(vector_store, "pool", FakePool(connection)), patch.object(
            vector_store, "register_vector_async", new=AsyncMock()
        ):
            with self.assertRaisesRegex(RuntimeError, "database write failed"):
                await vector_store.persist_pdf_chunks(
                    file_name="report.pdf", title=None,
                    chunks=[{"text": "care"}], embedder=FakeEmbedder(),
                )
        transaction.__aenter__.assert_awaited_once()
        self.assertIs(transaction.__aexit__.call_args.args[0], RuntimeError)

    async def test_no_retrieval_matches(self):
        connection = AsyncMock()
        connection.execute.return_value.fetchall.return_value = []
        with patch.object(vector_store, "pool", FakePool(connection)), patch.object(
            vector_store, "register_vector_async", new=AsyncMock()
        ):
            self.assertEqual(await vector_store.search_similar_chunks(
                query="care", embedder=FakeEmbedder()), [])


if __name__ == "__main__":
    unittest.main()
