import os
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, Mock, patch

os.environ.setdefault(
    "DATABASE_URL", "postgresql://user:password@localhost/healthassist"
)

import pymupdf

from app.generation.ollama import EvidenceSelection, EvidenceQuote
from app.query.retrieval_result import RetrievalResult
from app.rag import InvalidDocument, InvalidGeneration, RAGService, extract_chunks
from app.settings import Settings


TEXT = "The fictional Cedar clinic opens at nine each morning and its appointment desk closes at five every weekday."


def sample_pdf(text=TEXT):
    with pymupdf.open() as document:
        page = document.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 750), text)
        return document.tobytes()


def source(text=TEXT):
    return RetrievalResult(
        chunk_id="c1",
        text=text,
        similarity=0.9,
        distance=0.1,
        document_id=42,
        document_title="Guide",
        document_version="v1",
        page=2,
        section="Hours",
        content_type="prose",
        safety_critical=False,
        metadata={"source": "guide.pdf"},
    )


class RAGTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.service = RAGService(Settings())
        self.service.generator = Mock()

    async def test_ingests_real_pdf_preserves_source_and_namespaces_ids(self):
        with patch(
            "app.rag.vector_store.persist_pdf_chunks", new=AsyncMock(return_value=42)
        ) as persist:
            for _ in range(2):
                result = await self.service.ingest(
                    sample_pdf(), "guide.pdf", "Guide", "v1"
                )
                self.assertEqual(result["document_id"], 42)
                self.assertGreater(result["chunk_count"], 0)
            first, second = [call.kwargs for call in persist.call_args_list]
            self.assertNotEqual(
                first["chunks"][0]["chunk_id"], second["chunks"][0]["chunk_id"]
            )
            self.assertEqual(first["chunks"][0]["source"], "guide.pdf")
            self.assertEqual(first["chunks"][0]["page"], 1)
            self.assertEqual(first["version"], "v1")

    async def test_no_matches_never_calls_llm(self):
        with patch(
            "app.rag.vector_store.search_similar_chunks", new=AsyncMock(return_value=[])
        ):
            result = await self.service.ask("When does the clinic open?")
        self.assertEqual(result.status, "no_relevant_context")
        self.assertEqual(result.citations, [])
        self.service.generator.select.assert_not_called()

    async def test_whitespace_question_rejected_before_retrieval(self):
        with patch(
            "app.rag.vector_store.search_similar_chunks", new=AsyncMock()
        ) as search:
            with self.assertRaises(ValueError):
                await self.service.ask("  ")
        search.assert_not_called()

    async def test_grounded_answer_citations_and_retrieval_settings(self):
        self.service.generator.select.return_value = EvidenceSelection(
            quotes=[EvidenceQuote(source_id=1, quote=TEXT)]
        )
        with patch(
            "app.rag.vector_store.search_similar_chunks",
            new=AsyncMock(return_value=[source()]),
        ) as search:
            result = await self.service.ask("  When does the clinic open?  ")
        self.assertEqual(result.status, "answered")
        self.assertEqual(result.answer, f"“{TEXT}” [1]")
        self.assertEqual(result.citations[0].file_name, "guide.pdf")
        self.assertEqual(result.citations[0].page, 2)
        self.assertEqual(result.citations[0].document_id, 42)
        self.assertEqual(search.call_args.kwargs["top_k"], 5)
        self.assertEqual(search.call_args.kwargs["min_similarity"], 0.5)
        self.assertEqual(search.call_args.kwargs["query"], "When does the clinic open?")

    async def test_llm_can_decline_unsupported_answer(self):
        self.service.generator.select.return_value = EvidenceSelection(quotes=[])
        with patch(
            "app.rag.vector_store.search_similar_chunks",
            new=AsyncMock(return_value=[source()]),
        ):
            result = await self.service.ask("What is the weather?")
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.citations, [])

    async def test_forged_citation_and_paraphrase_are_withheld(self):
        for quote in [
            EvidenceQuote(source_id=2, quote=TEXT),
            EvidenceQuote(source_id=1, quote="The clinic opens at midnight."),
            EvidenceQuote(source_id=1, quote=" " * 20),
        ]:
            with self.subTest(quote=quote):
                self.service.generator.select.return_value = EvidenceSelection(
                    quotes=[quote]
                )
                with patch(
                    "app.rag.vector_store.search_similar_chunks",
                    new=AsyncMock(return_value=[source()]),
                ):
                    with self.assertRaises(InvalidGeneration):
                        await self.service.ask("When does the clinic open?")

    async def test_model_cannot_cite_text_outside_sent_context(self):
        self.service.generator.select.return_value = EvidenceSelection(
            quotes=[
                EvidenceQuote(
                    source_id=1,
                    quote="This hidden sentence was never sent to the model.",
                )
            ]
        )
        with patch(
            "app.rag.vector_store.search_similar_chunks",
            new=AsyncMock(
                return_value=[
                    source(
                        "x" * 4001 + "This hidden sentence was never sent to the model."
                    )
                ]
            ),
        ):
            with self.assertRaises(InvalidGeneration):
                await self.service.ask("What is known?")


class DocumentTests(unittest.TestCase):
    def test_invalid_empty_and_image_only_pdf(self):
        for data in (b"", b"not a PDF", b"%PDF-broken", sample_pdf("")):
            with self.subTest(data=data[:20]), self.assertRaises(InvalidDocument):
                extract_chunks(data, "guide.pdf", Settings())

    def test_page_limit(self):
        with pymupdf.open() as document:
            document.new_page()
            document.new_page()
            data = document.tobytes()
        with self.assertRaisesRegex(InvalidDocument, "page limit"):
            extract_chunks(data, "guide.pdf", Settings(max_pdf_pages=1))

    def test_chunk_limit(self):
        with pymupdf.open() as document:
            for _ in range(2):
                page = document.new_page()
                page.insert_textbox(pymupdf.Rect(50, 50, 550, 750), TEXT)
            data = document.tobytes()
        with self.assertRaisesRegex(InvalidDocument, "chunk limit"):
            extract_chunks(data, "guide.pdf", Settings(max_chunks=1))

    def test_encrypted_pdf(self):
        with pymupdf.open(stream=sample_pdf(), filetype="pdf") as document:
            data = document.tobytes(
                encryption=pymupdf.PDF_ENCRYPT_AES_256,
                owner_pw="test-owner",
                user_pw="test-user",
            )
        with self.assertRaisesRegex(InvalidDocument, "Encrypted"):
            extract_chunks(data, "guide.pdf", Settings())

    def test_temporary_upload_is_removed(self):
        with patch(
            "app.rag.chunk_pdf", side_effect=RuntimeError("parser failure")
        ) as chunk:
            with self.assertRaises(InvalidDocument):
                extract_chunks(sample_pdf(), "guide.pdf", Settings())
        self.assertFalse(Path(chunk.call_args.args[0]).exists())
