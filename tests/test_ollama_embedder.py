import unittest
from unittest.mock import patch

import requests

from app.embedding.ollama_embedder import OllamaEmbedder


class OllamaEmbedderTests(unittest.TestCase):
    @patch("app.embedding.ollama_embedder.requests.post")
    def test_embedding_request_and_response(self, post):
        post.return_value.json.return_value = {"embedding": [0.1, 0.2]}
        embedder = OllamaEmbedder()
        self.assertEqual(embedder.get_embedding("care"), [0.1, 0.2])
        post.assert_called_once_with(
            "http://localhost:11434/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": "care"},
            timeout=60,
        )
        post.return_value.raise_for_status.assert_called_once_with()

    @patch("app.embedding.ollama_embedder.requests.post")
    def test_http_failure_propagates(self, post):
        post.return_value.raise_for_status.side_effect = requests.HTTPError(
            "unavailable"
        )
        with self.assertRaises(requests.HTTPError):
            OllamaEmbedder().get_embedding("care")
        post.return_value.json.assert_not_called()

    @patch("app.embedding.ollama_embedder.requests.post", side_effect=requests.Timeout)
    def test_timeout_propagates(self, post):
        with self.assertRaises(requests.Timeout):
            OllamaEmbedder().get_embedding("care")


class MalformedEmbeddingTests(unittest.TestCase):
    @patch("app.embedding.ollama_embedder.requests.post")
    def test_malformed_embedding_envelopes_fail_cleanly(self, post):
        for data in ([], {}, None):
            post.return_value.json.return_value = data
            with self.subTest(data=data), self.assertRaises(ValueError):
                OllamaEmbedder().get_embedding("care")
