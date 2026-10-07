import requests

from .embedder import Embedder


class OllamaEmbedder(Embedder):
    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url

    def get_embedding(self, text: str) -> list[float]:
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={
                "model": self.model,
                "prompt": text,
            },
            timeout=60,
        )

        response.raise_for_status()

        data = response.json()

        return data["embedding"]