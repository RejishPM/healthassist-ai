import os

from dotenv import load_dotenv

from .embedder import Embedder
from .ollama_embedder import OllamaEmbedder
from .bedrock_embedder import BedrockEmbedder

load_dotenv()


def get_embedder() -> Embedder:
    provider = os.getenv(
        "EMBEDDING_PROVIDER",
        "ollama",
    ).lower()

    if provider == "ollama":
        return OllamaEmbedder(
            model=os.getenv(
                "OLLAMA_EMBEDDING_MODEL",
                "nomic-embed-text",
            ),
            base_url=os.getenv(
                "OLLAMA_BASE_URL",
                "http://localhost:11434",
            ),
        )

    if provider == "bedrock":
        return BedrockEmbedder(
            model_id=os.getenv(
                "BEDROCK_EMBEDDING_MODEL",
                "amazon.titan-embed-text-v2:0",
            ),
            region_name=os.getenv(
                "AWS_REGION",
                "us-east-1",
            ),
        )

    raise ValueError(
        f"Unsupported embedding provider: {provider}"
    )