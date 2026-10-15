"""Small, validated configuration for the local-only RAG API."""

import os
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    ollama_base_url: str = "http://127.0.0.1:11434"
    embedding_model: str = "nomic-embed-text"
    chat_model: str = "qwen2.5:0.5b"
    top_k: int = Field(default=5, ge=1, le=10)
    min_similarity: float = Field(default=0.5, ge=-1, le=1, allow_inf_nan=False)
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    max_pdf_pages: int = Field(default=100, ge=1)
    max_chunks: int = Field(default=200, ge=1)

    @field_validator("ollama_base_url")
    @classmethod
    def local_ollama_only(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme != "http"
            or parsed.hostname
            not in {"localhost", "127.0.0.1", "::1", "ollama", "host.docker.internal"}
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("OLLAMA_BASE_URL must be a local HTTP Ollama service")
        return value.rstrip("/")

    @field_validator("embedding_model", "chat_model")
    @classmethod
    def no_cloud_models(cls, value: str) -> str:
        value = value.strip()
        if not value or value.lower().endswith(":cloud"):
            raise ValueError("Use a downloaded local model, not a cloud model")
        return value

    @classmethod
    def from_env(cls):
        # This API deliberately never selects the paid Bedrock provider.
        if os.getenv("EMBEDDING_PROVIDER", "ollama").lower() != "ollama":
            raise ValueError("The MVP API supports only local Ollama embeddings")
        names = {
            "ollama_base_url": "OLLAMA_BASE_URL",
            "embedding_model": "OLLAMA_EMBEDDING_MODEL",
            "chat_model": "OLLAMA_CHAT_MODEL",
            "top_k": "RETRIEVAL_TOP_K",
            "min_similarity": "RETRIEVAL_SIMILARITY_THRESHOLD",
            "max_upload_bytes": "MAX_UPLOAD_BYTES",
            "max_pdf_pages": "MAX_PDF_PAGES",
            "max_chunks": "MAX_DOCUMENT_CHUNKS",
        }
        return cls(
            **{
                field: os.environ[name]
                for field, name in names.items()
                if name in os.environ
            }
        )
