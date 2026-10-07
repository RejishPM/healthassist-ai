from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    text: str
    similarity: float
    distance: float

    document_id: int
    document_title: str | None
    document_version: str | None

    page: int | None
    section: str | None
    content_type: str | None
    safety_critical: bool

    metadata: dict[str, Any]
