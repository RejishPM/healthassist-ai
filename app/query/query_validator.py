# app/query/query_validator.py

import os

DEFAULT_MAX_QUERY_LENGTH = 2000


class QueryValidationError(ValueError):
    """Raised when a user query is invalid."""


def get_max_query_length() -> int:
    value = os.getenv("RETRIEVAL_MAX_QUERY_LENGTH")

    if not value:
        return DEFAULT_MAX_QUERY_LENGTH

    try:
        max_length = int(value)
    except ValueError as exc:
        raise RuntimeError(
            "RETRIEVAL_MAX_QUERY_LENGTH must be an integer"
        ) from exc

    if max_length <= 0:
        raise RuntimeError(
            "RETRIEVAL_MAX_QUERY_LENGTH must be greater than zero"
        )

    return max_length


def validate_query(query: str) -> str:
    if query is None:
        raise QueryValidationError("Query cannot be None")

    normalized_query = query.strip()

    if not normalized_query:
        raise QueryValidationError("Query cannot be empty")

    max_length = get_max_query_length()

    if len(normalized_query) > max_length:
        raise QueryValidationError(
            f"Query exceeds maximum length of {max_length} characters"
        )

    return normalized_query