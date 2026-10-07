# app/query/conversation_context.py

import re
from dataclasses import dataclass
from enum import Enum


class QueryDependency(str, Enum):
    STANDALONE = "standalone"
    FOLLOW_UP = "follow_up"
    AMBIGUOUS = "ambiguous"


@dataclass
class ContextDecision:
    dependency: QueryDependency
    candidate_history: list[str]
    reason: str


CONTEXT_REFERENCE_PATTERNS = [
    r"\b(it|this|that|these|those)\b",
    r"\b(the same|same one|same thing)\b",
    r"^(and|also|then)\b",
    r"^(what about|how about)\b",
    r"\b(again|instead|too)\b",
]


def has_context_reference(query: str) -> bool:
    """
    Detect linguistic references that usually depend on
    something mentioned earlier in the conversation.
    """

    normalized = query.strip().lower()

    return any(
        re.search(pattern, normalized)
        for pattern in CONTEXT_REFERENCE_PATTERNS
    )


def appears_complete(query: str) -> bool:
    """
    Conservative check for whether the query appears capable
    of standing alone for retrieval.

    This does NOT determine medical meaning.
    """

    normalized = query.strip()

    if not normalized:
        return False

    words = normalized.split()

    # One-word queries are usually ambiguous.
    if len(words) == 1:
        return False

    # Explicit contextual references indicate dependency.
    if has_context_reference(normalized):
        return False

    # A short query can still be perfectly standalone.
    #
    # Examples:
    #   "DKA symptoms?"
    #   "Normal HbA1c range?"
    #
    # Therefore we deliberately DO NOT classify based only
    # on number of words.

    return True


def select_recent_history(
    history: list[str],
    *,
    max_turns: int = 3,
) -> list[str]:
    """
    Select recent user queries as candidate conversational context.

    Selection here is intentionally bounded. More intelligent
    relevance selection can later use embeddings or an LLM.
    """

    cleaned = [
        item.strip()
        for item in history
        if item and item.strip()
    ]

    return cleaned[-max_turns:]


def classify_query_dependency(
    query: str,
    history: list[str] | None = None,
    *,
    max_history_turns: int = 3,
) -> ContextDecision:
    """
    Determine whether a query can be retrieved independently
    or probably requires conversational context.

    This is the deterministic fallback classifier.
    """

    normalized = query.strip()
    history = history or []

    candidate_history = select_recent_history(
        history,
        max_turns=max_history_turns,
    )

    if not candidate_history:
        return ContextDecision(
            dependency=QueryDependency.STANDALONE,
            candidate_history=[],
            reason="No previous conversation context is available.",
        )

    if has_context_reference(normalized):
        return ContextDecision(
            dependency=QueryDependency.FOLLOW_UP,
            candidate_history=candidate_history,
            reason="Query contains a contextual reference.",
        )

    if not appears_complete(normalized):
        return ContextDecision(
            dependency=QueryDependency.AMBIGUOUS,
            candidate_history=candidate_history,
            reason="Query does not appear sufficiently self-contained.",
        )

    return ContextDecision(
        dependency=QueryDependency.STANDALONE,
        candidate_history=[],
        reason="Query appears self-contained.",
    )