# app/query/query_rewriter.py

def rewrite_query(
    current_query: str,
    previous_query: str | None = None,
) -> str:
    """
    Convert a conversational follow-up into a standalone retrieval query.

    First version:
    - If there is no previous query, return the current query unchanged.
    - If the current query looks like a short follow-up, combine it with
      the previous query.
    """

    current_query = current_query.strip()

    if not previous_query:
        return current_query

    previous_query = previous_query.strip()

    short_follow_up = len(current_query.split()) <= 5

    if short_follow_up:
        return f"{previous_query} Follow-up: {current_query}"

    return current_query