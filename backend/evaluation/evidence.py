"""Evidence gathering module for Vishwas AI."""

import re
from typing import Any

from backend.search import web_search


def _derive_search_query(claim: str) -> str:
    """Derive a clean and focused search query from an atomic factual claim."""
    query = claim.strip()
    # Remove wrapping quotation marks
    query = re.sub(r'^["\']|["\']$', '', query)
    # Remove trailing periods and punctuation
    query = query.rstrip(".?!; ")
    # Limit query length if excessively verbose
    if len(query) > 200:
        query = query[:200].rsplit(" ", 1)[0]
    return query.strip()


def gather_evidence(
    claim: str,
    max_results: int = 5,
    cost_tracker: dict[str, int] | None = None,
    trace: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Gather web search evidence for a factual claim.

    Args:
        claim: The factual claim string.
        max_results: Maximum search results to retrieve (default: 5).
        cost_tracker: Optional dictionary to increment 'search_calls'.
        trace: Optional dictionary to capture intermediate inspection data.

    Returns:
        List of evidence dicts with 'title', 'url', and 'snippet'.

    Raises:
        RuntimeError: If search operation encounters an error.
    """
    if not claim or not claim.strip():
        return []

    query = _derive_search_query(claim)

    if trace is not None:
        trace["search_query"] = query

    if cost_tracker is not None:
        cost_tracker["search_calls"] = cost_tracker.get("search_calls", 0) + 1

    max_retries = 3
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            results = web_search(query=query, max_results=max_results)
            if trace is not None:
                trace["results"] = results
            return results
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                import time
                time.sleep(1.0 * (attempt + 1))
                continue
            raise RuntimeError(
                f"Web search error while gathering evidence for '{query}': {last_exc}"
            ) from last_exc
