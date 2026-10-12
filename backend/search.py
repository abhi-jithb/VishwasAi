"""Search abstraction module for Vishwas AI.

Exposes web_search using Tavily search API.
"""

from tavily import TavilyClient
from backend.config import SEARCH_API_KEY


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Execute a web search using Tavily.

    Args:
        query: Search query string.
        max_results: Maximum number of search results to retrieve (default: 5).

    Returns:
        List of result dictionaries each containing 'title', 'url', and 'snippet'.
    """
    client = TavilyClient(api_key=SEARCH_API_KEY)
    response = client.search(query=query, max_results=max_results)
    raw_results = response.get("results", [])

    results: list[dict] = []
    for item in raw_results:
        snippet = item.get("content") or item.get("snippet") or ""
        results.append(
            {
                "title": str(item.get("title", "")),
                "url": str(item.get("url", "")),
                "snippet": str(snippet),
            }
        )
    return results
