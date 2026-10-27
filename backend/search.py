"""Search abstraction module for Vishwas AI.

Exposes web_search using Tavily search API.
"""

from tavily import TavilyClient
from backend.config import SEARCH_API_KEY
from backend.cache import get_search_cache, set_search_cache


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Execute a web search using Tavily, backed by SQLite disk cache.

    Args:
        query: Search query string.
        max_results: Maximum number of search results to retrieve (default: 5).

    Returns:
        List of result dictionaries each containing 'title', 'url', and 'snippet'.
    """
    clean_query = query.strip()
    if not clean_query:
        return []

    # 1. Check disk cache first
    cached_results = get_search_cache(query=clean_query, max_results=max_results)
    if cached_results is not None:
        return cached_results

    # 2. Query Search Provider (Tavily if key available, fallback to DuckDuckGo)
    results: list[dict] = []
    if SEARCH_API_KEY:
        try:
            client = TavilyClient(api_key=SEARCH_API_KEY)
            response = client.search(query=clean_query, max_results=max_results)
            raw_results = response.get("results", [])
            for item in raw_results:
                raw_text = str(item.get("snippet") or item.get("content") or "")
                snippet = (raw_text[:400] + "...") if len(raw_text) > 400 else raw_text
                results.append(
                    {
                        "title": str(item.get("title", "")),
                        "url": str(item.get("url", "")),
                        "snippet": snippet,
                    }
                )
        except Exception:
            results = []

    # Fallback to DuckDuckGo or Wikipedia if Tavily is unavailable or returned empty results
    if not results:
        try:
            from duckduckgo_search import DDGS

            ddgs = DDGS()
            ddg_results = list(ddgs.text(clean_query, max_results=max_results))
            for item in ddg_results:
                raw_text = str(item.get("body") or "")
                snippet = (raw_text[:400] + "...") if len(raw_text) > 400 else raw_text
                results.append(
                    {
                        "title": str(item.get("title", "")),
                        "url": str(item.get("href", "")),
                        "snippet": snippet,
                    }
                )
        except Exception:
            pass

    # Fallback to Wikipedia API for reliable factual encyclopedic evidence
    if not results:
        try:
            import html
            import json
            import re
            import urllib.parse
            import urllib.request

            encoded_q = urllib.parse.quote(clean_query)
            wiki_url = (
                f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded_q}"
                f"&utf8=&format=json&srlimit={max_results}"
            )
            req = urllib.request.Request(
                wiki_url,
                headers={"User-Agent": "VishwasAI-Audit/1.0 (academic research)"},
            )
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                search_items = data.get("query", {}).get("search", [])
                for item in search_items:
                    raw_snippet = html.unescape(item.get("snippet", ""))
                    clean_snippet = re.sub(r"<[^>]+>", "", raw_snippet).strip()
                    title = str(item.get("title", ""))
                    page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                    results.append(
                        {
                            "title": title,
                            "url": page_url,
                            "snippet": clean_snippet,
                        }
                    )
        except Exception:
            pass

    # 3. Save to disk cache for reproducible reruns
    if results:
        set_search_cache(query=clean_query, max_results=max_results, results=results)
    return results
