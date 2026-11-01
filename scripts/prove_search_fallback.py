"""Prove the search fallback chain: DuckDuckGo when Tavily is off, Wikipedia when DuckDuckGo fails.

Run with Tavily disabled and the cache off:
    SEARCH_API_KEY= DISABLE_CACHE=1 python scripts/prove_search_fallback.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.search import web_search

QUERY = "Alexander Fleming discovered penicillin in 1928"


def show(label: str) -> None:
    results = web_search(QUERY, max_results=3)
    print(f"--- {label}: {len(results)} result(s)")
    for r in results:
        print(f"  [{r.get('provider')}] {r.get('title')} | {r.get('url')}")


if __name__ == "__main__":
    print(f"SEARCH_API_KEY set: {bool(os.getenv('SEARCH_API_KEY'))} | DISABLE_CACHE={os.getenv('DISABLE_CACHE')}")
    show("Tavily disabled")
    # Force DuckDuckGo to fail: make both package imports raise ImportError.
    sys.modules["ddgs"] = None
    sys.modules["duckduckgo_search"] = None
    show("Tavily disabled + DuckDuckGo forced to fail")
