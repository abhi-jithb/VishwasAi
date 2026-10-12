"""Connectivity test script for web search abstraction."""

import sys


def main() -> None:
    """Test Tavily web search and print results or an informative error."""
    try:
        from backend.search import web_search

        query = "Vishwas AI fact checking"
        print(f"Testing web search connectivity [Provider: Tavily] with query: '{query}'...")
        results = web_search(query=query, max_results=3)

        print(f"\n--- Search Results Received ({len(results)} items) ---")
        for idx, item in enumerate(results, start=1):
            title = item.get("title", "No Title")
            url = item.get("url", "No URL")
            snippet = item.get("snippet", "").replace("\n", " ").strip()
            print(f"[{idx}] {title}")
            print(f"    URL: {url}")
            print(f"    Snippet: {snippet[:150]}..." if len(snippet) > 150 else f"    Snippet: {snippet}")
            print()
        print("-----------------------------------------------------")
        print("Search connectivity test completed successfully.")
    except Exception as exc:
        print(f"\n[ERROR] Search connectivity test failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
