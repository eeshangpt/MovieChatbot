"""
Search Wrappers to get news articles from DuckDuckGo.
Author: Eeshan Gupta
"""

import gc
import json
import sys
from datetime import datetime

try:
    from ddgs import DDGS
except ImportError:
    print("❌ Missing dependency. Run: pip install ddgs")
    sys.exit(1)

from .base import SearchEngine


class DuckDuckGoNewsSearch(SearchEngine):
    def __init__(self, query: str, max_results: int = 5) -> None:
        super().__init__(query=query, max_results=max_results)
        # self.query: str = query
        # self.max_results: int = max_results
        # self.results: List[Dict[str, str]] | None = None

    def search(self) -> None:
        """
        Search DuckDuckGo and return a list of structured result dicts.
        """
        self.results = []

        with DDGS() as ddgs:
            # for r in ddgs.text(self.query, max_results=self.max_results):
            for r in ddgs.news(self.query, max_results=self.max_results):
                important_fields = {
                    "title": r.get("title", "N/A"),
                    "url": r.get("url", "N/A"),
                    "summary": r.get("body", "N/A"),
                    "source": r.get("source", "N/A"),
                }
                self.results.append(important_fields)
                del important_fields

            gc.collect()

    def print_results(self) -> None:
        """Pretty-print titles, URLs, and summaries to the terminal."""
        print(f"\n🔍 Search results for: '{self.query}'")
        print("=" * 60)

        if not self.results:
            print("No results found.")
            return

        for i, r in enumerate(self.results, start=1):
            print(f"\n[{i}] {r['title']}")
            print(f"    🔗 {r['url']}")
            print(f"    📄 {r['summary']}")

        print("\n" + "=" * 60)

    def save_results(self, output_file: str) -> None:
        """Save results as structured JSON to a file."""
        payload = {
            "query": self.query,
            "timestamp": datetime.now().isoformat(),
            "total_results": len(self.results) if self.results else 0,
            "results": self.results,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Results saved to: {output_file}")
