"""
Running the search engine
Author: Eeshan Gupta
"""

from search_engine import NewsSearch

if __name__ == "__main__":
    search = NewsSearch(query="Narendra Modi", max_results=10)
    search.search()
    search.print_results()
