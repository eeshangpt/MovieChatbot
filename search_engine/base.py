"""
Base Class for Search Engines
Author: Eeshan Gupta
"""

from abc import ABC, abstractmethod
from typing import Dict, List


class SearchEngine(ABC):
    def __init__(self, query: str, max_results: int = 5) -> None:
        self.query: str = query
        self.max_results: int = max_results
        self.results: List[Dict[str, str]] | None = None

    @abstractmethod
    def search(self) -> None:
        pass

    @abstractmethod
    def print_results(self) -> None:
        pass
