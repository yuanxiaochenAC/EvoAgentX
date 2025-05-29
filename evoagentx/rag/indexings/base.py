from enum import Enum
from typing import List, Any
from abc import ABC, abstractmethod

from llama_index.core.indices.base import BaseIndex


class IndexType(str, Enum):
    VECTOR = "vector"
    GRAPH = "graph"
    SUMMARY = "summary"
    TREE = "tree"

class BaseIndexWrapper(ABC):
    """Base interface for index wrappers."""
    
    @abstractmethod
    def get_index(self) -> BaseIndex:
        """Return the LlamaIndex-compatible index."""
        pass
    
    @abstractmethod
    def insert_nodes(self, nodes: List[Any]):
        """Insert nodes into the index."""
        pass