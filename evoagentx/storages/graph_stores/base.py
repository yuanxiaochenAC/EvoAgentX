from typing import List, Any, Dict
from abc import ABC, abstractmethod

from llama_index.core.graph_stores.simple import GraphStore


class GraphStoreBase(ABC):
    """Base interface for graph stores."""
    
    @abstractmethod
    def get_graph_store(self) -> GraphStore:
        """Return the LlamaIndex-compatible graph store."""
        pass
    
    @abstractmethod
    def add_triples(self, triples: List[Dict[str, Any]]):
        """Add triples to the graph store."""
        pass
    
    @abstractmethod
    def query(self, query: str) -> List[Any]:
        """Query the graph store."""
        pass