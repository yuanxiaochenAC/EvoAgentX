from enum import Enum
from typing import List, Any, Dict
from abc import ABC, abstractmethod

from llama_index.core.graph_stores.simple import GraphStore


class GraphStoreType(str, Enum):
    NEO4J = "neo4j"

class GraphStoreBase(ABC):
    """Base interface for graph stores."""
    
    @abstractmethod
    def get_graph_store(self) -> GraphStore:
        """Return the LlamaIndex-compatible graph store."""
        pass