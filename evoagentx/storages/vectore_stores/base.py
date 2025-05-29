from enum import Enum
from typing import List, Dict, Any
from abc import ABC, abstractmethod

from llama_index.core.vector_stores.types import BasePydanticVectorStore

from evoagentx.rag.schema import Chunk


class VectorStoreType(str, Enum):
    FAISS = "faiss"


class VectorStoreBase(ABC):
    """Base interface for vector stores."""

    @abstractmethod
    def get_vector_store(self) -> BasePydanticVectorStore:
        """Return the LlamaIndex-compatible vector store."""
        pass

    @abstractmethod
    def add(self, chunks: List[Chunk]) -> List[str]:
        """Add chunks to the vector store.
        
        Args:
            chunks (List[Chunk]): Chunks to add.
            
        Returns:
            List[str]: IDs of added chunks.
        """
        pass
    
    @abstractmethod
    def query(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Query the vector store.
        
        Args:
            query_embedding (List[float]): Embedding of the query.
            top_k (int): Number of top results to return.
            
        Returns:
            List[Dict[str, Any]]: Retrieved chunks with metadata and scores.
        """
        pass
    
    @abstractmethod
    def delete(self, chunk_ids: List[str]) -> bool:
        """Delete chunks from the vector store.
        
        Args:
            chunk_ids (List[str]): IDs of chunks to delete.
            
        Returns:
            bool: True if deletion was successful.
        """
        pass