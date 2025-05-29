import logging
from typing import Dict, Any

from llama_index.core.vector_stores.types import VectorStore

from .base import VectorStoreType
from .faiss import FaissVectorStoreWrapper

__all__ = ['FaissVectorStoreWrapper', 'VectorStoreFactory']

class VectorStoreFactory:
    """Factory for creating vector stores."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create(
        self,
        store_type: str,
        store_config: Dict[str, Any] = None
    ) -> VectorStore:
        """Create a vector store based on configuration.
        
        Args:
            store_type (str): The type of vector store (e.g., 'faiss').
            store_config (Dict[str, Any], optional): Store configuration.
            
        Returns:
            VectorStore: A LlamaIndex-compatible vector store.
            
        Raises:
            ValueError: If the store type or configuration is invalid.
        """
        store_config = store_config or {}
        
        if store_type == VectorStoreType.FAISS:
            dimension = store_config.get("dimension")
            if not dimension or not isinstance(dimension, int):
                raise ValueError("FAISS requires a valid dimension")
            vector_store = FaissVectorStoreWrapper(**store_config).get_vector_store()
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")
        
        self.logger.info(f"Created vector store: {store_type}")
        return vector_store