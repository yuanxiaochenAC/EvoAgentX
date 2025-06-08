import logging
from typing import Dict, Any

from llama_index.core.vector_stores.types import VectorStore

from .base import VectorStoreType, VectorStoreBase
from .faiss import FaissVectorStoreWrapper

__all__ = ['VectorStoreBase', 'FaissVectorStoreWrapper', 'VectorStoreFactory']

class VectorStoreFactory:
    """Factory for creating vector stores."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create(self, store_type: str, store_config: Dict[str, Any] = None) -> VectorStore:
        store_config = store_config or {}
        if store_type == VectorStoreType.FAISS:
            dimensions = store_config.get("dimensions")
            if not dimensions or not isinstance(dimensions, int):
                raise ValueError("FAISS requires a valid dimension")
            vector_store = FaissVectorStoreWrapper(**store_config)
        # elif store_type == VectorStoreType.QDRANT:
        #     qdrant_url = store_config.get("qdrant_url")
        #     qdrant_api_key = store_config.get("qdrant_api_key")
        #     collection_name = store_config.get("qdrant_collection_name", "default_collection")
        #     if not qdrant_url:
        #         raise ValueError("Qdrant requires a valid URL")
        #     client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        #     vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")
        self.logger.info(f"Created vector store: {store_type}")
        return vector_store