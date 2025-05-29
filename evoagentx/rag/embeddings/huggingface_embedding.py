import logging
from typing import Dict, Any

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from .base import BaseEmbeddingWrapper


class HuggingFaceEmbeddingWrapper(BaseEmbeddingWrapper):
    """Wrapper for HuggingFace embedding models."""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        **kwargs
    ):
        self.model_name = model_name
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)
    
    def get_embedding_model(self) -> HuggingFaceEmbedding:
        """Return the HuggingFace embedding model."""
        try:
            model = HuggingFaceEmbedding(
                model_name=self.model_name,
                **self.kwargs
            )
            self.logger.debug(f"Initialized HuggingFace embedding model: {self.model_name}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to initialize HuggingFace embedding: {str(e)}")
            raise