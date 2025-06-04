import logging
from typing import Dict, Any

from llama_index.core.embeddings import BaseEmbedding

from .base import EmbeddingProvider
from .openai_embedding import OpenAIEmbeddingWrapper
from .huggingface_embedding import HuggingFaceEmbeddingWrapper

__all__ = ['OpenAIEmbeddingWrapper', 'HuggingFaceEmbeddingWrapper', 'EmbeddingFactory', 'BaseEmbedding']

class EmbeddingFactory:
    """Factory for creating embedding models based on configuration."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create(
        self,
        provider: EmbeddingProvider,
        model_config: Dict[str, Any] = None
    ) -> BaseEmbedding:
        """Create an embedding model based on the provider and configuration.
        
        Args:
            provider (EmbeddingProvider): The embedding provider (e.g., OpenAI, HuggingFace).
            model_config (Dict[str, Any], optional): Configuration for the embedding model.
            
        Returns:
            BaseEmbedding: A LlamaIndex-compatible embedding model.
            
        Raises:
            ValueError: If the provider or configuration is invalid.
        """
        model_config = model_config or {}
        
        if provider == EmbeddingProvider.OPENAI:
            from .openai_embedding import OpenAIEmbeddingWrapper
            wrapper = OpenAIEmbeddingWrapper(**model_config)
        elif provider == EmbeddingProvider.HUGGINGFACE:
            from .huggingface_embedding import HuggingFaceEmbeddingWrapper
            wrapper = HuggingFaceEmbeddingWrapper(**model_config)
        elif provider == EmbeddingProvider.CUSTOM:
            if "custom_model" not in model_config:
                raise ValueError("Custom embedding model must be provided in model_config.")
            wrapper = model_config["custom_model"]
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
        
        self.logger.info(f"Created embedding model for provider: {provider}")
        return wrapper.get_embedding_model()