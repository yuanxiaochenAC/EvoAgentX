import logging

from llama_index.embeddings.openai import OpenAIEmbedding

from .base import BaseEmbeddingWrapper


class OpenAIEmbeddingWrapper(BaseEmbeddingWrapper):
    """Wrapper for OpenAI embedding models."""
    
    def __init__(
        self,
        model_name: str = "text-embedding-ada-002",
        api_key: str = None,
        dimensions: int = None,
        **kwargs
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.dimensions = dimensions
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)
    
    def get_embedding_model(self) -> OpenAIEmbedding:
        """Return the OpenAI embedding model."""
        try:
            model = OpenAIEmbedding(
                model_name=self.model_name,
                api_key=self.api_key,
                dimensions=self.dimensions,
                **self.kwargs
            )
            self.logger.debug(f"Initialized OpenAI embedding model: {self.model_name}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI embedding: {str(e)}")
            raise