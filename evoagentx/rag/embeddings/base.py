from enum import Enum
from abc import ABC, abstractmethod

from llama_index.core.embeddings import BaseEmbedding


class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"

class BaseEmbeddingWrapper(ABC):
    """Base interface for embedding wrappers."""
    
    @abstractmethod
    def get_embedding_model(self) -> BaseEmbedding:
        """Return the LlamaIndex-compatible embedding model."""
        pass