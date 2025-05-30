from typing import Dict, Any

from ..core.base_config import BaseConfig
from .indexings.base import IndexType
from .chunkers.base import ChunkingStrategy
from .embeddings.base import EmbeddingProvider


class RAGConfig(BaseConfig):
    # chunking stage
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.SIMPLE   # Option: SIMPLE, SEMANTIC, HIERARCHICAL
    node_parser_config: Dict[str, Any] = {"chunk_size": 1024, "chunk_overlap": 20}
    # Indexing stage
    index_type: IndexType = IndexType.VECTOR
    index_config: Dict[str, Any] = {}
    # Embedding stage
    embedding_provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    embedding_config: Dict[str, Any] = {"model_name": "text-embedding-ada-002"}
    # retreiving stage
    retrieval_config: Dict[str, Any] = {"top_k": 5, "similarity_cutoff": 0.7}
