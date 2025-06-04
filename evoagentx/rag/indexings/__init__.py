import logging
from typing import Dict, Any, Optional

from llama_index.core.storage import StorageContext
from llama_index.core.embeddings import BaseEmbedding

from .base import IndexType, BaseIndexWrapper
from .vector_index import VectorIndexing
from .graph_index import GraphIndexing
from .summary_index import SummaryIndexing
from .tree_index import TreeIndexing

__all__ = ['VectorIndexing', 'GraphIndexing', 'SummaryIndexing', 'TreeIndexing', 'IndexFactory', 'BaseIndexWrapper']

class IndexFactory:
    """Factory for creating LlamaIndex indices."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create(
        self,
        index_type: IndexType,
        embed_model: BaseEmbedding,
        storage_context: StorageContext,
        index_config: Dict[str, Any] = None,
        node_parser: Optional[Any] = None  # Unused, kept for compatibility
    ) -> BaseIndexWrapper:
        """Create an index based on configuration.
        
        Args:
            index_type (IndexType): The type of index to create.
            embed_model (BaseEmbedding): Embedding model for the index.
            storage_context (StorageContext): Storage context for persistence.
            index_config (Dict[str, Any], optional): Index-specific configuration.
            node_parser (Any, optional): Node parser (unused, kept for compatibility).
            
        Returns:
            BaseIndexWrapper: A wrapped LlamaIndex index.
            
        Raises:
            ValueError: If the index type or configuration is invalid.
        """
        index_config = index_config or {}
        
        if index_type == IndexType.VECTOR:
            index = VectorIndex(
                embed_model=embed_model,
                storage_context=storage_context,
                index_config=index_config
            )
        elif index_type == IndexType.GRAPH:
            if not storage_context.graph_store:
                raise ValueError("Graph store required for PropertyGraphIndex")
            index = GraphIndex(
                embed_model=embed_model,
                storage_context=storage_context,
                index_config=index_config
            )
        elif index_type == IndexType.SUMMARY:
            index = SummaryIndex(
                embed_model=embed_model,
                storage_context=storage_context,
                index_config=index_config
            )
        elif index_type == IndexType.TREE:
            index = TreeIndex(
                embed_model=embed_model,
                storage_context=storage_context,
                index_config=index_config
            )
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        self.logger.info(f"Created index: {index_type}")
        return index