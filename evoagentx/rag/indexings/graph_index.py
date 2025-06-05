import logging
from typing import List, Dict, Any, Union

from llama_index.core.schema import BaseNode
from llama_index.core.storage import StorageContext
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.indices.property_graph import PropertyGraphIndex

from .base import BaseIndexWrapper, IndexType
from ..schema import Chunk


class GraphIndexing(BaseIndexWrapper):
    """Wrapper for LlamaIndex PropertyGraphIndex."""
    
    def __init__(
        self,
        embed_model: BaseEmbedding,
        storage_context: StorageContext,
        index_config: Dict[str, Any] = None
    ):
        super().__init__()
        self.logger = logging.getLogger(__file__)
        self.embed_model = embed_model
        self.storage_context = storage_context
        self.index_config = index_config or {}
        self.index_type = IndexType.GRAPH
        try:
            self.index = PropertyGraphIndex(
                nodes=[],
                embed_model=self.embed_model,
                storage_context=self.storage_context,
                kg_extractors=self.index_config.get("kg_extractors", []),
                show_progress=self.index_config.get("show_progress", False),
                **self.index_config.get("graph_store_args", {})
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize PropertyGraphIndex: {str(e)}")
            raise
    
    def get_index(self) -> PropertyGraphIndex:
        self.logger.debug("Returning PropertyGraphIndex")
        return self.index
    
    def insert_nodes(self, nodes: List[Union[Chunk, BaseNode]]):
        try:
            nodes = [(node.to_llama_node() if isinstance(node, Chunk) else node) for node in nodes]
            self.index.insert_nodes(nodes)
            self.logger.info(f"Inserted {len(nodes)} nodes into PropertyGraphIndex")
        except Exception as e:
            self.logger.error(f"Failed to insert nodes: {str(e)}")
            raise