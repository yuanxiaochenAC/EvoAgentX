import logging
from typing import List, Dict, Any, Union

from llama_index.core import TreeIndex
from llama_index.core.schema import BaseNode
from llama_index.core.storage import StorageContext
from llama_index.core.embeddings import BaseEmbedding

from .base import BaseIndexWrapper
from ..schema import Chunk


class TreeIndex(BaseIndexWrapper):
    """Wrapper for LlamaIndex TreeIndex."""
    
    def __init__(
        self,
        embed_model: BaseEmbedding,
        storage_context: StorageContext,
        index_config: Dict[str, Any] = None
    ):
        super().__init__()
        self.embed_model = embed_model
        self.storage_context = storage_context
        self.index_config = index_config or {}
        try:
            self.index = TreeIndex(
                nodes=[],
                storage_context=self.storage_context,
                show_progress=self.index_config.get("show_progress", False),
                **self.index_config.get("tree_args", {})
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize TreeIndex: {str(e)}")
            raise
    
    def get_index(self) -> TreeIndex:
        self.logger.debug("Returning TreeIndex")
        return self.index
    
    def insert_nodes(self, nodes: List[Union[Chunk, BaseNode]]):
        try:
            nodes = [(node.to_llama_node() if isinstance(node, Chunk) else node) for node in nodes]
            self.index.insert_nodes(nodes)
            self.logger.info(f"Inserted {len(nodes)} nodes into TreeIndex")
        except Exception as e:
            self.logger.error(f"Failed to insert nodes: {str(e)}")
            raise