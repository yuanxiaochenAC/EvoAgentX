from typing import Dict, Any, Union, List

from llama_index.core.schema import BaseNode
from llama_index.core.storage import StorageContext
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.indices import PropertyGraphIndex

from evoagentx.rag.schema import Chunk
from evoagentx.core.logging import logger
from .base import BaseIndexWrapper, IndexType
from evoagentx.models.base_model import BaseLLM
from evoagentx.storages.base import StorageHandler
from evoagentx.rag.transforms.graph_extractor import GraphExtractLLM

class GraphIndexing(BaseIndexWrapper):
    """Wrapper for LlamaIndex PropertyGraphIndex."""
    
    def __init__(
        self,
        embed_model: BaseEmbedding,
        storage_handler: StorageHandler,
        llm: BaseLLM,
        index_config: Dict[str, Any] = None
    ) -> None:
        super().__init__()
        self.index_type = IndexType.GRAPH
        self.embed_model = embed_model
        # create a storage_context for llama_index
        self._create_storage_context()
        # for caching llama_index node
        self.id_to_node = dict()

        self.index_config = index_config or {}
        # initlize the kg_extractor
        kg_extractor = GraphExtractLLM(
            llm=llm, 
            num_workers=self.index_config.get("num_workers", 4),
            max_paths_per_chunk=self.index_config.get("max_paths_per_chunk", 5),
        )
        try:
            self.index = PropertyGraphIndex(
                nodes=[],
                kg_extractors=[kg_extractor],
                embed_model=self.embed_model,
                storage_context=self.storage_context,
                show_progress=self.index_config.get("show_progress", False),
                use_async=self.index_config.get("use_async", True),
            )
        except Exception as e:
            logger.error(f"Failed to initialize VectorStoreIndex: {str(e)}")
            raise

    def _create_storage_context(self):
        """Create the LlamaIndex-compatible storage context."""
        super()._create_storage_context()
        # Construct a storage_context for llama_index
        assert self.storage_handler.graph_store is not None, "GraphIndexing must init a graph backend in 'storageHandler'"
        # check if a vector store is initilized in storageHandler,
        # then use the hybrid-search.
        vector_store = self.storage_handler.vector_store.get_vector_store() if self.storage_handler.vector_store is not None else None
        self.storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            graph_store=self.storage_handler.graph_store.get_graph_store()
        )

    def insert_nodes(self, nodes: List[Union[Chunk, BaseNode]]):
        """Insert oo Update nodes into the index.
        
        Args:
            nodes (List[Union[Chunk, BaseNode]]): The nodes to insert.
        """
        try:
            filted_node = [node.to_llama_node() if isinstance(node, Chunk) else node for node in nodes]
            self.index.insert_nodes(filted_node)

        except Exception as e:
            logger.error(f"Failed to insert nodes: {str(e)}")
            raise

    def delete_nodes(self, node_ids: Optional[List[str]] = None, metadata_filters: Optional[Dict[str, Any]] = None) -> None:
        return super().delete_nodes(node_ids, metadata_filters)