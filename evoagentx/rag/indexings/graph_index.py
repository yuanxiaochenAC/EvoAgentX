from typing import Dict, Any, Union, List, Optional

from llama_index.core.schema import BaseNode
from llama_index.core.storage import StorageContext
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.indices import PropertyGraphIndex
from llama_index.core.indices.property_graph.transformations import ImplicitPathExtractor

from evoagentx.rag.schema import Chunk
from evoagentx.core.logging import logger
from .base import BaseIndexWrapper, IndexType
from evoagentx.models.base_model import BaseLLM
from evoagentx.storages.base import StorageHandler
from evoagentx.rag.transforms.graph_extract import BasicGraphExtractLLM

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
        self._embed_model = embed_model
        self.storage_handler = storage_handler
        # create a storage_context for llama_index
        self._create_storage_context()
        # for caching llama_index node
        self.id_to_node = dict()

        self.index_config = index_config or {}
        # initlize the kg_extractor
        assert isinstance(llm, BaseLLM), f"The LLM model should be an instance class."
        kg_extractor = BasicGraphExtractLLM(
            llm=llm, 
            num_workers=self.index_config.get("num_workers", 4),
        )
        try:
            self.index = PropertyGraphIndex(
                nodes=[],
                kg_extractors=[kg_extractor, ImplicitPathExtractor()],  # 'ImplicitPathExtractor' for node basic relation(tree structure and so on).
                embed_model=self._embed_model,
                vector_store=self.storage_context.vector_store,
                property_graph_store=self.storage_context.graph_store,
                storage_context=self.storage_context,
                show_progress=self.index_config.get("show_progress", True),
                use_async=self.index_config.get("use_async", True),
            )
        except Exception as e:
            logger.error(f"Failed to initialize {self.__class__}: {str(e)}")
            raise

    def get_index(self) -> PropertyGraphIndex:
        return self.index

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
            filted_nodes = [node.to_llama_node() if isinstance(node, Chunk) else node for node in nodes]

            # Convert dict to string
            for node in filted_nodes:
                flattened_metadata = {}
                for key, value in node.metadata.items():
                    if isinstance(value, dict):
                        flattened_metadata[key] = str(value)
                    elif isinstance(value, (list, tuple)):
                        flattened_metadata[key] = [str(v) for v in value if not isinstance(v, (dict, list))]
                    else:
                        flattened_metadata[key] = value
                node.metadata = flattened_metadata
            nodes = self.index._insert_nodes(filted_nodes)
            for node in nodes:
                self.id_to_node[node.node_id] = node.model_copy()
            logger.info(f"Inserted {len(nodes)} nodes into PropertyGraphIndex")
            # Caching the node 
        except Exception as e:
            logger.error(f"Failed to insert nodes: {str(e)}")
            raise

    def delete_nodes(self, node_ids: Optional[List[str]] = None, metadata_filters: Optional[Dict[str, Any]] = None):
        try:
            if node_ids:
                for node_id in node_ids:
                    if node_id in self.id_to_node:
                        self.index.delete_nodes([node_id])
                        self.id_to_node.pop(node_id)
                        logger.info(f"Deleted node {node_id} from PropertyGraphIndex")
            elif metadata_filters:
                nodes_to_delete = []
                for node_id, node in self.id_to_node.items():
                    if all(node.metadata.get(k) == v for k, v in metadata_filters.items()):
                        nodes_to_delete.append(node_id)
                if nodes_to_delete:
                    self.index.delete_nodes(nodes_to_delete)
                    for node_id in nodes_to_delete:
                        self.id_to_node.pop(node_id)
                    logger.info(f"Deleted {len(nodes_to_delete)} nodes matching metadata filters from TreeIndex")
            else:
                logger.warning("No node_ids or metadata_filters provided for deletion")
        except Exception as e:
            logger.error(f"Failed to delete nodes: {str(e)}")
            raise

    def clear(self):
        try:
            node_ids = list(self.id_to_node.keys())
            self.index.delete_nodes(node_ids)
            self.id_to_node.clear()
            logger.info("Cleared all nodes from TreeIndex")
        except Exception as e:
            logger.error(f"Failed to clear index: {str(e)}")
            raise