from typing import List, Dict, Any, Optional
from enum import Enum
from abc import ABC, abstractmethod
from llama_index.core.indices import BaseIndex
from llama_index.core import VectorStoreIndex, SummaryIndex, TreeIndex, PropertyGraphIndex
from llama_index.core.node_parser import NodeParser, SentenceWindowNodeParser, HierarchicalNodeParser
from llama_index.core.embeddings import BaseEmbedding
from evoagentx.rag.schema import Document, Corpus
import logging

class IndexType(str, Enum):
    VECTOR = "vector"
    SUMMARY = "summary"
    TREE = "tree"
    GRAPH = "graph"

class NodeParserType(str, Enum):
    SIMPLE = "simple"
    SENTENCE_WINDOW = "sentence_window"
    HIERARCHICAL = "hierarchical"

class IndexBuilder(ABC):
    """Base interface for index builders."""
    
    @abstractmethod
    def build_index(self, documents: List[Document], corpus: Corpus) -> BaseIndex:
        """Build a LlamaIndex-compatible index."""
        pass

class IndexFactory:
    """Factory for creating LlamaIndex indices."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create(
        self,
        index_type: IndexType,
        embed_model: BaseEmbedding,
        node_parser: NodeParser,
        storage_context: Optional[Any] = None,
        index_config: Dict[str, Any] = None
    ) -> BaseIndex:
        """Create an index based on configuration.
        
        Args:
            index_type (IndexType): Type of index (e.g., vector, summary).
            embed_model (BaseEmbedding): Embedding model.
            node_parser (NodeParser): Node parser for chunking.
            storage_context (Any, optional): LlamaIndex storage context.
            index_config (Dict[str, Any], optional): Additional index configuration.
            
        Returns:
            BaseIndex: A LlamaIndex index.
        """
        index_config = index_config or {}
        storage_context = storage_context or {}
        
        if index_type == IndexType.VECTOR:
            index = VectorStoreIndex(
                nodes=[],
                embed_model=embed_model,
                storage_context=storage_context,
                **index_config
            )
        elif index_type == IndexType.SUMMARY:
            index = SummaryIndex(
                nodes=[],
                storage_context=storage_context,
                **index_config
            )
        elif index_type == IndexType.TREE:
            index = TreeIndex(
                nodes=[],
                storage_context=storage_context,
                **index_config
            )
        elif index_type == IndexType.GRAPH:
            index = PropertyGraphIndex(
                nodes=[],
                storage_context=storage_context,
                **index_config
            )
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        self.logger.info(f"Created index of type: {index_type}")
        return index
    
    def get_node_parser(self, parser_type: NodeParserType, parser_config: Dict[str, Any] = None) -> NodeParser:
        """Create a node parser based on configuration.
        
        Args:
            parser_type (NodeParserType): Type of node parser.
            parser_config (Dict[str, Any], optional): Parser configuration.
            
        Returns:
            NodeParser: A LlamaIndex node parser.
        """
        parser_config = parser_config or {}
        
        if parser_type == NodeParserType.SIMPLE:
            from llama_index.core.node_parser import SimpleNodeParser
            parser = SimpleNodeParser.from_defaults(**parser_config)
        elif parser_type == NodeParserType.SENTENCE_WINDOW:
            parser = SentenceWindowNodeParser.from_defaults(**parser_config)
        elif parser_type == NodeParserType.HIERARCHICAL:
            parser = HierarchicalNodeParser.from_defaults(**parser_config)
        else:
            raise ValueError(f"Unsupported node parser type: {parser_type}")
        
        self.logger.info(f"Created node parser of type: {parser_type}")
        return parser