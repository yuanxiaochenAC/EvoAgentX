from typing import Optional

from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.graph_stores.types import GraphStore
from llama_index.core.indices.property_graph.retriever import PGRetriever
from llama_index.core.indices.property_graph.sub_retrievers.vector import VectorContextRetriever
from llama_index.core.indices.property_graph.sub_retrievers.llm_synonym import LLMSynonymRetriever

from .base import BaseRetrieverWrapper
from evoagentx.core.logging import logger
from evoagentx.rag.schema import Query, RagResult, Corpus

"""
    def as_retriever(
        self,
        sub_retrievers: Optional[List["BasePGRetriever"]] = None,
        include_text: bool = True,
        **kwargs: Any,
    ) -> BaseRetriever:

        from llama_index.core.indices.property_graph.retriever import (
            PGRetriever,
        )
        from llama_index.core.indices.property_graph.sub_retrievers.vector import (
            VectorContextRetriever,
        )
        from llama_index.core.indices.property_graph.sub_retrievers.llm_synonym import (
            LLMSynonymRetriever,
        )

        if sub_retrievers is None:
            sub_retrievers = [
                LLMSynonymRetriever(
                    graph_store=self.property_graph_store,
                    include_text=include_text,
                    llm=self._llm,
                    **kwargs,
                ),
            ]

            if self._embed_model and (
                self.property_graph_store.supports_vector_queries or self.vector_store
            ):
                sub_retrievers.append(
                    VectorContextRetriever(
                        graph_store=self.property_graph_store,
                        vector_store=self.vector_store,
                        include_text=include_text,
                        embed_model=self._embed_model,
                        **kwargs,
                    )
                )

        return PGRetriever(sub_retrievers, use_async=self._use_async, **kwargs)
"""

class GraphRetriever(BaseRetrieverWrapper):
    """Wrapper for graph-based retrieval."""
    
    def __init__(self, graph_store: GraphStore, embed_model: Optional[BaseEmbedding], top_k: int = 5):
        super().__init__()
        self.graph_store = graph_store
        self.embed_model = embed_model
        self.top_k = top_k
        
        self.retriever = VectorContextRetriever(
            graph_store=self.graph_store,
            embed_model=self.embed_model,
            similarity_top_k=self.top_k
        )
    
    def retrieve(self, query: Query) -> RagResult:
        try:
            nodes = self.retriever.retrieve(query.query_str)
            corpus = Corpus.from_llama_nodes(nodes)
            scores = [node.score or 0.0 for node in nodes]
            
            for chunk, score in zip(corpus.chunks, scores):
                chunk.metadata.similarity_score = score
            
            result = RagResult(
                corpus=corpus,
                scores=scores,
                metadata={"query": query.query_str, "retriever": "graph"}
            )
            logger.info(f"Graph retrieved {len(corpus.chunks)} chunks")
            return result
        except Exception as e:
            logger.error(f"Graph retrieval failed: {str(e)}")
            raise
    
    def get_retriever(self) -> VectorContextRetriever:
        logger.debug("Returning graph retriever")
        return self.retriever