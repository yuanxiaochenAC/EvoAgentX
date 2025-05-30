import logging
from llama_index.core.indices.base import BaseIndex
from llama_index.core.retrievers import VectorIndexRetriever

from .base import BaseRetrieverWrapper
from ..schema import RagQuery, RagResult, Corpus


class VectorRetriever(BaseRetrieverWrapper):
    """Wrapper for vector-based retrieval."""
    
    def __init__(self, index: BaseIndex, top_k: int = 5):
        super().__init__()
        self.index = index
        self.top_k = top_k
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.top_k
        )
        self.logger = logging.getLogger(__file__)
    
    def retrieve(self, query: RagQuery) -> RagResult:
        try:
            nodes = self.retriever.retrieve(query.query_str)
            corpus = Corpus.from_llama_nodes(nodes)
            scores = [node.score or 0.0 for node in nodes]
            
            for chunk, score in zip(corpus.chunks, scores):
                chunk.metadata.similarity_score = score
            
            result = RagResult(
                corpus=corpus,
                scores=scores,
                metadata={"query": query.query_str, "retriever": "vector"}
            )
            self.logger.info(f"Vector retrieved {len(corpus.chunks)} chunks")
            return result
        except Exception as e:
            self.logger.error(f"Vector retrieval failed: {str(e)}")
            raise
    
    def get_retriever(self) -> VectorIndexRetriever:
        self.logger.debug("Returning vector retriever")
        return self.retriever