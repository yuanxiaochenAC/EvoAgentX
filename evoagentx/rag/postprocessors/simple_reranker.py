import logging
from typing import List, Optional

from llama_index.core.schema import NodeWithScore
from llama_index.core.postprocessor import SimilarityPostprocessor, KeywordNodePostprocessor

from .base import BasePostprocessor
from ..schema import RagQuery, RagResult, Corpus


class SimpleReranker(BasePostprocessor):
    """Post-processor for reranking retrieval results."""
    
    def __init__(
        self,
        similarity_cutoff: Optional[float] = None,
        keyword_filters: Optional[List[str]] = None
    ):
        super().__init__()
        self.postprocessors = []
        if similarity_cutoff:
            self.postprocessors.append(SimilarityPostprocessor(similarity_cutoff=similarity_cutoff))
        if keyword_filters:
            self.postprocessors.append(KeywordNodePostprocessor(required_keywords=keyword_filters))
    
    def postprocess(self, query: RagQuery, results: List[RagResult]) -> RagResult:
        try:
            nodes = []
            for result in results:
                for chunk, score in zip(result.corpus.chunks, result.scores):
                    node = chunk.to_llama_node()
                    nodes.append(NodeWithScore(node=node, score=score))
            
            for postprocessor in self.postprocessors:
                nodes = postprocessor.postprocess_nodes(nodes)
            
            corpus = Corpus.from_llama_nodes(nodes)
            scores = [node.score for node in nodes]
            
            for chunk, score in zip(corpus.chunks, scores):
                chunk.metadata.similarity_score = score
            
            result = RagResult(
                corpus=corpus,
                scores=scores,
                metadata={"query": query.query_str, "postprocessor": "reranker"}
            )
            self.logger.info(f"Reranked to {len(corpus.chunks)} chunks")
            return result
        except Exception as e:
            self.logger.error(f"Reranking failed: {str(e)}")
            raise