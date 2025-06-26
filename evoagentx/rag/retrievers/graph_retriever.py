from typing import Any, Callable, Optional, List


from llama_index.core.schema import NodeWithScore
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.indices.property_graph import PGRetriever
from llama_index.core.vector_stores.types import  BasePydanticVectorStore
from llama_index.core.graph_stores.types import PropertyGraphStore, KG_SOURCE_REL
from llama_index.core.indices.property_graph.sub_retrievers.base import BasePGRetriever
from llama_index.core.indices.property_graph.sub_retrievers.vector import VectorContextRetriever

from .base import BaseRetrieverWrapper
from evoagentx.models import BaseLLM
from evoagentx.core.logging import logger
from evoagentx.rag.schema import Query, RagResult, Corpus, Chunk
from evoagentx.prompts.rag.graph_synonym import DEFAULT_SYNONYM_EXPAND_TEMPLATE

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


class BasicLLMSynonymRetriever(BasePGRetriever):
    """
    """
    def __init__(
        self,
        graph_store: PropertyGraphStore,
        include_text: bool = True,
        include_properties: bool = False,
        synonym_prompt: str = DEFAULT_SYNONYM_EXPAND_TEMPLATE,
        max_keywords: int = 10,
        path_depth: int = 2,
        limit: int = 30,
        output_parsing_fn: Optional[Callable] = None,
        llm: Optional[BaseLLM] = None,
        **kwargs: Any,
    ) -> None:

        self._llm = llm
        self._synonym_prompt = synonym_prompt
        self._output_parsing_fn = output_parsing_fn
        self._max_keywords = max_keywords
        self._path_depth = path_depth
        self._limit = limit

        super().__init__(
            graph_store=graph_store,
            include_text=include_text,
            include_properties=include_properties,
            **kwargs,
        )

    def _parse_llm_output(self, output: str) -> List[str]:
        if self._output_parsing_fn:
            matches = self._output_parsing_fn(output)
        else:
            matches = output.strip().split("^")

        # capitalize to normalize with ingestion
        return [x.strip().capitalize() for x in matches if x.strip()]

    def _prepare_matches(
        self, matches: List[str], limit: Optional[int] = None
    ) -> List[NodeWithScore]:
        kg_nodes = self._graph_store.get(ids=matches)
        triplets = self._graph_store.get_rel_map(
            kg_nodes,
            depth=self._path_depth,
            limit=limit or self._limit,
            ignore_rels=[KG_SOURCE_REL],
        )

        return self._get_nodes_with_score(triplets)

    async def _aprepare_matches(
        self, matches: List[str], limit: Optional[int] = None
    ) -> List[NodeWithScore]:
        kg_nodes = await self._graph_store.aget(ids=matches)
        triplets = await self._graph_store.aget_rel_map(
            kg_nodes,
            depth=self._path_depth,
            limit=limit or self._limit,
            ignore_rels=[KG_SOURCE_REL],
        )

        return self._get_nodes_with_score(triplets)

    def retrieve_from_graph(
        self, query_bundle: Query, limit: Optional[int] = None
    ) -> List[NodeWithScore]:
        response = self._llm.predict(
            self._synonym_prompt,
            query_str=query_bundle.query_str,
            max_keywords=self._max_keywords,
        )
        matches = self._parse_llm_output(response)

        return self._prepare_matches(matches, limit=limit or self._limit)

    async def aretrieve_from_graph(
        self, query_bundle: Query, limit: Optional[int] = None
    ) -> List[NodeWithScore]:
        response = await self._llm.apredict(
            self._synonym_prompt,
            query_str=query_bundle.query_str,
            max_keywords=self._max_keywords,
        )
        matches = self._parse_llm_output(response)

        return await self._aprepare_matches(matches, limit=limit or self._limit)


class GraphRetriever(BaseRetrieverWrapper):
    """Wrapper for graph-based retrieval."""
    
    def __init__(self, graph_store: PropertyGraphStore, embed_model: Optional[BaseEmbedding], 
                 include_text: bool = True, _use_async: bool = True,
                 vector_store: Optional[BasePydanticVectorStore] = None, top_k: int = 5):
        super().__init__()
        self.graph_store = graph_store
        self.embed_model = embed_model
        self.vector_store = vector_store
        self.top_k = top_k
        
        sub_retrievers = [
            BasicLLMSynonymRetriever(),

        ]
        if self._embed_model and (
                self.graph_store.supports_vector_queries or self.vector_store
        ):
            sub_retrievers.append(
                VectorContextRetriever(
                        graph_store=self.graph_store,
                        vector_store=self.vector_store,
                        include_text=include_text,
                        embed_model=self._embed_model,
                    )
            )


        self.retriever = PGRetriever(
            sub_retrievers, use_async=_use_async
        )
    
    async def aretrieve(self, query: Query) -> RagResult:
        try:
            # config the top_k
            self.retriever.similarity_top_k = query.top_k
            nodes = await self.retriever.aretrieve(query.query_str)

            corpus = Corpus()
            scores = []
            
            if nodes is None:
                return RagResult(corpus=corpus, scores=scores, metadata={"query": query.query_str, "retriever": "vector"})
            
            for score_node in nodes:
                chunk = Chunk.from_llama_node(score_node.node)
                chunk.metadata.similarity_score = score_node.score or 0.0
                corpus.add_chunk(chunk)
                scores.extend([score_node.score or 0.0])
            
            result = RagResult(
                corpus=corpus,
                scores=scores,
                metadata={"query": query.query_str, "retriever": "vector"}
            )
            logger.info(f"Vector retrieved {len(corpus.chunks)} chunks")
            return result
        except Exception as e:
            logger.error(f"Vector retrieval failed: {str(e)}")
            raise

    def retrieve(self, query: Query) -> RagResult:
        try:
            # config the top_k
            self.retriever.similarity_top_k = query.top_k
            nodes = self.retriever.retrieve(query.query_str)
            corpus = Corpus()
            scores = []

            if nodes is None:
                return RagResult(corpus=corpus, scores=scores, metadata={"query": query.query_str, "retriever": "vector"})
            
            for score_node in nodes:
                chunk = Chunk.from_llama_node(score_node.node)
                chunk.metadata.similarity_score = score_node.score or 0.0
                corpus.add_chunk(chunk)
                scores.extend([score_node.score or 0.0])
            
            result = RagResult(
                corpus=corpus,
                scores=scores,
                metadata={"query": query.query_str, "retriever": "vector"}
            )
            logger.info(f"Vector retrieved {len(corpus.chunks)} chunks")
            return result
        except Exception as e:
            logger.error(f"Vector retrieval failed: {str(e)}")
            raise
    
    def get_retriever(self) -> PGRetriever:
        logger.debug("Returning graph retriever")
        return self.retriever