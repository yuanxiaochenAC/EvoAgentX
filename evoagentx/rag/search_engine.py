import logging
from typing import List, Dict, Any, Optional

from llama_index.core.retrievers import RouterRetriever
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.query_engine import RetrieverQueryEngine

from .rag_config import RAGConfig
# Factory
from .indexings import IndexFactory
from .chunkers import ChunkFactory
from .embeddings import EmbeddingFactory
from .retrievers import RetrieverFactory
from .postprocessors import PostprocessorFactory
# Data structure
from .indexings.base import IndexType
from .retrievers.base import RetrieverType
from .embeddings.base import EmbeddingProvider
from ..storages.base import StorageHandler
from .schema import Corpus, Document, Query, SchemaResult


class SearchEngine:
    def __init__(self, config: RAGConfig, storage_handler: StorageHandler):
        self.config = config
        self.storage_handler = storage_handler
        self.embedding_factory = EmbeddingFactory()
        self.index_factory = IndexFactory()
        self.chunk_factory = ChunkFactory()
        self.retriever_factory = RetrieverFactory()
        self.postprocessor_factory = PostprocessorFactory()
        self.embed_model = EmbeddingFactory().create(
            provider=config.embedding_provider,
            model_config=config.embedding_config
        )
        self.chunker = ChunkFactory().create(
            strategy=config.chunking_strategy,
            embed_model=self.embed_model,
            chunker_config=config.node_parser_config
        )
        self.index = IndexFactory().create(
            index_type=config.index_type,
            embed_model=self.embed_model,
            node_parser=None,   # Handle by Chunker
            storage_context=self.storage_handler.storage_context,
            index_config=config.index_config
        )
        self.retrievers = []
        self.query_engine = None
        self.logger = logging.getLogger(__name__)
    
    def add_documents(self, documents: List[Document]) -> Corpus:
        try:
            corpus = self.chunker.chunk(documents)
            nodes = corpus.to_llama_nodes()
            self.index.insert_nodes(nodes)
            
            if self.storage_handler.storage_db:
                for doc in documents:
                    self.storage_handler.storage_db.insert(
                        metadata={
                            "memory_id": doc.doc_id,
                            "doc_id": doc.doc_id,
                            "metadata": doc.metadata.model_dump()
                        },
                        store_type="memory",
                        table="memory"
                    )
                for chunk in corpus.chunks:
                    self.storage_handler.storage_db.insert(
                        metadata={
                            "chunk_id": chunk.chunk_id,
                            "doc_id": chunk.metadata.doc_id,
                            "metadata": chunk.metadata.to_dict()
                        },
                        store_type="memory_chunks",
                        table="memory_chunks"
                    )
            
            if self.config.index_type == IndexType.GRAPH:
                triples = self._generate_triples(corpus)
                for triple in triples:
                    self.storage_handler.storage_graph.upsert_triplet(
                        subject=triple["subject"],
                        relation=triple["relation"],
                        object=triple["object"]
                    )
            
            self.logger.info(f"Inserted {len(documents)} documents and {len(corpus.chunks)} chunks")
            return corpus
        except Exception as e:
            self.logger.error(f"Failed to index documents: {str(e)}")
            raise
    
    def _generate_triples(self, corpus: Corpus) -> List[Dict]:
        triples = []
        for chunk in corpus.chunks:
            if chunk.metadata.custom_fields.get("section_title"):
                triples.append({
                    "subject": chunk.metadata.doc_id,
                    "relation": "has_section",
                    "object": chunk.metadata.custom_fields["section_title"]
                })
            if chunk.metadata.custom_fields.get("entities"):
                for entity in chunk.metadata.custom_fields["entities"]:
                    triples.append({
                        "subject": chunk.metadata.doc_id,
                        "relation": "contains_entity",
                        "object": entity
                    })
        return triples
    
    def configure_retrieval(self, query: Query):
        try:
            self.retrievers = []
            if self.config.index_type in [IndexType.VECTOR, IndexType.SUMMARY, IndexType.TREE]:
                vector_retriever = self.retriever_factory.create(
                    retriever_type=RetrieverType.VECTOR,
                    index=self.index,
                    query=query
                )
                self.retrievers.append(vector_retriever)
            
            if query.use_graph and self.config.index_type == IndexType.GRAPH:
                graph_retriever = self.retriever_factory.create(
                    retriever_type=RetrieverType.GRAPH,
                    graph_store=self.storage_handler.storage_graph,
                    embed_model=self.embed_model,
                    query=query
                )
                self.retrievers.append(graph_retriever)
            
            if not self.retrievers:
                raise RuntimeError("No retrievers configured")
            
            router_retriever = RouterRetriever(
                retrievers=[r.get_retriever() for r in self.retrievers],
                selector=LLMSingleSelector.from_defaults()
            )
            
            postprocessor = self.postprocessor_factory.create(
                postprocessor_type="reranker",
                query=query
            )
            
            self.query_engine = RetrieverQueryEngine(
                retriever=router_retriever,
                node_postprocessors=[postprocessor]
            )
            
            self.logger.info("Configured retrieval with %d retrievers", len(self.retrievers))
        except Exception as e:
            self.logger.error(f"Failed to configure retrieval: {str(e)}")
            raise
    
    def retrieve(self, query: Query) -> SchemaResult:
        if not self.query_engine:
            self.configure_retrieval(query)
        
        try:
            response = self.query_engine.query(query.query_str)
            nodes = response.source_nodes
            corpus = Corpus.from_llama_nodes(nodes)
            scores = [node.score or 0.0 for node in nodes]
            
            for chunk, score in zip(corpus.chunks, scores):
                chunk.metadata.similarity_score = score
            
            result = SchemaResult(
                corpus=corpus,
                scores=scores,
                metadata={"query": query.query_str}
            )
            
            if self.storage_handler.storage_db:
                for chunk in result.corpus.chunks:
                    self.storage_handler.storage_db.insert(
                        metadata={
                            "chunk_id": chunk.chunk_id,
                            "doc_id": chunk.metadata.doc_id,
                            "metadata": {
                                **chunk.metadata.to_dict(),
                                "retrieval_query": query.query_str,
                                "retrieval_score": chunk.metadata.similarity_score
                            }
                        },
                        store_type="retrieval_log",
                        table="retrieval_log"
                    )
            
            self.logger.info(f"Retrieved {len(corpus.chunks)} chunks for query")
            return result
        except Exception as e:
            self.logger.error(f"Retrieval failed: {str(e)}")
            raise
    
    def get_index_data(self) -> Dict[str, Any]:
        try:
            nodes = list(self.index.storage_context.docstore.docs.values())
            corpus = Corpus.from_llama_nodes(nodes)
            
            documents = []
            if self.storage_handler.storage_db:
                for chunk in corpus.chunks:
                    doc_id = chunk.metadata.doc_id
                    doc_metadata = self.storage_handler.storage_db.get_by_id(
                        metadata_id=doc_id,
                        store_type="memory",
                        table="memory"
                    )
                    if doc_metadata and not any(doc.doc_id == doc_id for doc in documents):
                        documents.append(Document(
                            text="",
                            metadata=doc_metadata["metadata"],
                            doc_id=doc_id
                        ))
            
            return {
                "documents": documents,
                "corpus": corpus,
                "embeddings": {chunk.chunk_id: chunk.embedding for chunk in corpus.chunks if chunk.embedding}
            }
        except Exception as e:
            self.logger.error(f"Failed to extract index data: {str(e)}")
            raise