import logging
from typing import List, Dict, Any, Optional
from llama_index.core import StorageContext, BaseRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor, KeywordNodePostprocessor

from .rag_config import RAGConfig
from .schema import Corpus, Document, Chunk, ChunkingStrategy
from .embeddings.base import EmbeddingFactory, EmbeddingProvider
from .retrievers.base import IndexFactory, IndexType
from evoagentx.storages.base import StorageHandler
from .chunkers import SimpleChunker, SemanticChunker, HierarchicalChunker


class SearchEngine:
    def __init__(self, config: RAGConfig, storage_handler: StorageHandler):
        self.config = config
        self.storage_handler = storage_handler
        self.embedding_factory = EmbeddingFactory()
        self.index_factory = IndexFactory()
        self.embed_model = self.embedding_factory.create(
            provider=config.embedding_provider,
            model_config=config.embedding_config
        )
        self.chunker = self._get_chunker()
        self.index = self.index_factory.create(
            index_type=config.index_type,
            embed_model=self.embed_model,
            node_parser=None,  # Handled by chunker
            storage_context=self.storage_handler.storage_context,
            index_config=config.index_config
        )
        self.retriever = None
        self.query_engine = None
        self.logger = logging.getLogger(__name__)
    
    def _get_chunker(self):
        if self.config.chunking_strategy == ChunkingStrategy.SIMPLE:
            return SimpleChunker(
                chunk_size=self.config.node_parser_config.get("chunk_size", 1024),
                chunk_overlap=self.config.node_parser_config.get("chunk_overlap", 20)
            )
        elif self.config.chunking_strategy == ChunkingStrategy.SEMANTIC:
            return SemanticChunker(
                embed_model=self.embed_model,
                similarity_threshold=self.config.node_parser_config.get("similarity_threshold", 0.7)
            )
        elif self.config.chunking_strategy == ChunkingStrategy.HIERARCHICAL:
            return HierarchicalChunker(
                chunk_sizes=self.config.node_parser_config.get("chunk_sizes", [2048, 512, 128]),
                chunk_overlap=self.config.node_parser_config.get("chunk_overlap", 20)
            )
        else:
            raise ValueError(f"Unsupported chunking strategy: {self.config.chunking_strategy}")
    
    def add_documents(self, documents: List[Document]):
        try:
            corpus = self.chunker.chunk(documents)
            nodes = corpus.to_llama_nodes()
            self.index.insert_nodes(nodes)
            
            if self.storage_handler.storage_db:
                for doc in documents:
                    self.storage_handler.storage_db.insert(
                        metadata={"memory_id": doc.doc_id, "doc_id": doc.doc_id, "metadata": doc.metadata.model_dump()},
                        store_type="memory",
                        table="memory"
                    )
                for chunk in corpus.chunks:
                    self.storage_handler.storage_db.insert(
                        metadata={"chunk_id": chunk.chunk_id, "doc_id": chunk.metadata.doc_id, "metadata": chunk.metadata.model_dump()},
                        store_type="memory_chunks",
                        table="memory_chunks"
                    )
            
            if self.index_type == IndexType.GRAPH:
                triples = self._generate_triples(corpus)
                self.storage_handler.storage_graph.add_triples(triples)
            
            self.logger.info(f"Indexed {len(documents)} documents and {len(corpus.chunks)} chunks")
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
        return triples
    
    def configure_retrieval(
        self,
        top_k: int = 5,
        similarity_cutoff: Optional[float] = None,
        keyword_filters: Optional[List[str]] = None,
        use_graph: bool = False
    ):
        try:
            retrievers = []
            if self.config.index_type in [IndexType.VECTOR, IndexType.SUMMARY, IndexType.TREE]:
                retrievers.append(VectorIndexRetriever(
                    index=self.index,
                    similarity_top_k=top_k
                ))
            if use_graph and self.config.index_type == IndexType.GRAPH:
                retrievers.append(self._create_graph_retriever(top_k))
            
            self.retriever = self._combine_retrievers(retrievers)
            
            node_postprocessors = []
            if similarity_cutoff:
                node_postprocessors.append(SimilarityPostprocessor(similarity_cutoff=similarity_cutoff))
            if keyword_filters:
                node_postprocessors.append(KeywordNodePostprocessor(required_keywords=keyword_filters))
            
            self.query_engine = RetrieverQueryEngine(
                retriever=self.retriever,
                node_postprocessors=node_postprocessors
            )
            
            self.logger.info("Configured retriever and query engine")
        except Exception as e:
            self.logger.error(f"Failed to configure retrieval: {str(e)}")
            raise
    
    def _create_graph_retriever(self, top_k: int) -> BaseRetriever:
        from llama_index.core.indices.property_graph import VectorContextRetriever
        return VectorContextRetriever(
            graph_store=self.storage_handler.storage_graph.get_graph_store(),
            embed_model=self.embed_model,
            similarity_top_k=top_k
        )
    
    def _combine_retrievers(self, retrievers: List[BaseRetriever]) -> BaseRetriever:
        from llama_index.core.retrievers import RouterRetriever
        return RouterRetriever(retrievers=retrievers)
    
    def retrieve(self, query: str) -> Corpus:
        if not self.query_engine:
            raise ValueError("Query engine not configured. Call configure_retrieval first.")
        
        try:
            response = self.query_engine.query(query)
            nodes = response.source_nodes
            corpus = Corpus.from_llama_nodes(nodes)
            for chunk, node in zip(corpus.chunks, nodes):
                chunk.metadata.similarity_score = node.score
            
            self.logger.info(f"Retrieved {len(corpus.chunks)} chunks for query")
            return corpus
        except Exception as e:
            self.logger.error(f"Retrieval failed: {str(e)}")
            raise
    
    def get_index_data(self) -> Dict[str, Any]:
        try:
            nodes = self.index.docstore.docs.values()
            corpus = Corpus.from_llama_nodes(nodes)
            
            documents = {}
            if self.storage_handler.storage_db:
                for chunk in corpus.chunks:
                    doc_id = chunk.metadata.doc_id
                    if doc_id not in documents:
                        doc_metadata = self.storage_handler.storage_db.get_by_id(
                            metadata_id=doc_id,
                            store_type="memory",
                            table="memory"
                        )
                        if doc_metadata:
                            documents[doc_id] = Document(
                                text="",
                                metadata=doc_metadata["metadata"],
                                doc_id=doc_id
                            )
            
            return {
                "documents": list(documents.values()),
                "corpus": corpus,
                "embeddings": {chunk.chunk_id: chunk.embedding for chunk in corpus.chunks if chunk.embedding}
            }
        except Exception as e:
            self.logger.error(f"Failed to extract index data: {str(e)}")
            raise
    
    def persist(self, persist_dir: str):
        self.storage_handler.persist(persist_dir)