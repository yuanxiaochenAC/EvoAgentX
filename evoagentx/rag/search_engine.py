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
        self.logger = logging.getLogger(__name__)
        
        # Initialize embedding model
        self.embed_model = self.embedding_factory.create(
            provider=config.embedding_provider,
            model_config=config.embedding_config
        )
        
        # Initialize chunker
        self.chunker = self.chunk_factory.create(
            strategy=config.chunking_strategy,
            embed_model=self.embed_model,
            chunker_config=config.node_parser_config
        )
        
        # Initialize indices and retrievers
        self.indices = {}
        self.retrievers = {}
        self._initialize_indices_and_retrievers()
    
    def _initialize_indices_and_retrievers(self):
        """Initialize indices and their corresponding retrievers."""
        try:
            vector_index = self.index_factory.create(
                index_type=IndexType.VECTOR,
                embed_model=self.embed_model,
                storage_context=self.storage_handler.storage_context,
                index_config=self.config.index_config
            )
            self.indices[IndexType.VECTOR] = vector_index
            self.retrievers[IndexType.VECTOR] = self.retriever_factory.create(
                retriever_type=RetrieverType.VECTOR,
                index=vector_index.get_index(),
                query=Query(query_str="", top_k=self.config.retrieval_config.get("top_k", 5))
            )
            
            # Create additional indices based on config
            if self.config.index_type == IndexType.GRAPH and self.storage_handler.storage_context.graph_store:
                graph_index = self.index_factory.create(
                    index_type=IndexType.GRAPH,
                    embed_model=self.embed_model,
                    storage_context=self.storage_handler.storage_context,
                    index_config=self.config.index_config
                )
                self.indices[IndexType.GRAPH] = graph_index
                self.retrievers[IndexType.GRAPH] = self.retriever_factory.create(
                    retriever_type=RetrieverType.GRAPH,
                    graph_store=self.storage_handler.storage_context.graph_store,
                    embed_model=self.embed_model,
                    query=Query(query_str="", top_k=self.config.retrieval_config.get("top_k", 5))
                )
            
            if self.config.index_type == IndexType.SUMMARY:
                summary_index = self.index_factory.create(
                    index_type=IndexType.SUMMARY,
                    embed_model=self.embed_model,
                    storage_context=self.storage_handler.storage_context,
                    index_config=self.config.index_config
                )
                self.indices[IndexType.SUMMARY] = summary_index
                self.retrievers[IndexType.SUMMARY] = self.retriever_factory.create(
                    retriever_type=RetrieverType.VECTOR,  # Summary uses vector retriever
                    index=summary_index.get_index(),
                    query=Query(query_str="", top_k=self.config.retrieval_config.get("top_k", 5))
                )
            
            if self.config.index_type == IndexType.TREE:
                tree_index = self.index_factory.create(
                    index_type=IndexType.TREE,
                    embed_model=self.embed_model,
                    storage_context=self.storage_handler.storage_context,
                    index_config=self.config.index_config
                )
                self.indices[IndexType.TREE] = tree_index
                self.retrievers[IndexType.TREE] = self.retriever_factory.create(
                    retriever_type=RetrieverType.VECTOR,  # Tree uses vector retriever
                    index=tree_index.get_index(),
                    query=Query(query_str="", top_k=self.config.retrieval_config.get("top_k", 5))
                )
            
            self.logger.info(f"Initialized {len(self.indices)} indices and retrievers")
        except Exception as e:
            self.logger.error(f"Failed to initialize indices: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Document], index_type: Optional[IndexType] = None) -> Corpus:
        """Insert documents into specified or all indices.
        
        Args:
            documents: List of documents to insert.
            index_type: Specific index type to insert into (default: all indices).
            
        Returns:
            Corpus: The chunked corpus.
        """
        try:
            corpus = self.chunker.chunk(documents)
            nodes = corpus.to_llama_nodes()
            
            # Insert into specified or all indices
            target_indices = [self.indices[index_type]] if index_type else self.indices.values()
            for index in target_indices:
                index.insert_nodes(nodes)
            
            # Store metadata in database
            if self.storage_handler.storage_db:
                for doc in documents:
                    self.storage_handler.storage_db.insert(
                        metadata={
                            "memory_id": doc.doc_id,
                            "doc_id": doc.doc_id,
                            "metadata": doc.metadata.to_dict()
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
            
            # Generate and store graph triples if graph index is active
            if IndexType.GRAPH in self.indices:
                triples = self._generate_triples(corpus)
                for triple in triples:
                    self.storage_handler.storage_context.graph_store.upsert_triplet(
                        subject=triple["subject"],
                        relation=triple["relation"],
                        object_=triple["object"]
                    )
            
            self.logger.info(f"Inserted {len(documents)} documents and {len(corpus.chunks)} chunks")
            return corpus
        except Exception as e:
            self.logger.error(f"Failed to insert documents: {str(e)}")
            raise
    
    def update_documents(self, documents: List[Document], index_type: Optional[IndexType] = None) -> Corpus:
        """Update existing documents in specified or all indices.
        
        Args:
            documents: List of documents to update.
            index_type: Specific index type to update (default: all indices).
            
        Returns:
            Corpus: The updated chunked corpus.
        """
        try:
            # Re-chunk documents
            corpus = self.chunker.chunk(documents)
            nodes = corpus.to_llama_nodes()
            
            # Delete existing chunks
            if self.storage_handler.storage_db:
                for doc in documents:
                    self.storage_handler.storage_db.delete(
                        metadata_id=doc.doc_id,
                        store_type="memory",
                        table="memory"
                    )
                    chunks = self.storage_handler.storage_db.get_by_id(
                        metadata_id=doc.doc_id,
                        store_type="memory_chunks",
                        table="memory_chunks"
                    )
                    if chunks:
                        self.storage_handler.storage_db.delete(
                            metadata_id=chunks["chunk_id"],
                            store_type="memory_chunks",
                            table="memory_chunks"
                        )
            
            # Re-insert nodes
            target_indices = [self.indices[index_type]] if index_type else self.indices.values()
            for index in target_indices:
                index.insert_nodes(nodes)
            
            # Update database
            if self.storage_handler.storage_db:
                for doc in documents:
                    self.storage_handler.storage_db.insert(
                        metadata={
                            "memory_id": doc.doc_id,
                            "doc_id": doc.doc_id,
                            "metadata": doc.metadata.to_dict()
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
            
            # Update graph triples
            if IndexType.GRAPH in self.indices:
                triples = self._generate_triples(corpus)
                for triple in triples:
                    self.storage_handler.storage_context.graph_store.upsert_triplet(
                        subject=triple["subject"],
                        relation=triple["relation"],
                        object_=triple["object"]
                    )
            
            self.logger.info(f"Updated {len(documents)} documents and {len(corpus.chunks)} chunks")
            return corpus
        except Exception as e:
            self.logger.error(f"Failed to update documents: {str(e)}")
            raise
    
    def _generate_triples(self, corpus: Corpus) -> List[Dict]:
        """Generate knowledge graph triples from corpus."""
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
            if chunk.metadata.custom_fields.get("tool_id"):
                triples.append({
                    "subject": chunk.metadata.doc_id,
                    "relation": "is_tool",
                    "object": chunk.metadata.custom_fields["tool_id"]
                })
        return triples
    
    def retrieve(self, query: Query) -> SchemaResult:
        """Retrieve results from multiple indices.
        
        Args:
            query: Query object specifying retrieval parameters.
            
        Returns:
            SchemaResult: Combined retrieval results.
        """
        try:
            results = []
            
            # Retrieve from relevant indices
            target_indices = self.retrievers.keys() if query.use_graph else [IndexType.VECTOR]
            for index_type in target_indices:
                retriever = self.retrievers.get(index_type)
                if retriever:
                    result = retriever.retrieve(query)
                    results.append(result)
            
            # Post-process results
            postprocessor = self.postprocessor_factory.create(
                postprocessor_type="reranker",
                query=query
            )
            final_result = postprocessor.postprocess(query, results)
            
            # Store retrieval log
            if self.storage_handler.storage_db:
                for chunk in final_result.corpus.chunks:
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
            
            self.logger.info(f"Retrieved {len(final_result.corpus.chunks)} chunks for query")
            return final_result
        except Exception as e:
            self.logger.error(f"Retrieval failed: {str(e)}")
            raise
    
    def build_index(self, index_type: IndexType, index_config: Optional[Dict[str, Any]] = None):
        """Build a new index of the specified type.
        
        Args:
            index_type: Type of index to build.
            index_config: Optional configuration for the index.
        """
        try:
            if index_type in self.indices:
                self.logger.warning(f"Index {index_type} already exists, overwriting")
            
            index = self.index_factory.create(
                index_type=index_type,
                embed_model=self.embed_model,
                storage_context=self.storage_handler.storage_context,
                index_config=index_config or self.config.index_config
            )
            self.indices[index_type] = index
            
            # Initialize retriever
            retriever_type = RetrieverType.GRAPH if index_type == IndexType.GRAPH else RetrieverType.VECTOR
            retriever = self.retriever_factory.create(
                retriever_type=retriever_type,
                index=index.get_index() if retriever_type == RetrieverType.VECTOR else None,
                graph_store=self.storage_handler.storage_context.graph_store if retriever_type == RetrieverType.GRAPH else None,
                embed_model=self.embed_model if retriever_type == RetrieverType.GRAPH else None,
                query=Query(query_str="", top_k=self.config.retrieval_config.get("top_k", 5))
            )
            self.retrievers[index_type] = retriever
            
            self.logger.info(f"Built new index: {index_type}")
        except Exception as e:
            self.logger.error(f"Failed to build index {index_type}: {str(e)}")
            raise