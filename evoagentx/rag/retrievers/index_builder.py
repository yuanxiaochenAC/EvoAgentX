import logging
from typing import List, Dict, Any, Optional

from llama_index.core.indices.base import BaseIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.vector_stores import VectorStore
from llama_index.core.postprocessor import SimilarityPostprocessor, RankGPTRerank
from llama_index.core import VectorStoreIndex, TreeIndex, ListIndex, StorageContext
from llama_index.core.retrievers import VectorIndexRetriever, TreeIndexRetriever, ListIndexRetriever

from .base import BaseIndexBuilder
from ..schema import Corpus, Document, IndexType
from evoagentx.storages.base import StorageHandler


class IndexBuilder(BaseIndexBuilder):
    """Flexible index builder for constructing and querying LlamaIndex-based indices."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        embed_model: BaseEmbedding,
        storage_handler: Optional[StorageHandler] = None,
        index_type: str = IndexType.VECTOR.value,
        postprocessors: Optional[List[str]] = None,
        postprocessor_config: Optional[Dict[str, Any]] = None
    ):
        self.vector_store = vector_store
        self.embed_model = embed_model
        self.storage_handler = storage_handler
        self.storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index_type = IndexType(index_type)
        self.postprocessors = postprocessors or []
        self.postprocessor_config = postprocessor_config or {}
        self.index = None
        self.logger = logging.getLogger(__name__)
    
    def _build_index(self) -> BaseIndex:
        """Create a LlamaIndex index based on type."""
        if self.index_type == IndexType.VECTOR:
            return VectorStoreIndex(
                nodes=[],
                embed_model=self.embed_model,
                storage_context=self.storage_context
            )
        elif self.index_type == IndexType.TREE:
            return TreeIndex(
                nodes=[],
                embed_model=self.embed_model,
                storage_context=self.storage_context
            )
        elif self.index_type == IndexType.LIST:
            return ListIndex(
                nodes=[],
                embed_model=self.embed_model,
                storage_context=self.storage_context
            )
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
    
    def build_index(self, corpus: Corpus, documents: List[Document]) -> BaseIndex:
        """Build an index from a corpus and documents."""
        try:
            self.index = self._build_index()
            nodes = corpus.to_llama_nodes()
            self.index.insert_nodes(nodes)
            
            # Store metadata in SQLite
            if self.storage_handler:
                for doc in documents:
                    self.storage_handler.storageDB.insert(
                        metadata=doc.metadata.model_dump(),
                        store_type="memory",
                        table="memory"
                    )
                for chunk in corpus.chunks:
                    self.storage_handler.storageDB.insert(
                        metadata=chunk.metadata.model_dump(),
                        store_type="chunk",
                        table="memory_chunks"
                    )
            
            self.logger.info(f"Indexed {len(documents)} documents with {len(corpus.chunks)} chunks")
            return self.index
        except Exception as e:
            self.logger.error(f"Failed to build index: {str(e)}")
            raise
    
    def _get_postprocessors(self) -> List[Any]:
        """Initialize postprocessors for retrieval."""
        postprocessors = []
        for pp in self.postprocessors:
            if pp == "similarity":
                postprocessors.append(
                    SimilarityPostprocessor(
                        similarity_cutoff=self.postprocessor_config.get("similarity_cutoff", 0.7)
                    )
                )
            elif pp == "rankgpt":
                postprocessors.append(
                    RankGPTRerank(
                        top_n=self.postprocessor_config.get("rankgpt_top_n", 3),
                        llm=self.postprocessor_config.get("llm")
                    )
                )
            else:
                self.logger.warning(f"Unsupported postprocessor: {pp}")
        return postprocessors
    
    def retrieve(self, query: str, top_k: int = 5) -> Corpus:
        """Retrieve chunks for a query with optional postprocessing."""
        try:
            if self.index_type == IndexType.VECTOR:
                retriever = VectorIndexRetriever(
                    index=self.index,
                    similarity_top_k=top_k
                )
            elif self.index_type == IndexType.TREE:
                retriever = TreeIndexRetriever(
                    index=self.index,
                    similarity_top_k=top_k
                )
            elif self.index_type == IndexType.LIST:
                retriever = ListIndexRetriever(
                    index=self.index,
                    similarity_top_k=top_k
                )
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
            
            nodes = retriever.retrieve(query)
            postprocessors = self._get_postprocessors()
            if postprocessors:
                for pp in postprocessors:
                    nodes = pp.postprocess_nodes(nodes)
            
            corpus = Corpus.from_llama_nodes(nodes)
            for chunk, node in zip(corpus.chunks, nodes):
                chunk.metadata.similarity_score = node.score
            
            self.logger.info(f"Retrieved {len(corpus.chunks)} chunks for query")
            return corpus
        except Exception as e:
            self.logger.error(f"Retrieval failed: {str(e)}")
            raise
    
    def get_index_data(self) -> Dict[str, Any]:
        """Extract index data for external use."""
        try:
            nodes = self.index.docstore.docs.values()
            corpus = Corpus.from_llama_nodes(nodes)
            
            documents = {}
            for chunk in corpus.chunks:
                doc_id = chunk.metadata.doc_id
                if doc_id not in documents and self.storage_handler:
                    doc_metadata = self.storage_handler.storageDB.get_by_id(
                        metadata_id=doc_id,
                        store_type="memory",
                        table="memory"
                    )
                    if doc_metadata:
                        documents[doc_id] = Document(
                            text="",  # Text may need reconstruction
                            metadata=doc_metadata,
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
        """Persist the index to disk."""
        try:
            self.index.storage_context.persist(persist_dir=persist_dir)
            self.logger.info(f"Persisted index to {persist_dir}")
        except Exception as e:
            self.logger.error(f"Failed to persist index: {str(e)}")
            raise