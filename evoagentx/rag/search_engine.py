
from typing import List, Optional, Union, Dict, Any, Callable
from enum import Enum
from llama_index.core import (
    VectorStoreIndex, TreeIndex, ListIndex, StorageContext, SimpleDirectoryReader,
    SentenceSplitter, TokenTextSplitter, SemanticSplitter, BaseEmbedding, LLM
)
from llama_index.core.vector_stores import VectorStore, BasePydanticVectorStore
from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import NodeParser
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.embeddings import OpenAIEmbedding, HuggingFaceEmbedding
from schema import Document, Chunk, Corpus, DocumentMetadata, ChunkMetadata

# Enum for chunking strategies
class ChunkingStrategy(str, Enum):
    SIMPLE = "simple"
    TOKEN = "token"
    SEMANTIC = "semantic"

# Enum for index types
class IndexType(str, Enum):
    VECTOR = "vector"
    TREE = "tree"
    LIST = "list"

# Enum for embedding providers
class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"

class SearchEngine:
    """SearchEngine for integrating LlamaIndex with the framework's Agent."""

    def __init__(
        self,
        vector_store: BasePydanticVectorStore,
        embedding_provider: EmbeddingProvider = EmbeddingProvider.OPENAI,
        embedding_model: Optional[BaseEmbedding] = None,
        llm: Optional[LLM] = None,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.SIMPLE,
        chunk_size: int = 1024,
        chunk_overlap: int = 20,
        index_type: IndexType = IndexType.VECTOR,
        sqlite_handler: Optional[Any] = None,
        storage_context: Optional[StorageContext] = None,
        pre_retrieval_transform: Optional[Callable[[str], str]] = None,
        post_retrieval_processor: Optional[Callable[[List[Chunk]], List[Chunk]]] = None,
    ):
        """Initialize the middleware with configurable components."""
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.llm = llm
        self.sqlite_handler = sqlite_handler
        self.storage_context = storage_context or StorageContext.from_defaults(
            vector_store=vector_store
        )
        self.pre_retrieval_transform = pre_retrieval_transform or (lambda x: x)
        self.post_retrieval_processor = post_retrieval_processor or (lambda x: x)

        # Initialize embedding model
        self.embed_model = embedding_model or self._get_embedding_model()
        
        # Initialize chunking strategy
        self.text_splitter = self._get_text_splitter(
            chunking_strategy, chunk_size, chunk_overlap
        )

        # Initialize index
        self.index = self._get_index(index_type)
        self.corpus = Corpus()

    def _get_embedding_model(self) -> BaseEmbedding:
        """Select embedding model based on provider."""
        if self.embedding_provider == EmbeddingProvider.OPENAI:
            return OpenAIEmbedding()
        elif self.embedding_provider == EmbeddingProvider.HUGGINGFACE:
            return HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
        elif self.embedding_provider == EmbeddingProvider.CUSTOM:
            raise ValueError("Custom embedding model must be provided.")
        else:
            raise ValueError(f"Unsupported embedding provider: {self.embedding_provider}")

    def _get_text_splitter(
        self, strategy: ChunkingStrategy, chunk_size: int, chunk_overlap: int
    ) -> NodeParser:
        """Select text splitter based on chunking strategy."""
        if strategy == ChunkingStrategy.SIMPLE:
            return SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif strategy == ChunkingStrategy.TOKEN:
            return TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif strategy == ChunkingStrategy.SEMANTIC:
            return SemanticSplitter(embed_model=self.embed_model)
        else:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")

    def _get_index(self, index_type: IndexType) -> Any:
        """Select index type for retrieval."""
        if index_type == IndexType.VECTOR:
            return VectorStoreIndex(
                nodes=[], embed_model=self.embed_model, storage_context=self.storage_context
            )
        elif index_type == IndexType.TREE:
            return TreeIndex(
                nodes=[], embed_model=self.embed_model, storage_context=self.storage_context
            )
        elif index_type == IndexType.LIST:
            return ListIndex(
                nodes=[], embed_model=self.embed_model, storage_context=self.storage_context
            )
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

    def load_documents(self, directory: str, file_types: Optional[List[str]] = None) -> List[Document]:
        """Load documents from a directory."""
        reader = SimpleDirectoryReader(directory, file_extensions=file_types)
        llama_docs = reader.load_data()
        documents = [Document.from_llama_document(doc) for doc in llama_docs]
        for doc in documents:
            doc.metadata.hash_doc = doc.compute_hash()
            if self.sqlite_handler:
                self.sqlite_handler.store_metadata(doc.doc_id, doc.metadata.model_dump())
        return documents

    def chunk_document(self, document: Document) -> Corpus:
        """Chunk a document into a Corpus of Chunks."""
        llama_doc = document.to_llama_document()
        nodes = self.text_splitter.split_text_with_metadata(llama_doc)
        corpus = Corpus.from_llama_nodes(nodes)
        for chunk in corpus.chunks:
            chunk.metadata.doc_id = document.doc_id
            chunk.metadata.chunking_strategy = self.text_splitter.__class__.__name__
            if self.sqlite_handler:
                self.sqlite_handler.store_chunk_metadata(
                    chunk.chunk_id, chunk.metadata.model_dump()
                )
        return corpus

    def index_corpus(self, corpus: Corpus):
        """Index a Corpus into the selected index type."""
        nodes = corpus.to_llama_nodes()
        self.index.insert_nodes(nodes)
        self.corpus.add_chunk(corpus.chunks)
        if self.sqlite_handler:
            for chunk in corpus.chunks:
                self.sqlite_handler.store_chunk(chunk.to_dict())

    def retrieve(self, query: str, top_k: int = 5) -> Corpus:
        """Retrieve relevant chunks for a query with pre- and post-processing."""
        # Pre-retrieval transformation
        transformed_query = self.pre_retrieval_transform(query)
        
        # Retrieval
        retriever = VectorIndexRetriever(index=self.index, similarity_top_k=top_k)
        nodes = retriever.retrieve(transformed_query)
        
        # Convert to Corpus and add similarity scores
        corpus = Corpus.from_llama_nodes(nodes)
        for chunk, node in zip(corpus.chunks, nodes):
            chunk.metadata.similarity_score = node.score
        
        # Post-retrieval processing
        corpus.chunks = self.post_retrieval_processor(corpus.chunks)
        return corpus

    def query(self, query: str, top_k: int = 5) -> str:
        """Query the index and generate a response using LLM."""
        if not self.llm:
            raise ValueError("LLM not provided for query generation.")
        query_engine = RetrieverQueryEngine(
            retriever=VectorIndexRetriever(index=self.index, similarity_top_k=top_k),
            llm=self.llm,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
        )
        response = query_engine.query(self.pre_retrieval_transform(query))
        return str(response)

    def delete_document(self, doc_id: str):
        """Delete a document and its chunks."""
        self.index.delete_ref_doc(doc_id, delete_from_docstore=True)
        if self.sqlite_handler:
            self.sqlite_handler.delete_document(doc_id)
        self.corpus = Corpus(
            [chunk for chunk in self.corpus.chunks if chunk.metadata.doc_id != doc_id]
        )

    def persist(self, persist_dir: str):
        """Persist the index to disk."""
        self.index.storage_context.persist(persist_dir=persist_dir)

    def get_stats(self) -> Dict:
        """Get statistics about the corpus."""
        return self.corpus.get_stats()