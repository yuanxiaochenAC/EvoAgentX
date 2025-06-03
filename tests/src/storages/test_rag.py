import os
import logging
import tempfile
import unittest
from uuid import uuid4
from unittest.mock import patch, MagicMock

# import jieba

from evoagentx.rag.rag_config import RAGConfig
from evoagentx.rag.search_engine import SearchEngine
from evoagentx.rag.schema import Document, Corpus, RagQuery, RagResult, Chunk, DocumentMetadata, ChunkMetadata
from evoagentx.rag.embeddings.base import EmbeddingProvider
from evoagentx.storages.base import StorageHandler
from evoagentx.rag.retrievers import RetrieverType
from evoagentx.rag.indexings import IndexFactory
from evoagentx.rag.indexings.base import IndexType
from evoagentx.rag.chunkers import ChunkFactory
from evoagentx.rag.chunkers.base import ChunkingStrategy
from evoagentx.rag.retrievers.vector_retriever import VectorRetriever
from evoagentx.rag.retrievers.graph_retriever import GraphRetriever
from evoagentx.rag.postprocessors.simple_reranker import SimpleReranker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import dotenv
dotenv.load_dotenv()
ENV = os.environ


class TestRagEngine(unittest.TestCase):
    def setUp(self):
        """Set up test environment and SearchEngine instance."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_storage.db")

        # Sample configuration
        self.config = RAGConfig(
            embedding_provider=EmbeddingProvider.OPENAI,
            embedding_config={"model_name": "text-embedding-ada-002", "api_key": ENV["OPENAI_API_KEY"]},
            index_type=IndexType.VECTOR,
            chunking_strategy=ChunkingStrategy.SIMPLE,
            node_parser_config={"chunk_size": 100, "chunk_overlap": 20},
            vector_store_type="faiss",
            vector_store_config={"dimension": 1536},
            retrieval_config={"top_k": 3, "similarity_cutoff": 0.7}
        )

        # Initialize real storage components
        faiss_index = faiss.IndexFlatL2(1536)  # Dimension matches OpenAI embeddings
        self.vector_store = FaissVectorStore(faiss_index=faiss_index)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        # Initialize SQLite storage
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                text TEXT,
                metadata TEXT
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT,
                text TEXT,
                metadata TEXT,
                embedding BLOB
            )
        """)
        self.conn.commit()

        # Custom StorageHandler for testing
        class TestStorageHandler(StorageHandler):
            def __init__(self, conn, storage_context):
                self.conn = conn
                self.cursor = conn.cursor()
                self.storage_context = storage_context
                self.storage_db = self  # Self-reference for simplicity

            def insert(self, obj):
                if isinstance(obj, Document):
                    self.cursor.execute(
                        "INSERT OR REPLACE INTO documents (doc_id, text, metadata) VALUES (?, ?, ?)",
                        (obj.doc_id, obj.text, obj.metadata.to_json())
                    )
                elif isinstance(obj, Chunk):
                    self.cursor.execute(
                        "INSERT OR REPLACE INTO chunks (chunk_id, doc_id, text, metadata, embedding) VALUES (?, ?, ?, ?, ?)",
                        (obj.chunk_id, obj.metadata.doc_id, obj.text, obj.metadata.to_json(), str(obj.embedding))
                    )
                self.conn.commit()

            def get_by_id(self, obj_id, obj_type):
                if obj_type == "document":
                    self.cursor.execute("SELECT * FROM documents WHERE doc_id = ?", (obj_id,))
                elif obj_type == "chunk":
                    self.cursor.execute("SELECT * FROM chunks WHERE chunk_id = ?", (obj_id,))
                return self.cursor.fetchone()

            def delete(self, obj_id, obj_type):
                if obj_type == "document":
                    self.cursor.execute("DELETE FROM documents WHERE doc_id = ?", (obj_id,))
                    self.cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (obj_id,))
                elif obj_type == "chunk":
                    self.cursor.execute("DELETE FROM chunks WHERE chunk_id = ?", (obj_id,))
                self.conn.commit()

        self.storage_handler = TestStorageHandler(self.conn, self.storage_context)

        # Initialize SearchEngine
        self.search_engine = SearchEngine(config=self.config, storage_handler=self.storage_handler)

        # Sample test data
        self.sample_doc_en = Document(
            text="This is a sample English document about AI.",
            doc_id=str(uuid4()),
            metadata=DocumentMetadata(custom_fields={"section_title": "AI Overview"})
        )
        self.sample_doc_zh = Document(
            text="这是一个关于人工智能的中文样本文档。",
            doc_id=str(uuid4()),
            metadata=DocumentMetadata(custom_fields={"section_title": "人工智能概览"})
        )
        self.sample_tool_doc = Document(
            text="Tool for data analysis.",
            doc_id=str(uuid4()),
            metadata=DocumentMetadata(custom_fields={"tool_id": "tool_123"})
        )
        self.sample_query = RagQuery(
            query_str="What is AI?",
            top_k=3,
            similarity_cutoff=0.7,
            keyword_filters=["AI"],
            use_graph=False,  # Disable graph retrieval for testing
            metadata_filters={"tool_id": "tool_123"}
        )

    def tearDown(self):
        """Clean up test environment."""
        self.conn.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_add_documents(self):
        """Test adding documents to indices."""
        documents = [self.sample_doc_en, self.sample_doc_zh]
        corpus = self.search_engine.add_documents(documents)
        self.assertGreaterEqual(len(corpus.chunks), 2)  # At least one chunk per document
        self.cursor.execute("SELECT COUNT(*) FROM documents")
        self.assertEqual(self.cursor.fetchone()[0], 2)  # 2 documents stored
        self.cursor.execute("SELECT COUNT(*) FROM chunks")
        self.assertGreaterEqual(self.cursor.fetchone()[0], 2)  # At least 2 chunks

    def test_add_documents_specific_index(self):
        """Test adding documents to a specific index."""
        documents = [self.sample_doc_en]
        corpus = self.search_engine.add_documents(documents, index_type=IndexType.VECTOR)
        self.assertGreaterEqual(len(corpus.chunks), 1)
        self.cursor.execute("SELECT COUNT(*) FROM documents")
        self.assertEqual(self.cursor.fetchone()[0], 1)
        self.cursor.execute("SELECT COUNT(*) FROM chunks")
        self.assertGreaterEqual(self.cursor.fetchone()[0], 1)

    def test_add_documents_error(self):
        """Test error handling for add_documents."""
        with self.assertRaises(ValueError):
            self.search_engine.add_documents([])  # Empty list should raise error

    def test_update_documents(self):
        """Test updating existing documents."""
        # First, add a document
        self.search_engine.add_documents([self.sample_doc_en])
        updated_doc = Document(
            doc_id=self.sample_doc_en.doc_id,
            text="Updated English document about AI.",
            metadata=DocumentMetadata(custom_fields={"section_title": "AI Updated"})
        )
        corpus = self.search_engine.update_documents([updated_doc])
        self.assertGreaterEqual(len(corpus.chunks), 1)
        self.cursor.execute("SELECT text FROM documents WHERE doc_id = ?", (self.sample_doc_en.doc_id,))
        self.assertEqual(self.cursor.fetchone()[0], "Updated English document about AI.")

    def test_update_documents_no_existing(self):
        """Test updating non-existent documents."""
        updated_doc = Document(
            doc_id=str(uuid4()),
            text="New document.",
            metadata=DocumentMetadata(custom_fields={})
        )
        corpus = self.search_engine.update_documents([updated_doc])
        self.assertGreaterEqual(len(corpus.chunks), 1)
        self.cursor.execute("SELECT COUNT(*) FROM documents")
        self.assertEqual(self.cursor.fetchone()[0], 1)

    def test_build_index(self):
        """Test building a new index."""
        self.search_engine.build_index(IndexType.VECTOR)
        self.assertIn(IndexType.VECTOR, self.search_engine.indices)
        self.assertIn(IndexType.VECTOR, self.search_engine.retrievers)

    def test_retrieve(self):
        """Test vector-based retrieval."""
        # Add documents first
        documents = [self.sample_doc_en, self.sample_doc_zh, self.sample_tool_doc]
        self.search_engine.add_documents(documents)
        result = self.search_engine.retrieve(self.sample_query)
        self.assertGreaterEqual(len(result.corpus.chunks), 1)
        self.assertEqual(len(result.scores), len(result.corpus.chunks))
        self.assertEqual(result.metadata["query"], "What is AI?")

    def test_retrieve_vector_only(self):
        """Test retrieval with vector index only."""
        documents = [self.sample_doc_en]
        self.search_engine.add_documents(documents)
        query = RagQuery(
            query_str="What is AI?",
            top_k=3,
            use_graph=False
        )
        result = self.search_engine.retrieve(query)
        self.assertGreaterEqual(len(result.corpus.chunks), 1)

    def test_retrieve_with_metadata_filters(self):
        """Test retrieval with metadata filters for Agentic RAG."""
        documents = [self.sample_tool_doc]
        self.search_engine.add_documents(documents)
        result = self.search_engine.retrieve(self.sample_query)
        self.assertTrue(any(chunk.metadata.custom_fields.get("tool_id") == "tool_123" for chunk in result.corpus.chunks))

    def test_workflow(self):
        """Test complete SearchEngine workflow."""
        # Add documents
        documents = [self.sample_doc_en, self.sample_doc_zh, self.sample_tool_doc]
        corpus = self.search_engine.add_documents(documents)
        self.assertGreaterEqual(len(corpus.chunks), 3)

        # Update a document
        updated_doc = Document(
            doc_id=self.sample_doc_en.doc_id,
            text="Updated AI document.",
            metadata=DocumentMetadata(custom_fields={"section_title": "AI Updated"})
        )
        updated_corpus = self.search_engine.update_documents([updated_doc])
        self.assertGreaterEqual(len(updated_corpus.chunks), 1)

        # Build a new index
        self.search_engine.build_index(IndexType.VECTOR)
        self.assertIn(IndexType.VECTOR, self.search_engine.indices)

        # Retrieve
        result = self.search_engine.retrieve(self.sample_query)
        self.assertGreaterEqual(len(result.corpus.chunks), 1)
        self.assertTrue(any(chunk.metadata.custom_fields.get("tool_id") == "tool_123" for chunk in result.corpus.chunks))

    def test_empty_documents(self):
        """Test adding empty document list."""
        with self.assertRaises(ValueError):
            self.search_engine.add_documents([])

if __name__ == "__main__":
    unittest.main()