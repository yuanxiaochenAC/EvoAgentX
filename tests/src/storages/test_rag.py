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
        """Set up test environment, mocks, and RagEngine instance."""
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
            graph_store_type="neo4j",
            graph_store_config={
                "uri": "bolt://localhost:7687",
                "username": "neo4j",
                "password": "password"
            },
            retrieval_config={"top_k": 3, "similarity_cutoff": 0.7}
        )
        
        # Mock dependencies
        self.patches = []
        
        # Mock LlamaIndex components
        self.vector_store_mock = MagicMock()
        self.graph_store_mock = MagicMock()
        self.storage_context_mock = MagicMock()
        self.storage_context_mock.vector_store = self.vector_store_mock
        self.storage_context_mock.graph_store = self.graph_store_mock
        self.storage_context_mock.docstore = MagicMock()
        
        # Mock SQLiteStore
        self.sqlite_store_mock = MagicMock()
        self.storage_handler_mock = MagicMock()
        self.storage_handler_mock.storage_db = self.sqlite_store_mock
        self.storage_handler_mock.storage_context = self.storage_context_mock
        
        # Mock Embedding model
        self.embed_model_mock = MagicMock()
        self.embedding_patch = patch("evoagentx.rag.embeddings.EmbeddingFactory.create", return_value=self.embed_model_mock)
        self.patches.append(self.embedding_patch)
        
        # Mock IndexFactory
        self.vector_index_mock = MagicMock()
        self.graph_index_mock = MagicMock()
        self.vector_index_mock.get_index.return_value = MagicMock()
        self.graph_index_mock.get_index.return_value = MagicMock()
        self.index_factory_mock = MagicMock()
        self.index_factory_mock.create.side_effect = lambda index_type, **kwargs: (
            self.vector_index_mock if index_type == IndexType.VECTOR else self.graph_index_mock
        )
        self.index_patch = patch("evoagentx.rag.indexings.IndexFactory.create", return_value=self.index_factory_mock)
        self.patches.append(self.index_patch)
        
        # Mock ChunkFactory
        self.chunker_mock = MagicMock()
        self.chunk_factory_patch = patch("evoagentx.rag.chunkers.ChunkFactory.create", return_value=self.chunker_mock)
        self.patches.append(self.chunk_factory_patch)
        
        # Mock RetrieverFactory
        self.vector_retriever_mock = MagicMock(spec=VectorRetriever)
        self.graph_retriever_mock = MagicMock(spec=GraphRetriever)
        self.retriever_factory_mock = MagicMock()
        self.retriever_factory_mock.create.side_effect = lambda retriever_type, **kwargs: (
            self.vector_retriever_mock if retriever_type == RetrieverType.VECTOR else self.graph_retriever_mock
        )
        self.retriever_patch = patch("evoagentx.rag.retrievers.RetrieverFactory.create", return_value=self.retriever_factory_mock)
        self.patches.append(self.retriever_patch)
        
        # Mock PostprocessorFactory
        self.reranker_mock = MagicMock(spec=SimpleReranker)
        self.postprocessor_factory_mock = MagicMock()
        self.postprocessor_factory_mock.create.return_value = self.reranker_mock
        self.postprocessor_patch = patch("evoagentx.rag.postprocessors.PostprocessorFactory.create", return_value=self.postprocessor_factory_mock)
        self.patches.append(self.postprocessor_patch)
        
        # Start patches
        for p in self.patches:
            p.start()
        
        import pdb;pdb.set_trace()
        # Initialize RagEngine
        self.rag_engine = SearchEngine(config=self.config, storage_handler=self.storage_handler_mock)
        
        # Sample test data
        self.sample_doc_en = Document(
            doc_id=str(uuid4()),
            text="This is a sample English document about AI.",
            metadata=DocumentMetadata(custom_fields={"section_title": "AI Overview"})
        )
        self.sample_doc_zh = Document(
            doc_id=str(uuid4()),
            text="这是一个关于人工智能的中文样本文档。",
            metadata=DocumentMetadata(custom_fields={"section_title": "人工智能概览"})
        )
        self.sample_tool_doc = Document(
            doc_id=str(uuid4()),
            text="Tool for data analysis.",
            metadata=DocumentMetadata(custom_fields={"tool_id": "tool_123"})
        )
        self.sample_query = RagQuery(
            query_str="What is AI?",
            top_k=3,
            similarity_cutoff=0.7,
            keyword_filters=["AI"],
            use_graph=True,
            metadata_filters={"tool_id": "tool_123"}
        )
        
        # Mock chunker output
        self.sample_chunk_en = Chunk(
            chunk_id=str(uuid4()),
            text="This is a sample English document about AI.",
            metadata=ChunkMetadata(doc_id=self.sample_doc_en.doc_id, custom_fields={"section_title": "AI Overview"}),
            embedding=[0.1] * 1536
        )
        self.sample_chunk_zh = Chunk(
            chunk_id=str(uuid4()),
            text="这是一个关于人工智能的中文样本文档。",
            metadata=ChunkMetadata(doc_id=self.sample_doc_zh.doc_id, custom_fields={"section_title": "人工智能概览"}),
            embedding=[0.2] * 1536
        )
        self.sample_chunk_tool = Chunk(
            chunk_id=str(uuid4()),
            text="Tool for data analysis.",
            metadata=ChunkMetadata(doc_id=self.sample_tool_doc.doc_id, custom_fields={"tool_id": "tool_123"}),
            embedding=[0.3] * 1536
        )
        self.sample_corpus = Corpus(chunks=[self.sample_chunk_en, self.sample_chunk_zh, self.sample_chunk_tool])
        self.chunker_mock.chunk.return_value = self.sample_corpus
        
        # Mock retriever output
        self.sample_result = RagResult(
            corpus=self.sample_corpus,
            scores=[0.9, 0.8, 0.7],
            metadata={"query": "What is AI?"}
        )
        self.vector_retriever_mock.retrieve.return_value = self.sample_result
        self.graph_retriever_mock.retrieve.return_value = self.sample_result
        self.reranker_mock.postprocess.return_value = self.sample_result
    
    def tearDown(self):
        """Clean up test environment."""
        for p in self.patches:
            p.stop()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_add_documents(self):
        """Test adding documents to indices."""
        documents = [self.sample_doc_en, self.sample_doc_zh]
        
        # Execute
        corpus = self.rag_engine.add_documents(documents)
        
        # Verify
        self.chunker_mock.chunk.assert_called_once_with(documents)
        self.vector_index_mock.insert_nodes.assert_called_once_with(self.sample_corpus.to_llama_nodes())
        self.assertEqual(self.sqlite_store_mock.insert.call_count, 5)  # 2 docs + 3 chunks
        self.storage_context_mock.graph_store.upsert_triplet.assert_called()
        self.assertEqual(len(corpus.chunks), 3)
        self.assertEqual(corpus.chunks[0].text, "This is a sample English document about AI.")
        self.assertEqual(corpus.chunks[1].text, "这是一个关于人工智能的中文样本文档。")
    
    def test_add_documents_specific_index(self):
        """Test adding documents to a specific index."""
        documents = [self.sample_doc_en]
        
        # Execute
        corpus = self.rag_engine.add_documents(documents, index_type=IndexType.VECTOR)
        
        # Verify
        self.chunker_mock.chunk.assert_called_once_with(documents)
        self.vector_index_mock.insert_nodes.assert_called_once()
        self.graph_index_mock.insert_nodes.assert_not_called()
        self.assertEqual(self.sqlite_store_mock.insert.call_count, 2)  # 1 doc + 1 chunk
        self.assertEqual(len(corpus.chunks), 3)
    
    def test_add_documents_error(self):
        """Test error handling for add_documents."""
        self.chunker_mock.chunk.side_effect = ValueError("Chunking failed")
        
        with self.assertRaises(ValueError):
            self.rag_engine.add_documents([self.sample_doc_en])
        
        self.vector_index_mock.insert_nodes.assert_not_called()
        self.sqlite_store_mock.insert.assert_not_called()
    
    def test_update_documents(self):
        """Test updating existing documents."""
        updated_doc = Document(
            doc_id=self.sample_doc_en.doc_id,
            text="Updated English document about AI.",
            metadata=DocumentMetadata(custom_fields={"section_title": "AI Updated"})
        )
        
        # Mock existing chunks
        self.sqlite_store_mock.get_by_id.return_value = {"chunk_id": self.sample_chunk_en.chunk_id}
        
        # Execute
        corpus = self.rag_engine.update_documents([updated_doc])
        
        # Verify
        self.chunker_mock.chunk.assert_called_once_with([updated_doc])
        self.sqlite_store_mock.delete.assert_called()
        self.vector_index_mock.insert_nodes.assert_called_once_with(self.sample_corpus.to_llama_nodes())
        self.assertEqual(self.sqlite_store_mock.insert.call_count, 5)  # 1 doc + 3 chunks
        self.storage_context_mock.graph_store.upsert_triplet.assert_called()
        self.assertEqual(len(corpus.chunks), 3)
    
    def test_update_documents_no_existing(self):
        """Test updating non-existent documents."""
        updated_doc = Document(
            doc_id=str(uuid4()),
            text="New document.",
            metadata=DocumentMetadata(custom_fields={})
        )
        
        # Mock no existing chunks
        self.sqlite_store_mock.get_by_id.return_value = None
        
        # Execute
        corpus = self.rag_engine.update_documents([updated_doc])
        
        # Verify
        self.sqlite_store_mock.delete.assert_called_once()
        self.vector_index_mock.insert_nodes.assert_called_once()
        self.assertEqual(self.sqlite_store_mock.insert.call_count, 5)
    
    def test_build_index(self):
        """Test building a new index."""
        # Execute
        self.rag_engine.build_index(IndexType.GRAPH)
        
        # Verify
        self.index_factory_mock.create.assert_called_with(
            index_type=IndexType.GRAPH,
            embed_model=self.embed_model_mock,
            storage_context=self.storage_handler_mock.storage_context,
            index_config=self.config.index_config
        )
        self.retriever_factory_mock.create.assert_called_with(
            retriever_type=RetrieverType.GRAPH,
            graph_store=self.storage_context_mock.graph_store,
            embed_model=self.embed_model_mock,
            query=RagQuery(query_str="", top_k=3)
        )
        self.assertIn(IndexType.GRAPH, self.rag_engine.indices)
        self.assertIn(IndexType.GRAPH, self.rag_engine.retrievers)
    
    def test_build_index_error(self):
        """Test error handling for build_index."""
        self.index_factory_mock.create.side_effect = ValueError("Invalid index type")
        
        with self.assertRaises(ValueError):
            self.rag_engine.build_index(IndexType.SUMMARY)
        
        self.retriever_factory_mock.create.assert_not_called()
    
    def test_retrieve(self):
        """Test multi-index retrieval."""
        # Execute
        result = self.rag_engine.retrieve(self.sample_query)
        
        # Verify
        self.vector_retriever_mock.retrieve.assert_called_once_with(self.sample_query)
        self.graph_retriever_mock.retrieve.assert_called_once_with(self.sample_query)
        self.reranker_mock.postprocess.assert_called_once()
        self.assertEqual(self.sqlite_store_mock.insert.call_count, 3)  # 3 chunks in retrieval_log
        self.assertEqual(len(result.corpus.chunks), 3)
        self.assertEqual(result.metadata["query"], "What is AI?")
        self.assertEqual(result.scores, [0.9, 0.8, 0.7])
    
    def test_retrieve_vector_only(self):
        """Test retrieval with vector index only."""
        query = RagQuery(
            query_str="What is AI?",
            top_k=3,
            use_graph=False
        )
        
        # Execute
        result = self.rag_engine.retrieve(query)
        
        # Verify
        self.vector_retriever_mock.retrieve.assert_called_once_with(query)
        self.graph_retriever_mock.retrieve.assert_not_called()
        self.reranker_mock.postprocess.assert_called_once()
        self.assertEqual(self.sqlite_store_mock.insert.call_count, 3)
    
    def test_retrieve_with_metadata_filters(self):
        """Test retrieval with metadata filters for Agentic RAG."""
        # Execute
        result = self.rag_engine.retrieve(self.sample_query)
        
        # Verify
        self.assertTrue(any(chunk.metadata.custom_fields.get("tool_id") == "tool_123" for chunk in result.corpus.chunks))
    
    def test_retrieve_error(self):
        """Test error handling for retrieve."""
        self.vector_retriever_mock.retrieve.side_effect = ValueError("Retrieval failed")
        
        with self.assertRaises(ValueError):
            self.rag_engine.retrieve(self.sample_query)
        
        self.sqlite_store_mock.insert.assert_not_called()
    
    def test_workflow(self):
        """Test complete RagEngine workflow."""
        # Add documents
        documents = [self.sample_doc_en, self.sample_doc_zh, self.sample_tool_doc]
        corpus = self.rag_engine.add_documents(documents)
        self.assertEqual(len(corpus.chunks), 3)
        
        # Update a document
        updated_doc = Document(
            doc_id=self.sample_doc_en.doc_id,
            text="Updated AI document.",
            metadata=DocumentMetadata(custom_fields={"section_title": "AI Updated"})
        )
        self.sqlite_store_mock.get_by_id.return_value = {"chunk_id": self.sample_chunk_en.chunk_id}
        updated_corpus = self.rag_engine.update_documents([updated_doc])
        self.assertEqual(len(updated_corpus.chunks), 3)
        
        # Build a new index
        self.rag_engine.build_index(IndexType.SUMMARY)
        self.assertIn(IndexType.SUMMARY, self.rag_engine.indices)
        
        # Retrieve
        result = self.rag_engine.retrieve(self.sample_query)
        self.assertEqual(len(result.corpus.chunks), 3)
        self.assertTrue(any(chunk.metadata.custom_fields.get("tool_id") == "tool_123" for chunk in result.corpus.chunks))
        
        # Verify storage
        self.assertEqual(self.sqlite_store_mock.insert.call_count, 11)  # 3 docs + 3 chunks (add) + 1 doc + 3 chunks (update) + 3 retrieval_log
    
    # def test_chinese_text_processing(self):
    #     """Test handling of Chinese text with jieba."""
    #     # Mock jieba
    #     with patch("jieba.lcut", return_value=["这", "是", "一个", "关于", "人工智能", "的", "中文", "样本", "文档"]):
    #         corpus = self.rag_engine.add_documents([self.sample_doc_zh])
    #         self.assertEqual(corpus.chunks[1].compute_word_count(), 9)
    
    def test_empty_documents(self):
        """Test adding empty document list."""
        with self.assertRaises(ValueError):
            self.rag_engine.add_documents([])
        
        self.chunker_mock.chunk.assert_not_called()
        self.vector_index_mock.insert_nodes.assert_not_called()

if __name__ == "__main__":
    unittest.main()