import os
import logging
from evoagentx.storages import StorageHandler
from evoagentx.storages.storages_config import StoreConfig, VectorStoreConfig, DBConfig
from evoagentx.rag.schema import RagQuery, Corpus
from evoagentx.rag.search_engine import SearchEngine
from evoagentx.rag.rag_config import (
    RAGConfig, 
    ReaderConfig, 
    ChunkerConfig,
    EmbeddingConfig, 
    IndexConfig, 
    RetrievalConfig
)

import dotenv
dotenv.load_dotenv()


# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize SearchEngine
config = RAGConfig(
    num_workers=2,
    reader=ReaderConfig(recursive=False, exclude_hidden=True, num_files_limit=None,
                        custom_metadata_function=None, extern_file_extractor=None, 
                        errors="ignore", encoding="utf-8"),
    chunker=ChunkerConfig(strategy="simple", chunk_size=1024, chunk_overlap=20, max_chunks=None),
    embedding=EmbeddingConfig(provider="openai", model_name="text-embedding-ada-002",
                              api_key=os.environ["OPENAI_API_KEY"]),
    index=IndexConfig(index_type="faiss"),
    retrieval=RetrievalConfig(top_k=5, similarity_cutoff=None, keyword_filters=None, metadata_filters=None)
)
# Initialize storage
storage_config = StoreConfig(
    dbConfig=DBConfig(
        db_name="sqlite",
        path="./debug/cache/test.sql"
    ),
    vectorConfig=VectorStoreConfig(
        vector_name="faiss",
        dimension=1536,
        index_type="flat_l2",
    ),
    # file caching
    path="./debug/cache/indexing",
)

storage_handler = StorageHandler(storageConfig=storage_config)  # Configure with your SQLite/Neo4j setup
search_engine = SearchEngine(config, storage_handler)

import pdb;pdb.set_trace()
# Test Case 1: Basic Retrieval
corpus = search_engine.read(file_paths="./debug/data/source_files", filter_file_by_suffix=".txt")
query = RagQuery(query_str="What does Paul Graham say about the role of persistence in startups?", top_k=5)
search_engine.add("vector", corpus, corpus_id="paul_graham")
search_engine.save(storage_config.path, corpus_id=None, index_type=None)
result = search_engine.query(query, corpus_id="paul_graham")
print("Test Case 1 Results:", [chunk.text for chunk in result.corpus.chunks])

# Test Case 2: Multi-Index Retrieval
search_engine.config.supported_index_types = ["vector", "graph"]
corpus = search_engine.read(file_paths="./data", corpus_id="paul_graham", filter_file_by_suffix=".txt")
query = RagQuery(
    query_str="How does Paul Graham describe the balance between passion and discipline in work?",
    top_k=5,
    metadata_filters={"index_type": "vector"}
)
result_vector = search_engine.query(query, corpus_id="paul_graham")
query.metadata_filters = {"index_type": "graph"}
result_graph = search_engine.query(query, corpus_id="paul_graham")
print("Test Case 2 Vector Results:", [chunk.text for chunk in result_vector.corpus.chunks])
print("Test Case 2 Graph Results:", [chunk.text for chunk in result_graph.corpus.chunks])

# Test Case 3: Node Management
subset_corpus = Corpus(chunks=corpus.chunks[:10])
search_engine.add(corpus_id="test_subset", index_type="vector", nodes=subset_corpus)
search_engine.delete(corpus_id="test_subset", index_type="vector", metadata_filters={"file_name": "document1.txt"})
search_engine.save(corpus_id="test_subset", index_type="vector")
search_engine.indices.clear()
search_engine.load(corpus_id="test_subset", index_type="vector")
query = RagQuery(query_str="What is Paul Graham’s advice on hiring for startups?", top_k=5)
result = search_engine.query(query, corpus_id="test_subset")
print("Test Case 3 Results:", [chunk.text for chunk in result.corpus.chunks])