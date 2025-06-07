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
    retrieval=RetrievalConfig(
        retrivel_type="vector",
        postprocessor_type="simple",
        top_k=5, similarity_cutoff=None, keyword_filters=None, metadata_filters=None)
)
# Initialize storage
storage_config = StoreConfig(
    dbConfig=DBConfig(
        db_name="sqlite",
        path="./debug.bk/cache/test.sql"
    ),
    vectorConfig=VectorStoreConfig(
        vector_name="faiss",
        dimension=1536,
        index_type="flat_l2",
    ),
    # file caching
    path="./debug.bk/cache/indexing",
)

storage_handler = StorageHandler(storageConfig=storage_config)  # Configure with your SQLite/Neo4j setup
search_engine = SearchEngine(config, storage_handler)

# Test Case 1: Basic Retrieval
corpus = search_engine.read(file_paths="./debug.bk/data/source_files", filter_file_by_suffix=".txt")
search_engine.add("vector", corpus, corpus_id="paul_graham")
query = RagQuery(query_str="What does Paul Graham say about the role of persistence in startups?", top_k=10)
result = search_engine.query(query, corpus_id="paul_graham")
print("Test Case 1 Results:", [chunk.text for chunk in result.corpus.chunks])

# Test Case 3: Node Management
subset_corpus = Corpus(chunks=corpus.chunks[:10])
search_engine.add(corpus_id="paul_graham", index_type="vector", nodes=subset_corpus)
search_engine.save(corpus_id="paul_graham", index_type="vector")
search_engine.save(output_path=storage_config.path, corpus_id="paul_graham", index_type="vector")
import pdb;pdb.set_trace()
search_engine.delete(corpus_id="paul_graham", index_type="vector", metadata_filters={"file_name": "source.txt"})
search_engine.add(corpus_id="paul_graham", index_type="vector", nodes=subset_corpus)
search_engine.delete(corpus_id="paul_graham", index_type="vector")
search_engine.add(corpus_id="paul_graham", index_type="vector", nodes=subset_corpus)
search_engine.clear()

# search_engine2 = SearchEngine(config, storage_handler)
# search_engine2.load(corpus_id="paul_graham", index_type="vector")
# query = RagQuery(query_str="What does Paul Graham say about the role of persistence in startups?", top_k=5)
# result = search_engine2.query(query, corpus_id="paul_graham")
# print("Test Case 3 Results:", [chunk.text for chunk in result.corpus.chunks])

search_engine2 = SearchEngine(config, storage_handler)
search_engine2.load(source=storage_config.path, corpus_id="paul_graham", index_type="vector")
query = RagQuery(query_str="What does Paul Graham say about the role of persistence in startups?", top_k=5)
result = search_engine2.query(query, corpus_id="paul_graham")
print("Test Case 4 Results:", [chunk.text for chunk in result.corpus.chunks])