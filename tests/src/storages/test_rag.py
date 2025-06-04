import logging
from evoagentx.storages import StorageHandler
from evoagentx.rag.schema import RagQuery, Corpus
from evoagentx.rag.search_engine import SearchEngine, RAGConfig

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize SearchEngine
config = RAGConfig(
    embedding_provider="openai",
    embedding_config={"model": "text-embedding-ada-002"},
    supported_index_types=["vector"],
    retrieval_config={"top_k": 5},
    chunking_strategy="semantic",
    num_workers=4
)
storage_handler = StorageHandler(...)  # Configure with your SQLite/Neo4j setup
search_engine = SearchEngine(config, storage_handler)

# Test Case 1: Basic Retrieval
corpus = search_engine.read(file_paths="./data", corpus_id="paul_graham", filter_file_by_suffix=".txt")
query = RagQuery(query_str="What does Paul Graham say about the role of persistence in startups?", top_k=5)
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