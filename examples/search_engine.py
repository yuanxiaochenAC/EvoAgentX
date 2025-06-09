import os

from dotenv import load_dotenv

from evoagentx.rag.rag_config import RAGConfig, ReaderConfig, ChunkerConfig, IndexConfig, EmbeddingConfig, RetrievalConfig
from evoagentx.rag.search_engine import SearchEngine
# import the benchmark
from evoagentx.benchmark.hotpotqa import HotPotQA


load_dotenv()


