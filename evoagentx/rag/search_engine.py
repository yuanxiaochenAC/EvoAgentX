import os
import logging
from uuid import uuid4
from typing import List, Union, Optional, Sequence, Dict, Any, Tuple

from llama_index.core.indices.query.query_transform import HyDEQueryTransform

from .rag_config import RAGConfig
from .readers import LLamaIndexReader
from .indexings import IndexFactory
from .chunkers import ChunkFactory
from .embeddings import EmbeddingFactory
from .retrievers import RetrieverFactory
from .postprocessors import PostprocessorFactory
from .indexings.base import IndexType
from .retrievers.base import RetrieverType
from ..storages.base import StorageHandler
from .schema import Corpus, Document, RagQuery, RagResult


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

        # Initialize readers
        self.reader = LLamaIndexReader(
            recursive=self.config.recursive,
            exclude_hidden=self.config.exclude_hidden,
            num_workers=self.config.num_workers,
            num_files_limits=self.config.num_files_limits,
            custom_metadata_function=self.config.custom_metadata_function,
            extern_file_extractor=self.config.extern_file_extractor,
            errors=self.config.errors,
            encoding=self.config.encoding
        )

        # Initialize embedding model
        self.embed_model = self.embedding_factory.create(
            provider=self.config.embedding_provider,
            model_config=self.config.embedding_config
        )

        # Initialize chunker
        self.chunker = self.chunk_factory.create(
            strategy=self.config.chunking_strategy,
            embed_model=self.embed_model,
            chunker_config=self.config.node_parser_config
        )

        # Initialize indices and retrievers
        self.indices = {}
        self.retrievers = {}
        self._initialize_indices_and_retrievers()

    def _initialize_indices_and_retrievers(self):
        """Initialize indices and their corresponding retrievers."""
        try:
            # Vector index is always initialized
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
                query=RagQuery(query_str="", top_k=self.config.retrieval_config.get("top_k", 5))
            )

            # Initialize additional indices based on config
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
                    query=RagQuery(query_str="", top_k=self.config.retrieval_config.get("top_k", 5))
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
                    retriever_type=RetrieverType.VECTOR,
                    index=summary_index.get_index(),
                    query=RagQuery(query_str="", top_k=self.config.retrieval_config.get("top_k", 5))
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
                    retriever_type=RetrieverType.VECTOR,
                    index=tree_index.get_index(),
                    query=RagQuery(query_str="", top_k=self.config.retrieval_config.get("top_k", 5))
                )

            self.logger.info(f"Initialized {len(self.indices)} indices and retrievers")
        except Exception as e:
            self.logger.error(f"Failed to initialize indices: {str(e)}")
            raise

    def read(self, file_paths: Union[Sequence[str], str], 
             exclude_files: Optional[Union[str, List, Tuple, Sequence]] = None,
             filter_file_by_suffix: Optional[Union[str, List, Tuple, Sequence]] = None,
             merge_by_file: bool = False,
             show_progress: bool = False) -> Corpus:
        """
        Read documents from files, chunk them, and build indices.
        
        Args:
            file_paths: Path(s) to files or directories.
            exclude_files: Files to exclude.
            filter_file_by_suffix: Filter files by suffix (e.g., '.pdf').
            merge_by_file: Merge documents by file.
            show_progress: Show loading progress.
            
        Returns:
            Corpus: The chunked corpus.
        """
        try:
            documents = self.reader.load(
                file_paths=file_paths,
                exclude_files=exclude_files,
                filter_file_by_suffix=filter_file_by_suffix,
                merge_by_file=merge_by_file,
                show_progress=show_progress
            )
            corpus = self.chunker.chunk(documents)
            nodes = corpus.to_llama_nodes()

            for index in self.indices.values():
                index.insert_nodes(nodes)

            self.logger.info(f"Read {len(documents)} documents and {len(corpus.chunks)} chunks")
            return corpus
        except Exception as e:
            self.logger.error(f"Failed to read documents: {str(e)}")
            raise

    def save(self, output_path: Optional[str] = None, index_type: Optional[IndexType] = None) -> None:
        """
        Save indices to database or JSON files.
        
        Args:
            output_path: Directory to save index files as JSON (if None, saves to database).
            index_type: Specific index type to save (default: all indices).
        """
        try:
            target_indices = [self.indices[index_type]] if index_type else self.indices.values()
            for index in target_indices:
                index_id = str(uuid4())
                if output_path:
                    # Save to JSON file
                    os.makedirs(output_path, exist_ok=True)
                    file_path = os.path.join(output_path, f"{index.index_type}_{index_id}.json")
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(index.get_index().to_json())
                    self.logger.info(f"Saved {index.index_type} index to {file_path}")
                else:
                    # Save to database
                    self.storage_handler.storageDB.insert(
                        metadata={
                            "memory_id": index_id,
                            "content": index.get_index().to_json(),
                            "date": "",
                            "key_words": [index.index_type],
                            "entity_content": {"index_type": index.index_type},
                            "embedding": []
                        },
                        store_type="memory",
                        table="knowledge_indices"
                    )
                    self.logger.info(f"Saved {index.index_type} index to database with ID {index_id}")
            self.logger.info(f"Saved {len(target_indices)} indices")
        except Exception as e:
            self.logger.error(f"Failed to save indices: {str(e)}")
            raise

    def load(self, source: Optional[str] = None, index_type: Optional[IndexType] = None) -> None:
        """
        Load indices from database or JSON files.
        
        Args:
            source: Directory containing JSON index files or None for database.
            index_type: Specific index type to load (default: all supported types).
        """
        try:
            if source:
                # Load from JSON files
                os.makedirs(source, exist_ok=True)
                for file_name in os.listdir(source):
                    if file_name.endswith('.json'):
                        file_path = os.path.join(source, file_name)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            index_data = f.read()
                        index_type_str = file_name.split('_')[0]
                        if index_type and index_type != index_type_str:
                            continue
                        index = self.index_factory.create(
                            index_type=IndexType(index_type_str),
                            embed_model=self.embed_model,
                            storage_context=self.storage_handler.storage_context,
                            index_config=self.config.index_config
                        )
                        index.get_index().from_json(index_data)
                        self.indices[IndexType(index_type_str)] = index
                        self.retrievers[IndexType(index_type_str)] = self.retriever_factory.create(
                            retriever_type=RetrieverType.VECTOR if index_type_str in [IndexType.VECTOR, IndexType.SUMMARY, IndexType.TREE] else RetrieverType.GRAPH,
                            index=index.get_index() if index_type_str in [IndexType.VECTOR, IndexType.SUMMARY, IndexType.TREE] else None,
                            graph_store=self.storage_handler.storage_context.graph_store if index_type_str == IndexType.GRAPH else None,
                            embed_model=self.embed_model,
                            query=RagQuery(query_str="", top_k=self.config.retrieval_config.get("top_k", 5))
                        )
                        self.logger.info(f"Loaded {index_type_str} index from {file_path}")
            else:
                # Load from database
                results = self.storage_handler.load(tables=["knowledge_indices"])
                for record in results.get("knowledge_indices", []):
                    parsed_record = self.storage_handler.parse_result(record, MemoryStore)
                    index_type_str = parsed_record["entity_content"].get("index_type")
                    if index_type and index_type != index_type_str:
                        continue
                    index = self.index_factory.create(
                        index_type=IndexType(index_type_str),
                        embed_model=self.embed_model,
                        storage_context=self.storage_handler.storage_context,
                        index_config=self.config.index_config
                    )
                    index.get_index().from_json(parsed_record["content"])
                    self.indices[IndexType(index_type_str)] = index
                    self.retrievers[IndexType(index_type_str)] = self.retriever_factory.create(
                        retriever_type=RetrieverType.VECTOR if index_type_str in [IndexType.VECTOR, IndexType.SUMMARY, IndexType.TREE] else RetrieverType.GRAPH,
                        index=index.get_index() if index_type_str in [IndexType.VECTOR, IndexType.SUMMARY, IndexType.TREE] else None,
                        graph_store=self.storage_handler.storage_context.graph_store if index_type_str == IndexType.GRAPH else None,
                        embed_model=self.embed_model,
                        query=RagQuery(query_str="", top_k=self.config.retrieval_config.get("top_k", 5))
                    )
                    self.logger.info(f"Loaded {index_type_str} index from database with ID {parsed_record['memory_id']}")
            self.logger.info(f"Loaded {len(self.indices)} indices")
        except Exception as e:
            self.logger.error(f"Failed to load indices: {str(e)}")
            raise

    def query(self, query: RagQuery) -> RagResult:
        """
        Execute a query through preprocessing, retrieval, and post-processing.
        
        Args:
            query: RagQuery object containing query string and parameters.
            
        Returns:
            RagResult: Retrieved results with scores and metadata.
        """
        try:
            # Preprocessing: Apply HyDE query transformation
            hyde_transform = HyDEQueryTransform(include_original=True)
            transformed_query = hyde_transform(query.query_str)

            # Retrieval: Use all available retrievers
            results = []
            for index_type, retriever in self.retrievers.items():
                if index_type == IndexType.GRAPH and not query.use_graph:
                    continue
                result = retriever.retrieve(RagQuery(
                    query_str=transformed_query,
                    top_k=query.top_k,
                    similarity_cutoff=query.similarity_cutoff,
                    keyword_filters=query.keyword_filters,
                    use_graph=query.use_graph,
                    metadata_filters=query.metadata_filters
                ))
                results.append(result)
                self.logger.info(f"Retrieved {len(result.corpus.chunks)} chunks from {index_type} retriever")

            # Post-processing: Rerank results
            postprocessor = self.postprocessor_factory.create(
                similarity_cutoff=query.similarity_cutoff,
                keyword_filters=query.keyword_filters
            )
            final_result = postprocessor.postprocess(query, results)

            # Apply metadata filters if provided
            if query.metadata_filters:
                filtered_chunks = [
                    chunk for chunk in final_result.corpus.chunks
                    if all(chunk.metadata.model_dump().get(k) == v for k, v in query.metadata_filters.items())
                ]
                final_result.corpus.chunks = filtered_chunks
                final_result.scores = [chunk.metadata.similarity_score for chunk in filtered_chunks]
                self.logger.info(f"Applied metadata filters, retained {len(filtered_chunks)} chunks")

            self.logger.info(f"Query returned {len(final_result.corpus.chunks)} chunks after post-processing")
            return final_result
        except Exception as e:
            self.logger.error(f"Query failed: {str(e)}")
            raise