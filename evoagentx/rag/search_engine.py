import os
import asyncio
import logging
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Optional, Sequence, Dict, Any, Tuple

from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
# from llama_index.core.indices.query.query_transform import HyDEQueryTransform

from .rag_config import RAGConfig
from .readers import LLamaIndexReader
from .indexings import IndexFactory
from .chunkers import ChunkFactory
from .embeddings import EmbeddingFactory
from .retrievers import RetrieverFactory, BaseRetrieverWrapper
from .postprocessors import PostprocessorFactory
from .indexings.base import IndexType
from .retrievers.base import RetrieverType
from ..storages.base import StorageHandler
from ..storages.schema import MemoryStore
from .schema import Corpus, RagQuery, RagResult


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
        self.indices = {}   # Nested: {corpus_id: {index_type: index}}
        self.retrievers = {}    # Nested: {corpus_id: {index_type: retriever}}

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

    def save(self, output_path: Optional[str] = None, corpus_id: Optional[str] = None,
             index_type: Optional[IndexType] = None) -> None:
        """
        Save indices to database or JSON files.
        
        Args:
            output_path: Directory to save index files as JSON (if None, saves to database).
            corpus_id: Specific corpus to save (default: all corpora).
            index_type: Specific index type to save (default: all types).
        """
        try:
            target_corpora = [corpus_id] if corpus_id else self.indices.keys()
            for cid in target_corpora:
                if cid not in self.indices:
                    self.logger.warning(f"No indices found for corpus {cid}")
                    continue
                target_indices = [self.indices[cid][index_type]] if index_type else self.indices[cid].values()
                for index in target_indices:
                    index_id = str(uuid4())
                    if output_path:
                        os.makedirs(output_path, exist_ok=True)
                        file_path = os.path.join(output_path, f"{cid}_{index.index_type}_{index_id}.json")
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(index.get_index().to_json())
                        self.logger.info(f"Saved {index.index_type} index for corpus {cid} to {file_path}")
                    else:
                        self.storage_handler.storageDB.insert(
                            metadata={
                                "memory_id": index_id,
                                "content": index.get_index().to_json(),
                                "date": "",
                                "key_words": [index.index_type, cid],
                                "entity_content": {"index_type": index.index_type, "corpus_id": cid},
                                "embedding": []
                            },
                            store_type="memory",
                            table="knowledge_indices"
                        )
                        self.logger.info(f"Saved {index.index_type} index for corpus {cid} to database")
            self.logger.info(f"Saved indices for {len(target_corpora)} corpora")
        except Exception as e:
            self.logger.error(f"Failed to save indices: {str(e)}")
            raise

    def load(self, source: Optional[str] = None, corpus_id: Optional[str] = None,
             index_type: Optional[IndexType] = None) -> None:
        """
        Load indices from database or JSON files.
        
        Args:
            source: Directory containing JSON index files or None for database.
            corpus_id: Specific corpus to load (default: all corpora).
            index_type: Specific index type to load (default: all types).
        """
        try:
            if source:
                os.makedirs(source, exist_ok=True)
                for file_name in os.listdir(source):
                    if file_name.endswith('.json'):
                        file_path = os.path.join(source, file_name)
                        cid, index_type_str, _ = file_name.split('_', 2)
                        if (corpus_id and corpus_id != cid) or (index_type and index_type != index_type_str):
                            continue
                        with open(file_path, 'r', encoding='utf-8') as f:
                            index_data = f.read()
                        if cid not in self.indices:
                            self.indices[cid] = {}
                            self.retrievers[cid] = {}
                        index = self.index_factory.create(
                            index_type=IndexType(index_type_str),
                            embed_model=self.embed_model,
                            storage_context=self.storage_handler.storage_context,
                            index_config=self.config.index_config
                        )
                        index.get_index().from_json(index_data)
                        self.indices[cid][IndexType(index_type_str)] = index
                        self.retrievers[cid][IndexType(index_type_str)] = self.retriever_factory.create(
                            retriever_type=RetrieverType.VECTOR if index_type_str in [IndexType.VECTOR, IndexType.SUMMARY, IndexType.TREE] else RetrieverType.GRAPH,
                            index=index.get_index() if index_type_str in [IndexType.VECTOR, IndexType.SUMMARY, IndexType.TREE] else None,
                            graph_store=self.storage_handler.storage_context.graph_store if index_type_str == IndexType.GRAPH else None,
                            embed_model=self.embed_model,
                            query=RagQuery(query_str="", top_k=self.config.retrieval_config.get("top_k", 5))
                        )
                        self.logger.info(f"Loaded {index_type_str} index for corpus {cid} from {file_path}")
            else:
                results = self.storage_handler.load(tables=["knowledge_indices"])
                for record in results.get("knowledge_indices", []):
                    parsed_record = self.storage_handler.parse_result(record, MemoryStore)
                    index_type_str = parsed_record["entity_content"].get("index_type")
                    cid = parsed_record["entity_content"].get("corpus_id", "default")
                    if (corpus_id and corpus_id != cid) or (index_type and index_type != index_type_str):
                        continue
                    if cid not in self.indices:
                        self.indices[cid] = {}
                        self.retrievers[cid] = {}
                    index = self.index_factory.create(
                        index_type=IndexType(index_type_str),
                        embed_model=self.embed_model,
                        storage_context=self.storage_handler.storage_context,
                        index_config=self.config.index_config
                    )
                    index.get_index().from_json(parsed_record["content"])
                    self.indices[cid][IndexType(index_type_str)] = index
                    self.retrievers[cid][IndexType(index_type_str)] = self.retriever_factory.create(
                        retriever_type=RetrieverType.VECTOR if index_type_str in [IndexType.VECTOR, IndexType.SUMMARY, IndexType.TREE] else RetrieverType.GRAPH,
                        index=index.get_index() if index_type_str in [IndexType.VECTOR, IndexType.SUMMARY, IndexType.TREE] else None,
                        graph_store=self.storage_handler.storage_context.graph_store if index_type_str == IndexType.GRAPH else None,
                        embed_model=self.embed_model,
                        query=RagQuery(query_str="", top_k=self.config.retrieval_config.get("top_k", 5))
                    )
                    self.logger.info(f"Loaded {index_type_str} index for corpus {cid} from database")
            self.logger.info(f"Loaded indices for {len(self.indices)} corpora")
        except Exception as e:
            self.logger.error(f"Failed to load indices: {str(e)}")
            raise

    def add(self, index_type: IndexType, nodes: Union[Corpus, List[NodeWithScore], List[TextNode]], corpus_id: str = str(uuid4())) -> None:
        """
        Add corpus/nodes to an index for a specific index type.
        
        Args:
            index_type: Type of index to add nodes to.
            nodes: Corpus or list of nodes to add.
            corpus_id: Identifier for the corpus.(optional)
        """
        try:
            if corpus_id not in self.indices:
                self.indices[corpus_id] = {}
                self.retrievers[corpus_id] = {}

            if index_type not in self.indices[corpus_id]:
                index = self.index_factory.create(
                    index_type=index_type,
                    embed_model=self.embed_model,
                    storage_context=self.storage_handler.storage_context,
                    index_config=self.config.index_config
                )
                self.indices[corpus_id][index_type] = index
                self.retrievers[corpus_id][index_type] = self.retriever_factory.create(
                    retriever_type=RetrieverType.VECTOR if index_type in [IndexType.VECTOR, IndexType.SUMMARY, IndexType.TREE] else RetrieverType.GRAPH,
                    index=index.get_index() if index_type in [IndexType.VECTOR, IndexType.SUMMARY, IndexType.TREE] else None,
                    graph_store=self.storage_handler.storage_context.graph_store if index_type == IndexType.GRAPH else None,
                    embed_model=self.embed_model,
                    query=RagQuery(query_str="", top_k=self.config.retrieval_config.get("top_k", 5))
                )

            nodes_to_insert = nodes.to_llama_nodes() if isinstance(nodes, Corpus) else nodes
            self.indices[corpus_id][index_type].insert_nodes(nodes_to_insert)
            self.logger.info(f"Added {len(nodes_to_insert)} nodes to {index_type} index for corpus {corpus_id}")
        except Exception as e:
            self.logger.error(f"Failed to add nodes to {index_type} index for corpus {corpus_id}: {str(e)}")
            raise

    def delete(self, corpus_id: str, index_type: Optional[IndexType] = None,
               node_ids: Optional[List[str]] = None, metadata_filters: Optional[Dict[str, Any]] = None) -> None:
        """
        Delete nodes or entire index from a corpus.
        
        Args:
            corpus_id: Identifier for the corpus.
            index_type: Specific index type to delete (default: all types).
            node_ids: Specific node IDs to delete (optional).
            metadata_filters: Metadata filters to identify nodes to delete (optional).
        """
        try:
            if corpus_id not in self.indices:
                self.logger.warning(f"No indices found for corpus {corpus_id}")
                return

            if index_type:
                target_indices = [(index_type, self.indices[corpus_id][index_type])]
            else:
                target_indices = self.indices[corpus_id].items()

            for idx_type, index in target_indices:
                if node_ids or metadata_filters:
                    # Delete specific nodes
                    nodes = index.get_index().as_retriever().retrieve(QueryBundle(query_str=""))
                    nodes_to_delete = []
                    for node in nodes:
                        if node_ids and node.node_id in node_ids:
                            nodes_to_delete.append(node.node_id)
                        elif metadata_filters and all(
                            node.metadata.get(k) == v for k, v in metadata_filters.items()
                        ):
                            nodes_to_delete.append(node.node_id)
                    for node_id in nodes_to_delete:
                        index.get_index().delete_node(node_id)
                    self.logger.info(f"Deleted {len(nodes_to_delete)} nodes from {idx_type} index for corpus {corpus_id}")
                else:
                    # Delete entire index
                    del self.indices[corpus_id][idx_type]
                    del self.retrievers[corpus_id][idx_type]
                    self.logger.info(f"Deleted {idx_type} index for corpus {corpus_id}")
            
            if not self.indices[corpus_id]:
                del self.indices[corpus_id]
                del self.retrievers[corpus_id]
                self.logger.info(f"Removed empty corpus {corpus_id}")
        except Exception as e:
            self.logger.error(f"Failed to delete from corpus {corpus_id}: {str(e)}")
            raise

    async def _retrieve_async(self, retriever: BaseRetrieverWrapper, query: RagQuery) -> RagResult:
        """Helper method to run async retrieval."""
        return await retriever.aretrieve(query)

    def query(self, query: Union[str, RagQuery], corpus_id: Optional[str] = None) -> RagResult:
        """
        Execute a query through preprocessing, multi-threaded async retrieval, and post-processing.
        
        Args:
            query: RagQuery object containing query string and parameters.
            corpus_id: Specific corpus to query (default: all corpora).
            
        Returns:
            RagResult: Retrieved results with scores and metadata.
        """
        try:
            # Handle the query type
            if isinstance(query, str):
                transformed_query = query
                query = RagQuery(query_str=query)
            
            if not self.indices or (corpus_id and corpus_id not in self.indices):
                self.logger.warning(f"No indices found for corpus {corpus_id or 'any'}")
                return RagResult(corpus=Corpus(chunks=[]), scores=[], metadata={"query": query.query_str})

            # TODO: Add Preprocessor
            # Example:
            # hyde_transform = HyDEQueryTransform(include_original=True)
            # transformed_query = hyde_transform(query.query_str)

            # Retrieval: Use ThreadPoolExecutor for parallel async retrieval
            results = []
            target_corpora = [corpus_id] if corpus_id else self.indices.keys()
            with ThreadPoolExecutor(max_workers=self.config.num_workers or 4) as executor:
                future_to_retriever = {}
                for cid in target_corpora:
                    for index_type, retriever in self.retrievers[cid].items():
                        if query.metadata_filters and query.metadata_filters.get("index_type") and \
                           query.metadata_filters["index_type"] != index_type.value:
                            continue
                        future = executor.submit(
                            asyncio.run, self._retrieve_async(
                                retriever, RagQuery(
                                    query_str=transformed_query,
                                    top_k=query.top_k,
                                    similarity_cutoff=query.similarity_cutoff,
                                    keyword_filters=query.keyword_filters,
                                    metadata_filters=query.metadata_filters
                                )
                            )
                        )
                        future_to_retriever[future] = (cid, index_type)
                
                for future in as_completed(future_to_retriever):
                    cid, index_type = future_to_retriever[future]
                    try:
                        result = future.result()
                        results.append(result)
                        self.logger.info(f"Retrieved {len(result.corpus.chunks)} chunks from {index_type} retriever for corpus {cid}")
                    except Exception as e:
                        self.logger.error(f"Retrieval failed for {index_type} in corpus {cid}: {str(e)}")

            # Check for empty results
            if not results:
                self.logger.warning("No results retrieved from any retriever")
                return RagResult(
                    corpus=Corpus(chunks=[]),
                    scores=[],
                    metadata={"query": query.query_str}
                )
        
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