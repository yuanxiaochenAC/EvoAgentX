import os
import re
import asyncio
import logging
from uuid import uuid4
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Optional, Sequence, Dict, Any, Tuple, Literal

from llama_index.core.schema import NodeWithScore, TextNode, RelatedNodeInfo
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
from ..storages.schema import IndexStore
from .schema import Chunk, Corpus, ChunkMetadata, IndexMetadata, RagQuery, RagResult


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

        # Initialize reader
        self.reader = LLamaIndexReader(
            recursive=self.config.reader.recursive,
            exclude_hidden=self.config.reader.exclude_hidden,
            num_workers=self.config.num_workers,
            num_files_limits=self.config.reader.num_files_limit,
            custom_metadata_function=self.config.reader.custom_metadata_function,
            extern_file_extractor=self.config.reader.extern_file_extractor,
            errors=self.config.reader.errors,
            encoding=self.config.reader.encoding
        )

        # Initialize embedding model
        self.embed_model = self.embedding_factory.create(
            provider=self.config.embedding.provider,
            model_config={
                "model_name": self.config.embedding.model_name,
                "api_key": self.config.embedding.api_key,
                "api_base": self.config.embedding.api_url
            }
        )

        # Initialize chunker
        self.chunker = self.chunk_factory.create(
            strategy=self.config.chunker.strategy,
            embed_model=self.embed_model,
            chunker_config={
                "chunk_size": self.config.chunker.chunk_size,
                "chunk_overlap": self.config.chunker.chunk_overlap,
                "max_chunks": self.config.chunker.max_chunks
            }
        )

        # Initialize indices and retrievers
        self.indices = {}  # Nested: {corpus_id: {index_type: index}}
        self.retrievers = {}  # Nested: {corpus_id: {index_type: retriever}}

    def manage_storage(self, corpus_id: str, operation: Literal["add", "delete"], 
                       index_type: Optional[str] = None) -> None:
        """
        Manage storage backends (vector_store, graph_store, storage_context) for a corpus.

        Args:
            corpus_id: Identifier for the corpus.
            operation: Operation to perform ("add" or "delete").
            index_type: Specific index type (e.g., VECTOR, GRAPH) to manage storage for. If None, manages both.
        
        Raises:
            ValueError: If operation is invalid.
            Exception: If storage operation fails.
        """
        try:
            if operation == "add":
                # Initialize vector store for the corpus
                if index_type is None or index_type in [IndexType.VECTOR, IndexType.SUMMARY, IndexType.TREE]:
                    if corpus_id not in self.storage_handler.storageVectors:
                        self.storage_handler._init_vector_store(
                            corpus_id=corpus_id,
                            collection_name=f"corpus_{corpus_id}"
                        )
                        self.logger.info(f"Initialized vector store for corpus {corpus_id}")

                if index_type == IndexType.GRAPH and self.storage_handler.storageGraph is None:
                    self.storage_handler._init_graph_store()
                    self.logger.info(f"Initialized graph store for corpus {corpus_id}")

            elif operation == "delete":
                # Delete vector store and storage context
                if index_type is None or index_type in [IndexType.VECTOR, IndexType.SUMMARY, IndexType.TREE]:
                    if corpus_id in self.storage_handler.storageVectors:
                        del self.storage_handler.storageVectors[corpus_id]
                        if corpus_id in self.storage_handler.storage_contexts:
                            del self.storage_handler.storage_contexts[corpus_id]
                        self.logger.info(f"Deleted vector store and storage context for corpus {corpus_id}")
                # Clear graph store data for the corpus
                if (index_type is None or index_type == IndexType.GRAPH) and self.storage_handler.storageGraph:
                    self.storage_handler._clear_graph_store(corpus_id)
                    self.logger.info(f"Cleared graph store data for corpus {corpus_id}")

        except Exception as e:
            self.logger.error(f"Failed to {operation} storage for corpus {corpus_id}: {str(e)}")
            raise

    def read(self, file_paths: Union[Sequence[str], str], 
             exclude_files: Optional[Union[str, List, Tuple, Sequence]] = None,
             filter_file_by_suffix: Optional[Union[str, List, Tuple, Sequence]] = None,
             merge_by_file: bool = False,
             show_progress: bool = False,
             corpus_id: str = None) -> Corpus:
        """
        Read documents from files, chunk them, and build indices.
        
        Args:
            file_paths: Path(s) to files or directories.
            exclude_files: Files to exclude.
            filter_file_by_suffix: Filter files by suffix (e.g., '.pdf').
            merge_by_file: Merge documents by file.
            show_progress: Show loading progress.
            corpus_id: Identifier for the corpus (optional, defaults to UUID).
            
        Returns:
            Corpus: The chunked corpus.
        """
        try:
            corpus_id = corpus_id or str(uuid4())
            documents = self.reader.load(
                file_paths=file_paths,
                exclude_files=exclude_files,
                filter_file_by_suffix=filter_file_by_suffix,
                merge_by_file=merge_by_file,
                show_progress=show_progress
            )
            corpus = self.chunker.chunk(documents)
            self.logger.info(f"Read {len(documents)} documents and {len(corpus.chunks)} chunks for corpus {corpus_id}")
            return corpus
        except Exception as e:
            self.logger.error(f"Failed to read documents: {str(e)}")
            raise

    def save(self, output_path: Optional[str] = None, corpus_id: Optional[str] = None,
             index_type: Optional[str] = None) -> None:
        """
        Save indices to database or JSONL files (nodes) and JSON file (metadata).
        """
        try:
            def sanitize_filename(s: str) -> str:
                return re.sub(r'[^\w\-]', '_', s)

            saved_corpora = set()
            target_corpora = [corpus_id] if corpus_id else self.indices.keys()
            for cid in target_corpora:
                if cid not in self.indices:
                    self.logger.warning(f"No indices found for corpus {cid}")
                    continue
                target_indices = [self.indices[cid][index_type]] if index_type and index_type in self.indices[cid] else self.indices[cid].values()
                if not target_indices and index_type:
                    self.logger.warning(f"Index type {index_type} not found for corpus {cid}")
                    continue
                for index in target_indices:
                    index_id = str(uuid4())
                    storage_type = "graph" if index.index_type == IndexType.GRAPH else "vector"
                    vector_config = self.storage_handler.storageConfig.vectorConfig.model_dump() or {}
                    graph_config = self.storage_handler.storageConfig.graphConfig.model_dump() or {}
                    metadata = IndexMetadata(
                        corpus_id=cid,
                        index_type=index.index_type,
                        storage_type=storage_type,
                        collection_name=vector_config.get("qdrant_collection_name", "default"),
                        dimension=vector_config.get("dimension", 1536),
                        vector_db_type=vector_config.get("vector_name"),
                        graph_db_type=graph_config.get("graph_name"),
                        date=datetime.now().isoformat(),
                        key_words=[cid, index.index_type]
                    )
                    docstore = index.get_index().storage_context.docstore
                    nodes = docstore.get_all_nodes()
                    corpus = Corpus.from_llama_nodes(nodes, corpus_id=cid)
                    if output_path:
                        os.makedirs(output_path, exist_ok=True)
                        safe_cid = sanitize_filename(cid)
                        safe_index_type = sanitize_filename(index.index_type)
                        nodes_file = os.path.join(output_path, f"{safe_cid}_{safe_index_type}_{index_id}_nodes.jsonl")
                        metadata_file = os.path.join(output_path, f"{safe_cid}_{safe_index_type}_{index_id}_metadata.json")
                        corpus.to_jsonl(nodes_file)
                        with open(metadata_file, 'w', encoding='utf-8') as f:
                            f.write(metadata.to_json(indent=2))
                        self.logger.info(f"Saved {index.index_type} index for corpus {cid} to {nodes_file} and {metadata_file}")
                    else:
                        self.storage_handler.save_index({
                            "index_id": index_id,
                            "corpus_id": cid,
                            "index_type": index.index_type,
                            "storage_type": storage_type,
                            "content": corpus.to_dict(),
                            "date": datetime.now().isoformat(),
                            "key_words": [cid, index.index_type],
                            "metadata": metadata.to_dict()
                        })
                        self.logger.info(f"Saved {index.index_type} index for corpus {cid} to database")
                    saved_corpora.add(cid)
            self.logger.info(f"Saved indices for {len(saved_corpora)} corpora")
        except FileNotFoundError as fne:
            self.logger.error(f"File error saving indices: {str(fne)}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to save indices: {str(e)}")
            raise

    def load(self, source: Optional[str] = None, corpus_id: Optional[str] = None,
            index_type: Optional[str] = None) -> None:
        """
        Load indices from a directory (JSONL nodes + JSON metadata) or SQLite database.
        """
        try:
            if source:
                if not os.path.exists(source):
                    raise FileNotFoundError(f"Source directory {source} does not exist")
                for file_name in os.listdir(source):
                    if file_name.endswith('_metadata.json'):
                        cid, index_type_str, index_id, _ = file_name.split('_', 3)
                        if (corpus_id and corpus_id != cid) or (index_type and index_type != index_type_str):
                            continue
                        metadata_file = os.path.join(source, file_name)
                        nodes_file = os.path.join(source, f"{cid}_{index_type_str}_{index_id}_nodes.jsonl")

                        corpus = Corpus.from_jsonl(nodes_file, corpus_id=cid)
                        if cid not in self.indices:
                            self.indices[cid] = {}
                            self.retrievers[cid] = {}
                        index = self.index_factory.create(
                            index_type=IndexType(index_type_str),
                            embed_model=self.embed_model,
                            storage_context=self.storage_handler.storage_context,
                            index_config=self.config.index_config.model_dump()
                        )
                        index.insert_nodes(corpus.to_llama_nodes())
                        self.indices[cid][IndexType(index_type_str)] = index
                        self.retrievers[cid][IndexType(index_type_str)] = self.retriever_factory.create(
                            retriever_type=RetrieverType.VECTOR if index_type_str in [IndexType.VECTOR, IndexType.SUMMARY, IndexType.TREE] else RetrieverType.GRAPH,
                            index=index.get_index(),
                            graph_store=self.storage_handler.storage_context.graph_store if index_type_str == IndexType.GRAPH else None,
                            embed_model=self.embed_model,
                            query=RagQuery(query_str="", top_k=self.config.retrieval_config.top_k)
                        )
                        self.logger.info(f"Loaded {index_type_str} index for corpus {cid} from {metadata_file}")
            else:
                results = self.storage_handler.load(tables=["index"])
                for record in results.get("index", []):
                    parsed_record = self.storage_handler.parse_result(record, IndexStore)
                    index_type_str = parsed_record["index_type"]
                    cid = parsed_record["corpus_id"]
                    if (corpus_id and corpus_id != cid) or (index_type and index_type != index_type_str):
                        continue
                    corpus_data = parsed_record["content"]
                    metadata_dict = parsed_record["metadata"]
                    self.logger.debug(f"Loaded index metadata for corpus {cid}, index_type {index_type_str}: {metadata_dict}")
                    chunks = [
                        Chunk(
                            chunk_id=chunk["chunk_id"],
                            text=chunk["text"],
                            metadata=ChunkMetadata(**chunk["metadata"]),
                            embedding=chunk["embedding"],
                            start_char_idx=chunk["start_char_idx"],
                            end_char_idx=chunk["end_char_idx"],
                            excluded_embed_metadata_keys=chunk["excluded_embed_metadata_keys"],
                            excluded_llm_metadata_keys=chunk["excluded_llm_metadata_keys"],
                            relationships={k: RelatedNodeInfo(**v) for k, v in chunk["relationships"].items()}
                        ) for chunk in corpus_data["chunks"]
                    ]
                    corpus = Corpus(chunks=chunks, corpus_id=cid)
                    if cid not in self.indices:
                        self.indices[cid] = {}
                        self.retrievers[cid] = {}
                    index = self.index_factory.create(
                        index_type=IndexType(index_type_str),
                        embed_model=self.embed_model,
                        storage_context=self.storage_handler.storage_context,
                        index_config=self.config.index_config
                    )
                    index.insert_nodes(corpus.to_llama_nodes())
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

    def add(self, index_type: str, nodes: Union[Corpus, List[NodeWithScore], List[TextNode]], corpus_id: str = str(uuid4())) -> None:
        """
        Add corpus/nodes to an index for a specific index type.
        """
        try:
            if corpus_id not in self.indices:
                self.indices[corpus_id] = {}
                self.retrievers[corpus_id] = {}

            if index_type not in self.indices[corpus_id]:
                index_config = self.config.index.model_dump() if self.config.index else {}
                index = self.index_factory.create(
                    index_type=index_type,
                    embed_model=self.embed_model,
                    storage_context=self.storage_handler.storage_context,
                    index_config=index_config
                )
                self.indices[corpus_id][index_type] = index
                retriever_type = RetrieverType.VECTOR if index_type in [IndexType.VECTOR, IndexType.SUMMARY, IndexType.TREE] else RetrieverType.GRAPH
                top_k = self.config.retrieval.top_k if hasattr(self.config, "retrieval") else 5
                self.retrievers[corpus_id][index_type] = self.retriever_factory.create(
                    retriever_type=retriever_type,
                    index=index.get_index() if retriever_type == RetrieverType.VECTOR else None,
                    graph_store=self.storage_handler.storage_context.graph_store if retriever_type == RetrieverType.GRAPH else None,
                    embed_model=self.embed_model,
                    query=RagQuery(query_str="", top_k=top_k)
                )

            nodes_to_insert = nodes.to_llama_nodes() if isinstance(nodes, Corpus) else nodes
            for node in nodes_to_insert:
                if node.metadata.get("corpus_id") and node.metadata["corpus_id"] != corpus_id:
                    self.logger.warning(f"Node {node.node_id} has conflicting corpus_id {node.metadata['corpus_id']} (expected {corpus_id})")
                node.metadata["corpus_id"] = corpus_id
                if node.metadata.get("index_type") and node.metadata["index_type"] != index_type:
                    self.logger.warning(f"Node {node.node_id} has conflicting index_type {node.metadata['index_type']} (expected {index_type})")
                node.metadata["index_type"] = index_type
                if isinstance(node, NodeWithScore) and node.score is not None:
                    self.logger.debug(f"Ignoring score {node.score} for node {node.node_id} during insertion")
            self.indices[corpus_id][index_type].insert_nodes(nodes_to_insert)
            self.logger.info(f"Added {len(nodes_to_insert)} nodes to {index_type} index for corpus {corpus_id}")
        except ValueError as ve:
            self.logger.error(f"Configuration error adding nodes to {index_type} index for corpus {corpus_id}: {str(ve)}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to add nodes to {index_type} index for corpus {corpus_id}: {str(e)}")
            raise

    def delete(self, corpus_id: str, index_type: Optional[IndexType] = None,
               node_ids: Optional[List[str]] = None, metadata_filters: Optional[Dict[str, Any]] = None) -> None:
        """
        Delete nodes or entire index from a corpus.
        """
        pass

    async def _retrieve_async(self, retriever: BaseRetrieverWrapper, query: RagQuery):
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
            if isinstance(query, str):
                transformed_query = query
                query = RagQuery(query_str=query)
            else:
                transformed_query = query.query_str

            if not self.indices or (corpus_id and corpus_id not in self.indices):
                self.logger.warning(f"No indices found for corpus {corpus_id or 'any'}")
                return RagResult(corpus=Corpus(chunks=[]), scores=[], metadata={"query": query.query_str})

            # TODO: Add Preprocessor
            # hyde_transform = HyDEQueryTransform(include_original=True)
            # transformed_query = hyde_transform(query.query_str)

            results = []
            target_corpora = [corpus_id] if corpus_id else self.indices.keys()
            with ThreadPoolExecutor(max_workers=self.config.num_workers or 4) as executor:
                future_to_retriever = {}
                for cid in target_corpora:
                    for index_type, retriever in self.retrievers[cid].items():
                        if query.metadata_filters and query.metadata_filters.get("index_type") and \
                           query.metadata_filters["index_type"] != index_type:
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

            if not results:
                self.logger.warning("No results retrieved from any retriever")
                return RagResult(
                    corpus=Corpus(chunks=[]),
                    scores=[],
                    metadata={"query": query.query_str}
                )
        
            postprocessor = self.postprocessor_factory.create(
                similarity_cutoff=query.similarity_cutoff,
                keyword_filters=query.keyword_filters
            )
            final_result = postprocessor.postprocess(query, results)

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