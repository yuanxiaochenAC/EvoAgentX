import os
import json
import asyncio
import logging
from uuid import uuid4
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Optional, Sequence, Dict, Any, Tuple

from llama_index.core.schema import NodeWithScore, TextNode, RelatedNodeInfo
# from llama_index.core.indices.query.query_transform import HyDEQueryTransform

from .rag_config import RAGConfig
from .readers import LLamaIndexReader
from .indexings import IndexFactory, BaseIndexWrapper
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
        self.indices: Dict[str, Dict[str, BaseIndexWrapper]] = {}  # Nested: {corpus_id: {index_type: index}}
        self.retrievers: Dict[str, Dict[str, BaseRetrieverWrapper]] = {}  # Nested: {corpus_id: {index_type: retriever}}

    def read(self, file_paths: Union[Sequence[str], str], 
             exclude_files: Optional[Union[str, List, Tuple, Sequence]] = None,
             filter_file_by_suffix: Optional[Union[str, List, Tuple, Sequence]] = None,
             merge_by_file: bool = False,
             show_progress: bool = False,
             corpus_id: str = None) -> Corpus:
        """Load and chunk documents from files.

        Reads files from specified paths, processes them into documents, and chunks them into a Corpus.

        Args:
            file_paths (Union[Sequence[str], str]): Path(s) to files or directories.
            exclude_files (Optional[Union[str, List, Tuple, Sequence]]): Files to exclude.
            filter_file_by_suffix (Optional[Union[str, List, Tuple, Sequence]]): Filter files by suffix (e.g., '.pdf').
            merge_by_file (bool): Merge documents by file.
            show_progress (bool): Show loading progress.
            corpus_id (Optional[str]): Identifier for the corpus. Defaults to a UUID if None.

        Returns:
            Corpus: The chunked corpus containing processed document chunks.

        Raises:
            Exception: If document reading or chunking fails.
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
            self.logger.info(f"Read {len(documents)} documents and created {len(corpus.chunks)} chunks for corpus {corpus_id}")
            return corpus
        except Exception as e:
            self.logger.error(f"Failed to read documents for corpus {corpus_id}: {str(e)}")
            raise

    def add(self, index_type: str, nodes: Union[Corpus, List[NodeWithScore], List[TextNode]], 
            corpus_id: str = None) -> None:
        """Add nodes to an index for a specific corpus.

        Initializes an index if it doesn't exist and inserts nodes, updating metadata with corpus_id and index_type.

        Args:
            index_type (str): Type of index (e.g., VECTOR, GRAPH).
            nodes (Union[Corpus, List[NodeWithScore], List[TextNode]]): Nodes or Corpus to add.
            corpus_id (str, optional): Identifier for the corpus. Defaults to a UUID if None.

        Raises:
            Exception: If index creation or node insertion fails.
        """
        try:
            corpus_id = corpus_id or str(uuid4())
            if corpus_id not in self.indices:
                self.indices[corpus_id] = {}
                self.retrievers[corpus_id] = {}

            if index_type not in self.indices[corpus_id]:
                index = self.index_factory.create(
                    index_type=index_type,
                    embed_model=self.embed_model,
                    storage_handler=self.storage_handler,
                    index_config=self.config.index.model_dump() if self.config.index else {}
                )
                self.indices[corpus_id][index_type] = index
                self.retrievers[corpus_id][index_type] = self.retriever_factory.create(
                    retriever_type=self.config.retrieval.retrivel_type,
                    index=index.get_index(),
                    graph_store=index.get_index().storage_context.graph_store,
                    embed_model=self.embed_model,
                    query=RagQuery(query_str="", top_k=self.config.retrieval.top_k if self.config.retrieval else 5)
                )

            nodes_to_insert = nodes.to_llama_nodes() if isinstance(nodes, Corpus) else nodes
            for node in nodes_to_insert:
                node.metadata.update({"corpus_id": corpus_id, "index_type": index_type})
            self.indices[corpus_id][index_type].insert_nodes(nodes_to_insert)
            self.logger.info(f"Added {len(nodes_to_insert)} nodes to {index_type} index for corpus {corpus_id}")
        except Exception as e:
            self.logger.error(f"Failed to add nodes to {index_type} index for corpus {corpus_id}: {str(e)}")
            raise

    def delete(self, corpus_id: str, index_type: Optional[str] = None, 
               node_ids: Optional[Union[str, List[str]]] = None, 
               metadata_filters: Optional[Dict[str, Any]] = None) -> None:
        """Delete nodes or an entire index from a corpus.

        Removes specific nodes by ID or metadata filters, or deletes the entire index if no filters are provided.

        Args:
            corpus_id (str): Identifier for the corpus.
            index_type (Optional[IndexType]): Specific index type to delete from. If None, affects all indices.
            node_ids (Union[str, Optional[List[str]]]): List of node IDs to delete.
            metadata_filters (Optional[Dict[str, Any]]): Metadata filters to select nodes for deletion.

        Raises:
            Exception: If deletion fails.
        """
        try:
            if corpus_id not in self.indices:
                self.logger.warning(f"No indices found for corpus {corpus_id}")
                return

            target_indices = [index_type] if index_type else self.indices[corpus_id].keys()
            for idx_type in list(target_indices):  # Use list to avoid runtime modification issues
                if idx_type not in self.indices[corpus_id]:
                    self.logger.warning(f"Index type {idx_type} not found for corpus {corpus_id}")
                    continue

                index = self.indices[corpus_id][idx_type]
                if node_ids or metadata_filters:
                    # Convert single node_id to list for consistency
                    node_ids_list = [node_ids] if isinstance(node_ids, str) else node_ids
                    index.delete_nodes(node_ids=node_ids_list, metadata_filters=metadata_filters)
                    self.logger.info(f"Deleted nodes from {idx_type} index for corpus {corpus_id}")
                else:
                    # Delete entire index
                    index.clear()
                    self.indices[corpus_id].pop(idx_type)
                    self.retrievers[corpus_id].pop(idx_type)
                    self.logger.info(f"Deleted entire {idx_type} index for corpus {corpus_id}")

            # Clean up corpus if no indices remain
            if not self.indices[corpus_id]:
                self.indices.pop(corpus_id)
                self.retrievers.pop(corpus_id)
                self.logger.info(f"Removed empty corpus {corpus_id}")

        except Exception as e:
            self.logger.error(f"Failed to delete from corpus {corpus_id}, index {index_type}: {str(e)}")
            raise

    def clear(self, corpus_id: Optional[str] = None) -> None:
        """Clear all indices for a specific corpus or all corpora.

        Args:
            corpus_id (Optional[str]): Specific corpus to clear. If None, clears all corpora.

        Raises:
            Exception: If clearing fails.
        """
        try:
            target_corpora = [corpus_id] if corpus_id else list(self.indices.keys())
            for cid in target_corpora:
                if cid not in self.indices:
                    self.logger.warning(f"No indices found for corpus {cid}")
                    continue

                for idx_type in list(self.indices[cid].keys()):
                    index = self.indices[cid][idx_type]
                    index.clear()
                    del self.indices[cid][idx_type]
                    del self.retrievers[cid][idx_type]
                    self.logger.info(f"Cleared {idx_type} index for corpus {cid}")

                # Clean up corpus if no indices remain
                del self.indices[cid]
                del self.retrievers[cid]
                self.logger.info(f"Cleared corpus {cid}")

        except Exception as e:
            self.logger.error(f"Failed to clear indices for corpus {corpus_id or 'all'}: {str(e)}")
            raise

    def save(self, output_path: Optional[str] = None, corpus_id: Optional[str] = None, 
             index_type: Optional[str] = None) -> None:
        """Save indices to files or database.

        Serializes corpus chunks and metadata to JSONL/JSON files or SQLite database.

        Args:
            output_path (Optional[str]): Directory to save files. If None, saves to database.
            corpus_id (Optional[str]): Specific corpus to save. If None, saves all corpora.
            index_type (Optional[str]): Specific index type to save. If None, saves all indices.

        Raises:
            Exception: If saving fails.
        """
        try:
            target_corpora = [corpus_id] if corpus_id else self.indices.keys()
            for cid in target_corpora:
                if cid not in self.indices:
                    self.logger.warning(f"No indices found for corpus {cid}")
                    continue

                target_indices = [index_type] if index_type and index_type in self.indices[cid] else self.indices[cid].keys()
                for idx_type in target_indices:
                    index = self.indices[cid][idx_type]

                    # Convert docstore/data to Corpus
                    chunks = []
                    for node_id, node_data in index.id_to_node.items():
                        chunk = Chunk.from_llama_node(node_data)
                        chunks.append(chunk)

                    corpus = Corpus(chunks=chunks, corpus_id=cid)

                    # Get config
                    vector_config = {}
                    graph_config = {}
                    if self.storage_handler.storageConfig.vectorConfig is not None:
                        vector_config = self.storage_handler.storageConfig.vectorConfig.model_dump()
                    if self.storage_handler.storageConfig.graphConfig is not None:
                        graph_config = self.storage_handler.storageConfig.graphConfig.model_dump()

                    metadata = IndexMetadata(
                        corpus_id=cid,
                        index_type=idx_type,
                        collection_name=vector_config.get("qdrant_collection_name", "default_collection"),
                        dimension=vector_config.get("dimension", 1536),
                        vector_db_type=vector_config.get("vector_name", None),
                        graph_db_type=graph_config.get("graph_name", None),
                        date=str(datetime.now()),
                    )

                    if output_path:
                        os.makedirs(output_path, exist_ok=True)
                        safe_cid = "".join(c if c.isalnum() or c in ["-", "_"] else "_" for c in cid)
                        safe_idx_type = "".join(c if c.isalnum() or c in ["-", "_"] else "_" for c in idx_type)
                        nodes_file = os.path.join(output_path, f"{safe_cid}_{safe_idx_type}_nodes.jsonl")
                        metadata_file = os.path.join(output_path, f"{safe_cid}_{safe_idx_type}_metadata.json")
                        corpus.to_jsonl(nodes_file)
                        with open(metadata_file, "w", encoding="utf-8") as f:
                            json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)
                        self.logger.info(f"Saved {idx_type} index for corpus {cid} to {nodes_file}")
                    else:
                        self.storage_handler.save_index({
                            "corpus_id": cid,
                            "index_type": idx_type,
                            "content": corpus.to_dict(),
                            "date": str(datetime.now()),
                            "metadata": metadata.to_dict()
                        })
                        self.logger.info(f"Saved {idx_type} index for corpus {cid} to database")
        except Exception as e:
            self.logger.error(f"Failed to save indices: {str(e)}")
            raise

    def load(self, source: Optional[str] = None, corpus_id: Optional[str] = None, 
             index_type: Optional[str] = None, table: Optional[str] = None) -> None:
        """Load indices from files or database.

        Reconstructs indices and retrievers from JSONL/JSON files or SQLite database records.

        Args:
            source (Optional[str]): Directory containing JSONL/JSON files. If None, loads from database.
            corpus_id (Optional[str]): Specific corpus to load. If None, loads all corpora.
            index_type (Optional[str]): Specific index type to load. If None, loads all indices.

        Raises:
            Exception: If loading fails.
        """
        try:
            if source:
                for file_name in os.listdir(source):
                    if not file_name.endswith("_metadata.json"):
                        continue
                    cid, idx_type, index_id, _ = file_name.split("_", 3)
                    if (corpus_id and corpus_id != cid) or (index_type and index_type != idx_type):
                        continue

                    nodes_file = os.path.join(source, f"{cid}_{idx_type}_{index_id}_nodes.jsonl")
                    corpus = Corpus.from_jsonl(nodes_file, corpus_id=cid)
                    self._load_index(corpus, cid, idx_type)
                    self.logger.info(f"Loaded {idx_type} index for corpus {cid} from {nodes_file}")
            else:
                for record in self.storage_handler.load_index("", table=table):
                    parsed = self.storage_handler.parse_result(record, IndexStore)
                    cid, idx_type = parsed["corpus_id"], parsed["index_type"]
                    if (corpus_id and corpus_id != cid) or (index_type and index_type != idx_type):
                        continue

                    corpus = Corpus(
                        chunks=[
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
                            ) for chunk in parsed["content"]["chunks"]
                        ],
                        corpus_id=cid
                    )
                    self._load_index(corpus, cid, idx_type)
                    self.logger.info(f"Loaded {idx_type} index for corpus {cid} from database")
        except Exception as e:
            self.logger.error(f"Failed to load indices: {str(e)}")
            raise

    def _load_index(self, corpus: Corpus, corpus_id: str, index_type: str) -> None:
        """Helper method to load an index and its retriever.

        Initializes storage, creates an index, inserts nodes, and sets up the retriever.

        Args:
            corpus (Corpus): Corpus containing chunks to load.
            corpus_id (str): Identifier for the corpus.
            index_type (str): Type of index to load.
        """
        if corpus_id not in self.indices:
            self.indices[corpus_id] = {}
            self.retrievers[corpus_id] = {}
        index = self.index_factory.create(
            index_type=index_type,
            embed_model=self.embed_model,
            storage_handler=self.storage_handler,
            index_config=self.config.index.model_dump() if self.config.index else {}
        )
        index.insert_nodes(corpus.to_llama_nodes())
        self.indices[corpus_id][index_type] = index
        self.retrievers[corpus_id][index_type] = self.retriever_factory.create(
            retriever_type=RetrieverType.GRAPH if index_type == IndexType.GRAPH else RetrieverType.VECTOR,
            index=index.get_index(),
            graph_store=self.storage_handler.storage_context.graph_store if index_type == IndexType.GRAPH else None,
            embed_model=self.embed_model,
            query=RagQuery(query_str="", top_k=self.config.retrieval.top_k if self.config.retrieval else 5)
        )

    async def _retrieve_async(self, retriever: BaseRetrieverWrapper, query: RagQuery):
        """Asynchronously retrieve results using a retriever.

        Args:
            retriever (BaseRetrieverWrapper): Retriever to process the query.
            query (RagQuery): Query parameters for retrieval.

        Returns:
            RagResult: Retrieved results.
        """
        return await retriever.aretrieve(query)

    def query(self, query: Union[str, RagQuery], corpus_id: Optional[str] = None) -> RagResult:
        """Execute a query across indices and return processed results.

        Performs query preprocessing, multi-threaded retrieval, and post-processing.

        Args:
            query (Union[str, RagQuery]): Query string or RagQuery object.
            corpus_id (Optional[str]): Specific corpus to query. If None, queries all corpora.

        Returns:
            RagResult: Retrieved chunks with scores and metadata.

        Raises:
            Exception: If query processing fails.
        """
        try:
            if isinstance(query, str):
                query = RagQuery(query_str=query)
            
            if not self.indices or (corpus_id and corpus_id not in self.indices):
                self.logger.warning(f"No indices found for corpus {corpus_id or 'any'}")
                return RagResult(corpus=Corpus(chunks=[]), scores=[], metadata={"query": query.query_str})

            results = []
            target_corpora = [corpus_id] if corpus_id else self.indices.keys()
            with ThreadPoolExecutor(max_workers=self.config.num_workers or 4) as executor:
                future_to_retriever = {}
                for cid in target_corpora:
                    for idx_type, retriever in self.retrievers[cid].items():
                        if query.metadata_filters and query.metadata_filters.get("index_type") and \
                           query.metadata_filters["index_type"] != idx_type:
                            continue
                        future = executor.submit(
                            asyncio.run, self._retrieve_async(
                                retriever, RagQuery(
                                    query_str=query.query_str,
                                    top_k=query.top_k,
                                    similarity_cutoff=query.similarity_cutoff,
                                    keyword_filters=query.keyword_filters,
                                    metadata_filters=query.metadata_filters
                                )
                            )
                        )
                        future_to_retriever[future] = (cid, idx_type)
                
                for future in as_completed(future_to_retriever):
                    cid, idx_type = future_to_retriever[future]
                    try:
                        result = future.result()
                        results.append(result)
                        self.logger.info(f"Retrieved {len(result.corpus.chunks)} chunks from {idx_type} retriever for corpus {cid}")
                    except Exception as e:
                        self.logger.error(f"Retrieval failed for {idx_type} in corpus {cid}: {str(e)}")

            if not results:
                return RagResult(corpus=Corpus(chunks=[]), scores=[], metadata={"query": query.query_str})

            postprocessor = self.postprocessor_factory.create(
                self.config.retrieval.postprocessor_type,
                query=query
            )
            final_result = postprocessor.postprocess(query, results)

            if query.metadata_filters:
                final_result.corpus.chunks = [
                    chunk for chunk in final_result.corpus.chunks
                    if all(chunk.metadata.model_dump().get(k) == v for k, v in query.metadata_filters.items())
                ]
                final_result.scores = [chunk.metadata.similarity_score for chunk in final_result.corpus.chunks]
                self.logger.info(f"Applied metadata filters, retained {len(final_result.corpus.chunks)} chunks")

            self.logger.info(f"Query returned {len(final_result.corpus.chunks)} chunks after post-processing")
            return final_result
        except Exception as e:
            self.logger.error(f"Query failed: {str(e)}")
            raise