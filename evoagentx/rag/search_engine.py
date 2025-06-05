import os
import asyncio
import logging
from uuid import uuid4
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Optional, Sequence, Dict, Any, Tuple, Literal

from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
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
             index_type: Optional[IndexType] = None) -> None:
        """
        Save indices to database or file system.
        
        Args:
            output_path: Directory to save index files (if None, saves to database).
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
                    storage_type = "graph" if index.index_type == IndexType.GRAPH else "vector"
                    storage_context = self.storage_handler.storage_contexts[cid]
                    docstore = storage_context.docstore
                    index_store = storage_context.index_store
                    metadata = {
                        "collection_name": f"corpus_{cid}" if storage_type == "vector" else None,
                        "graph_uri": self.storage_handler.storageConfig.graphConfig.uri if storage_type == "graph" else None,
                        "persist_path": os.path.join(output_path or "", f"corpus_{cid}/{index.index_type}") if output_path else None
                    }
                    if output_path:
                        # Save to file system
                        persist_dir = os.path.join(output_path, f"corpus_{cid}/{index.index_type}")
                        os.makedirs(persist_dir, exist_ok=True)
                        storage_context.persist(persist_dir=persist_dir)
                        self.logger.info(f"Saved {index.index_type} index for corpus {cid} to {persist_dir}")
                    else:
                        # Save to database
                        content = {
                            "docstore": docstore.to_dict(),
                            "index_store": index_store.to_dict()
                        }
                        self.storage_handler.save_index(
                            index_data={
                                "index_id": index_id,
                                "corpus_id": cid,
                                "index_type": index.index_type,
                                "storage_type": storage_type,
                                "content": content,
                                "date": datetime.now().isoformat(),
                                "key_words": [cid, index.index_type],
                                "metadata": metadata
                            }
                        )
                        self.logger.info(f"Saved {index.index_type} index for corpus {cid} to database")
            self.logger.info(f"Saved indices for {len(target_corpora)} corpora")
        except Exception as e:
            self.logger.error(f"Failed to save indices: {str(e)}")
            raise

    def load(self, source: Optional[str] = None, corpus_id: Optional[str] = None,
             index_type: Optional[IndexType] = None) -> None:
        """
        Load indices from database or file system.
        
        Args:
            source: Directory containing index files or None for database.
            corpus_id: Specific corpus to load (default: all corpora).
            index_type: Specific index type to load (default: all types).
        """
        try:
            if source:
                # Load from file system
                os.makedirs(source, exist_ok=True)
                corpora_dirs = [d for d in os.listdir(source) if os.path.isdir(os.path.join(source, d)) and d.startswith("corpus_")]
                for corpus_dir in corpora_dirs:
                    cid = corpus_dir.replace("corpus_", "")
                    if corpus_id and corpus_id != cid:
                        continue
                    index_dirs = [d for d in os.listdir(os.path.join(source, corpus_dir)) if os.path.isdir(os.path.join(source, corpus_dir, d))]
                    for index_type_str in index_dirs:
                        if index_type and index_type != IndexType(index_type_str):
                            continue
                        if cid not in self.indices:
                            self.indices[cid] = {}
                            self.retrievers[cid] = {}
                        self.storage_handler._init_vector_store(corpus_id=cid)
                        persist_dir = os.path.join(source, corpus_dir, index_type_str)
                        index = self.index_factory.create(
                            index_type=IndexType(index_type_str),
                            embed_model=self.embed_model,
                            storage_context=self.storage_handler.storage_contexts[cid],
                            index_config=self.config.index_config
                        )
                        # Rebuild index from persisted storage
                        storage_context = self.storage_handler.storage_contexts[cid]
                        index.get_index().storage_context.docstore = SimpleDocumentStore.from_persist_dir(persist_dir)
                        index.get_index().storage_context.index_store = SimpleIndexStore.from_persist_dir(persist_dir)
                        index.get_index().storage_context.vector_store.load_from_persist_dir(persist_dir)
                        self.indices[cid][IndexType(index_type_str)] = index
                        self.retrievers[cid][IndexType(index_type_str)] = self.retriever_factory.create(
                            retriever_type=RetrieverType.VECTOR if index_type_str in [IndexType.VECTOR, IndexType.SUMMARY, IndexType.TREE] else RetrieverType.GRAPH,
                            index=index.get_index() if index_type_str in [IndexType.VECTOR, IndexType.SUMMARY, IndexType.TREE] else None,
                            graph_store=self.storage_handler.storage_contexts[cid].graph_store if index_type_str == IndexType.GRAPH else None,
                            embed_model=self.embed_model,
                            query=RagQuery(query_str="", top_k=self.config.retrieval_config.get("top_k", 5))
                        )
                        self.logger.info(f"Loaded {index_type_str} index for corpus {cid} from {persist_dir}")
            else:
                # Load from database
                results = self.storage_handler.load(tables=["index"])
                for record in results.get("index", []):
                    parsed_record = self.storage_handler.parse_result(record, IndexStore)
                    index_type_str = parsed_record["index_type"]
                    cid = parsed_record["corpus_id"]
                    if (corpus_id and corpus_id != cid) or (index_type and index_type != index_type_str):
                        continue
                    if cid not in self.indices:
                        self.indices[cid] = {}
                        self.retrievers[cid] = {}
                    collection_name = parsed_record["metadata"].get("collection_name", f"corpus_{cid}")
                    self.storage_handler._init_vector_store(corpus_id=cid)
                    storage_context = self.storage_handler.storage_contexts[cid]
                    # Restore docstore and index_store
                    storage_context.docstore = SimpleDocumentStore.from_dict(parsed_record["content"]["docstore"])
                    storage_context.index_store = SimpleIndexStore.from_dict(parsed_record["content"]["index_store"])
                    index = self.index_factory.create(
                        index_type=IndexType(index_type_str),
                        embed_model=self.embed_model,
                        storage_context=storage_context,
                        index_config=self.config.index_config
                    )
                    self.indices[cid][IndexType(index_type_str)] = index
                    self.retrievers[cid][IndexType(index_type_str)] = self.retriever_factory.create(
                        retriever_type=RetrieverType.VECTOR if index_type_str in [IndexType.VECTOR, IndexType.SUMMARY, IndexType.TREE] else RetrieverType.GRAPH,
                        index=index.get_index() if index_type_str in [IndexType.VECTOR, IndexType.SUMMARY, IndexType.TREE] else None,
                        graph_store=storage_context.graph_store if index_type_str == IndexType.GRAPH else None,
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
            corpus_id: Identifier for the corpus.
        """
        try:
            if corpus_id not in self.indices:
                self.indices[corpus_id] = {}
                self.retrievers[corpus_id] = {}
                self.storage_handler._init_vector_store(corpus_id=corpus_id)

            if index_type not in self.indices[corpus_id]:
                index = self.index_factory.create(
                    index_type=index_type,
                    embed_model=self.embed_model,
                    storage_context=self.storage_handler.storage_contexts[corpus_id],
                    index_config=self.config.index.model_dump()
                )
                self.indices[corpus_id][index_type] = index
                self.retrievers[corpus_id][index_type] = self.retriever_factory.create(
                    retriever_type=RetrieverType.VECTOR if index_type in [IndexType.VECTOR, IndexType.SUMMARY, IndexType.TREE] else RetrieverType.GRAPH,
                    index=index.get_index() if index_type in [IndexType.VECTOR, IndexType.SUMMARY, IndexType.TREE] else None,
                    graph_store=self.storage_handler.storage_contexts[corpus_id].graph_store if index_type == IndexType.GRAPH else None,
                    embed_model=self.embed_model,
                    query=RagQuery(query_str="", top_k=self.config.retrieval.top_k)
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
                storage_context = self.storage_handler.storage_contexts[corpus_id]
                if node_ids or metadata_filters:
                    if idx_type == IndexType.GRAPH:
                        if not storage_context.graph_store:
                            self.logger.error(f"No graph store configured for {idx_type} index")
                            continue
                        nodes_to_delete = []
                        all_nodes = storage_context.graph_store.get_all_nodes()
                        for node in all_nodes:
                            if node_ids and node.node_id in node_ids:
                                nodes_to_delete.append(node.node_id)
                            elif metadata_filters and all(
                                node.metadata.get(k) == v for k, v in metadata_filters.items()
                            ):
                                nodes_to_delete.append(node.node_id)
                        for node_id in nodes_to_delete:
                            storage_context.graph_store.delete(node_id)
                        self.logger.info(f"Deleted {len(nodes_to_delete)} nodes from {idx_type} index for corpus {corpus_id}")
                    else:
                        nodes_to_delete = []
                        docstore = storage_context.docstore
                        all_nodes = docstore.get_all_nodes()
                        for node in all_nodes:
                            if node_ids and node.node_id in nodes_to_delete:
                                nodes_to_delete.append(node.node_id)
                            elif metadata_filters and all(
                                node.metadata.get(k) == v for k, v in metadata_filters.items()
                            ):
                                nodes_to_delete.append(node.node_id)
                        for node_id in nodes_to_delete:
                            index.get_index().delete_ref_doc(node_id, delete_from_docstore=True)
                        self.logger.info(f"Deleted {len(nodes_to_delete)} nodes from {idx_type} index for corpus {corpus_id}")
                else:
                    # Delete entire index
                    del self.indices[corpus_id][idx_type]
                    del self.retrievers[corpus_id][idx_type]
                    self.logger.info(f"Deleted {idx_type} index for corpus {corpus_id}")

            if not self.indices[corpus_id]:
                del self.indices[corpus_id]
                del self.retrievers[corpus_id]
                del self.storage_handler.storageVectors[corpus_id]
                del self.storage_handler.storage_contexts[corpus_id]
                self.logger.info(f"Removed empty corpus {corpus_id}")

            if self.storage_handler.storage_contexts.get(corpus_id):
                self.storage_handler.storage_contexts[corpus_id].persist()
            self.logger.info(f"Updated persistent storage for corpus {corpus_id}")
        except Exception as e:
            self.logger.error(f"Failed to delete from corpus {corpus_id}: {str(e)}")
            raise

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