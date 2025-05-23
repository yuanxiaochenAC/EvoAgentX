import logging
from typing import List, Optional, Dict
from uuid import uuid4

from llama_index.core.node_parser import HierarchicalNodeParser, SentenceSplitter
from llama_index.core.schema import NodeRelationship

from .base import BaseChunker
from .simple_chunker import SimpleChunker
from evoagentx.rag.schema import Document, Corpus, Chunk, ChunkMetadata


class HierarchicalChunker(BaseChunker):
    """Enhanced hierarchical chunker with multiple strategies and dynamic chunk size adjustment.

    Creates a multi-level hierarchy of chunks:
    1. Parent nodes: Larger semantic units (e.g., sections, topics).
    2. Child nodes: Smaller chunks for detailed retrieval.

    Dynamically adjusts chunk sizes to accommodate metadata length, ensuring compatibility
    with LlamaIndex's HierarchicalNodeParser.

    Attributes:
        level_parsers (Dict[str, BaseChunker]): Custom parsers for each hierarchy level.
        chunk_sizes (List[int]): Chunk sizes for hierarchical levels (e.g., [2048, 512, 128]).
        chunk_overlap (int): Overlap between adjacent chunks.
        parser (HierarchicalNodeParser): LlamaIndex parser for hierarchical chunking.
        include_metadata (bool): Whether to include metadata in nodes.
        include_prev_next_rel (bool): Whether to include previous/next node relationships.
    """

    def __init__(
        self,
        level_parsers: Dict[str, BaseChunker] = None,
        chunk_sizes: Optional[List[int]] = None,
        chunk_overlap: int = 20,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
    ):
        """Initialize the HierarchicalChunker.

        Args:
            level_parsers (Dict[str, BaseChunker], optional): Custom parsers for hierarchy levels.
            chunk_sizes (List[int], optional): Chunk sizes for levels (default: [2048, 512, 128]).
            chunk_overlap (int): Overlap between adjacent chunks (default: 20).
            include_metadata (bool): Include metadata in nodes (default: True).
            include_prev_next_rel (bool): Include prev/next relationships (default: True).
        """
        self.level_parsers = level_parsers or {}
        self.chunk_sizes = chunk_sizes or [2048, 512, 128]
        self.chunk_overlap = chunk_overlap
        self.include_metadata = include_metadata
        self.include_prev_next_rel = include_prev_next_rel
        self.logger = logging.getLogger(__name__)

        node_parser_ids = None
        node_parser_map = None

        if not self.level_parsers:
            # Default to SimpleChunker for each chunk size
            node_parser_ids = [f"chunk_size_{size}" for size in self.chunk_sizes]
            node_parser_map = {
                node_id: SimpleChunker(
                    chunk_size=size,
                    chunk_overlap=chunk_overlap,
                    include_metadata=include_metadata,
                    include_prev_next_rel=include_prev_next_rel
                ).parser
                for size, node_id in zip(self.chunk_sizes, node_parser_ids)
            }
        else:
            if chunk_sizes is not None:
                raise ValueError("If level_parsers is provided, chunk_sizes should be None.")
            node_parser_ids = list(self.level_parsers.keys())
            node_parser_map = {k: v.parser for k, v in self.level_parsers.items()}

        self.parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=None,
            chunk_overlap=self.chunk_overlap,
            node_parser_ids=node_parser_ids,
            node_parser_map=node_parser_map,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel
        )

    def _calculate_metadata_length(self, doc: Document) -> int:
        """Calculate the length of serialized metadata for a document.

        Args:
            doc (Document): The Document object to analyze.

        Returns:
            int: Length of the serialized metadata string.
        """
        try:
            serialized = doc.metadata.model_dump(exclude_unset=True, mode="json")
            # # Include additional LlamaIndex metadata (e.g., start_char_idx)
            # metadata_dict.update({
            #     "start_char_idx": 0,
            #     "end_char_idx": len(doc.text),
            #     "doc_id": doc.doc_id
            # })
            return len(serialized)
        except Exception as e:
            self.logger.warning(f"Failed to calculate metadata length for doc {doc.doc_id}: {e}")
            return 0

    def _adjust_chunk_sizes(self, documents: List[Document]) -> List[int]:
        """Dynamically adjust chunk sizes based on maximum metadata length.

        Args:
            documents (List[Document]): List of documents to analyze.

        Returns:
            List[int]: Adjusted chunk sizes ensuring smallest size accommodates metadata.
        """
        max_metadata_length = max(self._calculate_metadata_length(doc) for doc in documents)
        buffer = 50  # Extra buffer to avoid edge cases
        min_chunk_size = max_metadata_length + buffer

        adjusted_sizes = sorted(
            [size for size in self.chunk_sizes if size >= min_chunk_size] +
            [min_chunk_size] if min_chunk_size > min(self.chunk_sizes) else [],
            reverse=True
        )

        if not adjusted_sizes:
            adjusted_sizes = [max(min_chunk_size, 512)]  # Ensure at least one size
        if len(adjusted_sizes) < 2:
            adjusted_sizes.insert(0, adjusted_sizes[0] * 2)  # Ensure at least two levels

        if adjusted_sizes != self.chunk_sizes:
            self.logger.info(
                f"Adjusted chunk sizes from {self.chunk_sizes} to {adjusted_sizes} "
                f"due to metadata length {max_metadata_length}"
            )
        return adjusted_sizes

    def chunk(self, documents: List[Document], **kwargs) -> Corpus:
        """Chunk documents using hierarchical strategy with dynamic chunk size adjustment.

        Args:
            documents (List[Document]): List of Document objects to chunk.
            **kwargs: Additional parameters, e.g., custom_metadata for section titles.

        Returns:
            Corpus: A collection of hierarchically organized chunks.
        """
        if not documents:
            return Corpus(chunks=[])

        # Dynamically adjust chunk sizes based on metadata length
        adjusted_chunk_sizes = self._adjust_chunk_sizes(documents)

        # Reinitialize parser with adjusted chunk sizes if necessary
        if adjusted_chunk_sizes != self.chunk_sizes:
            node_parser_ids = [f"chunk_size_{size}" for size in adjusted_chunk_sizes]
            node_parser_map = {
                node_id: SentenceSplitter(
                    chunk_size=size,
                    chunk_overlap=self.chunk_overlap
                )
                for size, node_id in zip(adjusted_chunk_sizes, node_parser_ids)
            }
            self.parser = HierarchicalNodeParser.from_defaults(
                chunk_sizes=None,
                chunk_overlap=self.chunk_overlap,
                node_parser_ids=node_parser_ids,
                node_parser_map=node_parser_map,
                include_metadata=self.include_metadata,
                include_prev_next_rel=self.include_prev_next_rel
            )
            self.chunk_sizes = adjusted_chunk_sizes

        # Convert to LlamaIndex documents and parse
        llama_docs = [d.to_llama_document() for d in documents]
        nodes = self.parser.get_nodes_from_documents(llama_docs)

        chunks = []
        custom_metadata = kwargs.get("custom_metadata", {})  # User-defined metadata

        for i, node in enumerate(nodes):
            doc_id = node.metadata.get("doc_id", "")
            if not doc_id:
                self.logger.warning(f"Node {node.id_} missing doc_id, skipping")
                continue

            # Determine hierarchy level based on chunk size
            node_chunk_size = node.metadata.get("chunk_size", self.chunk_sizes[-1])
            hierarchy_level = (
                self.chunk_sizes.index(node_chunk_size) + 1
                if node_chunk_size in self.chunk_sizes
                else len(self.chunk_sizes)
            )

            # Extract parent node ID for hierarchical relationships
            parent_id = ""
            if node.relationships.get(NodeRelationship.PARENT):
                parent_node = node.relationships[NodeRelationship.PARENT]
                parent_id = parent_node.node_id if parent_node else ""

            # Merge custom metadata
            metadata_fields = {
                "section_title": custom_metadata.get(doc_id, {}).get("section_title", ""),
                "hierarchy_level": hierarchy_level,
                "section_id": str(uuid4()),
                "parent_section_id": parent_id,
                "node_chunk_size": node_chunk_size
            }

            try:
                chunks.append(Chunk(
                    text=node.text,
                    doc_id=doc_id,
                    metadata=ChunkMetadata(
                        doc_id=doc_id,
                        chunk_size=len(node.text),
                        chunk_overlap=self.chunk_overlap,
                        chunk_index=i,
                        chunking_strategy="hierarchical",
                        custom_fields=metadata_fields
                    ),
                    chunk_id=node.id_
                ))
            except Exception as e:
                self.logger.warning(f"Failed to create chunk for node {node.id_}: {e}")
                continue

        return Corpus(chunks=chunks)