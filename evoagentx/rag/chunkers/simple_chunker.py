import asyncio
import logging
from typing import List
from concurrent.futures import ThreadPoolExecutor

from llama_index.core.node_parser import SimpleNodeParser
from .base import BaseChunker
from evoagentx.rag.schema import Document, Corpus, Chunk, ChunkMetadata


class SimpleChunker(BaseChunker):
    """Chunker that splits documents into fixed-size chunks using multi-threading and async parsing.

    Uses LlamaIndex's SimpleNodeParser with async support to create chunks with a specified size
    and overlap, suitable for general-purpose text splitting in RAG pipelines.

    Attributes:
        chunk_size (int): The target size of each chunk in characters.
        chunk_overlap (int): The number of overlapping characters between adjacent chunks.
        parser (SimpleNodeParser): The LlamaIndex parser for chunking.
        max_workers (int): Maximum number of threads for parallel processing.
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 20,
        tokenizer=None,
        chunking_tokenizer_fn=None,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        max_workers: int = 4,
    ):
        """Initialize the SimpleChunker.

        Args:
            chunk_size (int, optional): Target size of each chunk in characters (default: 1024).
            chunk_overlap (int, optional): Overlap between adjacent chunks in characters (default: 20).
            tokenizer: Optional tokenizer for chunking.
            chunking_tokenizer_fn: Optional tokenizer function for chunking.
            include_metadata (bool): Whether to include metadata in nodes (default: True).
            include_prev_next_rel (bool): Whether to include previous/next relationships (default: True).
            max_workers (int): Maximum number of threads for parallel processing (default: 4).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tokenizer
        self.chunking_tokenizer_fn = chunking_tokenizer_fn
        self.max_workers = max_workers
        self.parser = SimpleNodeParser(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=tokenizer,
            chunking_tokenizer_fn=chunking_tokenizer_fn,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
        )
        self.logger = logging.getLogger(__name__)

    async def _process_document_async(self, doc: Document) -> List[Chunk]:
        """Asynchronously process a single document into chunks.

        Args:
            doc (Document): The document to chunk.

        Returns:
            List[Chunk]: List of Chunk objects with metadata.
        """
        try:
            # Convert to LlamaIndex Document
            llama_doc = doc.to_llama_document()
            llama_doc.metadata["doc_id"] = doc.doc_id

            # Parse document into nodes using async method
            nodes = await self.parser.aget_nodes_from_documents([llama_doc])

            # Convert nodes to Chunks
            chunks = []
            for idx, node in enumerate(nodes):
                chunks.append(
                    Chunk(
                        text=node.text,
                        doc_id=doc.doc_id,
                        metadata=ChunkMetadata(
                            doc_id=doc.doc_id,
                            chunk_size=self.chunk_size,
                            chunk_overlap=self.chunk_overlap,
                            chunk_index=idx,
                            chunking_strategy="simple",
                        ),
                        chunk_id=node.id_,
                    )
                )
            self.logger.debug(f"Processed document {doc.doc_id} into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            self.logger.error(f"Failed to process document {doc.doc_id}: {str(e)}")
            return []

    def _process_document(self, doc: Document) -> List[Chunk]:
        """Process a single document into chunks in a thread.

        Args:
            doc (Document): The document to chunk.

        Returns:
            List[Chunk]: List of Chunk objects with metadata.
        """
        return asyncio.run(self._process_document_async(doc))

    def chunk(self, documents: List[Document], **kwargs) -> Corpus:
        """Chunk documents into fixed-size chunks using multi-threading.

        Args:
            documents (List[Document]): List of Document objects to chunk.

        Returns:
            Corpus: A collection of Chunk objects with metadata.
        """
        if not documents:
            self.logger.info("No documents provided, returning empty Corpus")
            return Corpus([])

        chunks = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_doc = {executor.submit(self._process_document, doc): doc for doc in documents}
            for future in future_to_doc:
                doc = future_to_doc[future]
                try:
                    chunks.extend(future.result())
                except Exception as e:
                    self.logger.error(f"Error processing document {doc.doc_id}: {str(e)}")

        self.logger.info(f"Chunked {len(documents)} documents into {len(chunks)} chunks")
        return Corpus(chunks=chunks)