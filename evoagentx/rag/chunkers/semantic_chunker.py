import asyncio
import logging
from typing import List
from concurrent.futures import ThreadPoolExecutor

from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser

from .base import BaseChunker
from evoagentx.rag.schema import Document, Corpus, Chunk, ChunkMetadata


class SemanticChunker(BaseChunker):
    """Chunker that splits documents based on semantic similarity.

    Uses LlamaIndex's SemanticChunker with an embedding model to create chunks that preserve
    semantic coherence, ideal for improving retrieval accuracy in RAG pipelines.

    Attributes:
        embed_model (BaseEmbedding): The embedding model for semantic similarity.
        parser (SemanticChunker): The LlamaIndex parser for semantic chunking.
    """

    def __init__(self, embed_model: BaseEmbedding, similarity_threshold: float = 0.7):
        """Initialize the SemanticChunker.

        Args:
            embed_model_name (BaseEmbedding): the embedding model.
            similarity_threshold (float, optional): Threshold for semantic similarity to split chunks (default: 0.7).
        """
        self.embed_model = embed_model
        self.parser = SemanticSplitterNodeParser(
            embed_model=self.embed_model,
            similarity_threshold=similarity_threshold
        )
        self.logger = logging.getLogger(__name__)

    def _process_document(self, doc: Document) -> List[Chunk]:
        """Process a single document into chunks.

        Args:
            doc (Document): The document to chunk.

        Returns:
            List[Chunk]: List of Chunk objects with metadata.
        """
        try:
            llama_doc = doc.to_llama_document()
            llama_doc.metadata["doc_id"] = doc.doc_id

            # Parse document into nodes
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            nodes = loop.run_until_complete(self.parser.aget_nodes_from_documents([llama_doc]))

            # Convert nodes to Chunks
            chunks = []
            for idx, node in enumerate(nodes):
                chunks.append(
                    Chunk(
                        text=node.text,
                        doc_id=doc.doc_id,
                        metadata=ChunkMetadata(
                            doc_id=doc.doc_id,
                            chunk_index=len(chunks),
                            custom_fields={"similarity_threshold": self.parser.similarity_threshold},
                            chunking_strategy="semantic",
                        ),
                        chunk_id=node.id_,
                    )
                )
            self.logger.debug(f"Processed document {doc.doc_id} into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            self.logger.error(f"Failed to process document {doc.doc_id}: {str(e)}")
            return []

    def chunk(self, documents: List[Document], **kwargs) -> Corpus:
        """Chunk documents based on semantic similarity.

        Args:
            documents (List[Document]): List of Document objects to chunk.
            **kwargs: Additional parameters (e.g., max_chunk_size).

        Returns:
            Corpus: A collection of Chunk objects with semantic metadata.
        """
        if not documents:
            self.logger.info("No documents provided, returning empty Corpus")
            return Corpus([])

        chunks = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all documents for processing
            future_to_doc = {executor.submit(self._process_document, doc): doc for doc in documents}
            for future in future_to_doc:
                doc = future_to_doc[future]
                try:
                    chunks.extend(future.result())
                except Exception as e:
                    self.logger.error(f"Error processing document {doc.doc_id}: {str(e)}")

        self.logger.info(f"Chunked {len(documents)} documents into {len(chunks)} chunks")
        return Corpus(chunks=chunks)
    
        llama_docs = [doc.to_llama_document() for doc in documents]
        nodes = self.parser.get_nodes_from_documents(llama_docs)

        chunks = []
        for node in nodes:
            doc_id = node.metadata.get("doc_id", "")
            chunks.append(Chunk(
                text=node.text,
                doc_id=doc_id,
                metadata=ChunkMetadata(
                    doc_id=doc_id,
                    chunk_index=len(chunks),
                    chunking_strategy="semantic",
                    custom_fields={"similarity_threshold": self.parser.similarity_threshold}
                ),
                chunk_id=node.id_
            ))
        return Corpus(chunks=chunks)