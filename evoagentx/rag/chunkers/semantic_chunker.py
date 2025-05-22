from typing import List

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

    def chunk(self, documents: List[Document], **kwargs) -> Corpus:
        """Chunk documents based on semantic similarity.

        Args:
            documents (List[Document]): List of Document objects to chunk.
            **kwargs: Additional parameters (e.g., max_chunk_size).

        Returns:
            Corpus: A collection of Chunk objects with semantic metadata.
        """
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