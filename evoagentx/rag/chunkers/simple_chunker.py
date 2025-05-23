from typing import List

from llama_index.core.node_parser import SimpleNodeParser

from .base import BaseChunker
from evoagentx.rag.schema import Document, Corpus, Chunk, ChunkMetadata


class SimpleChunker(BaseChunker):
    """Chunker that splits documents into fixed-size chunks.

    Uses LlamaIndex's SimpleNodeParser to create chunks with a specified size and overlap,
    suitable for general-purpose text splitting in RAG pipelines.

    Attributes:
        chunk_size (int): The target size of each chunk in characters.
        chunk_overlap (int): The number of overlapping characters between adjacent chunks.
        parser (SimpleNodeParser): The LlamaIndex parser for chunking.
    """

    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 20, 
                 tokenizer=None, chunking_tokenizer_fn=None):
        """Initialize the SimpleChunker.

        Args:
            chunk_size (int, optional): Target size of each chunk in characters (default: 1024).
            chunk_overlap (int, optional): Overlap between adjacent chunks in characters (default: 20).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tokenizer
        self.chunking_tokenizer_fn = chunking_tokenizer_fn
        self.parser = SimpleNodeParser(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                       tokenizer=self.tokenizer, chunking_tokenizer_fn=self.chunking_tokenizer_fn)

    def chunk(self, documents: List[Document], **kwargs) -> Corpus:
        """Chunk documents into fixed-size chunks.

        Args:
            documents (List[Document]): List of Document objects to chunk.

        Returns:
            Corpus: A collection of Chunk objects with metadata.
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
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    chunk_index=len(chunks),
                    chunking_strategy="simple"
                ),
                chunk_id=node.id_
            ))
        return Corpus(chunks=chunks)