
import json
import hashlib
from uuid import uuid4
from datetime import datetime
from typing import List, Dict, Optional, Union, Any

from pydantic import BaseModel, Field
from llama_index.core.schema import BaseNode
from llama_index.core import Document as LlamaIndexDocument


class DocumentMetadata(BaseModel):
    """
    This class ensures type safety and validation for metadata associated with a document,
    such as file information, creation date, and custom fields.
    """

    file_path: Optional[str] = Field(
        default=None,
        description="The file path or URL where the document is stored."
    )
    file_type: Optional[str] = Field(
        default=None,
        description="The type of the document (e.g., '.pdf', '.docx', '.md', '.txt')."
    )
    file_name: Optional[str] = Field(
        default=None,
        description="The name of the document file, excluding the path."
    )
    page_count: Optional[int] = Field(
        default=None,
        description="The number of pages in the document, if applicable (e.g., for PDFs)."
    )
    creation_date: Optional[datetime] = Field(
        default=None,
        description="The creation date and time of the document."
    )
    language: Optional[str] = Field(
        default=None,
        description="The primary language of the document (e.g., 'en', 'zh')."
    )
    source: Optional[str] = Field(
        default=None,
        description="The source of the document, such as a file path, URL, or other identifier."
    )
    word_count: Optional[int] = Field(
        default=None,
        description="The number of words in the document, calculated during initialization."
    )
    custom_fields: Dict[str, Any] = Field(
        default_factory=dict,
        description="A dictionary for storing additional user-defined metadata."
    )
    hash_doc: Optional[str] = Field(
        default=None,
        description="The hash code of this Document for deduplication"
    )


class ChunkMetadata(BaseModel):
    """
    This class holds metadata for a chunk, including its relationship to the parent document,
    chunking parameters, and retrieval-related information.
    """

    doc_id: str = Field(
        description="The unique identifier of the parent document."
    )
    chunk_size: Optional[int] = Field(
        default=None,
        description="The size of the chunk in characters, if applicable."
    )
    chunk_overlap: Optional[int] = Field(
        default=None,
        description="The number of overlapping characters between adjacent chunks."
    )
    chunk_index: Optional[int] = Field(
        default=None,
        description="The index of the chunk within the parent document."
    )
    embedding: Optional[List[float]] = Field(
        default=None,
        description="The embedding vector representing the chunk's content."
    )
    similarity_score: Optional[float] = Field(
        default=None,
        description="The similarity score of the chunk during retrieval."
    )
    chunking_strategy: Optional[str] = Field(
        default=None,
        description="The strategy used to create the chunk (e.g., 'simple', 'semantic', 'tree')."
    )
    word_count: Optional[int] = Field(
        default=None,
        description="The number of words in the chunk, calculated during initialization."
    )
    custom_fields: Dict[str, Any] = Field(
        default_factory=dict,
        description="A dictionary for storing additional user-defined metadata."
    )
    source_id: Optional[str] = Field(default=None, description="ID of the source document node.")
    previous_id: Optional[str] = Field(default=None, description="ID of the previous node in the document.")
    next_id: Optional[str] = Field(default=None, description="ID of the next node in the document.")
    parent_id: Optional[str] = Field(default=None, description="ID of the parent node in the hierarchy.")
    child_ids: Optional[List[str]] = Field(default=None, description="List of IDs of child nodes in the hierarchy.")


class Document:
    """A custom document class for managing documents in the RAG pipeline.

    Attributes:
        text (str): The full content of the document.
        doc_id (str): Unique identifier for the document.
        metadata (DocumentMetadata): Metadata including file info, creation date, etc.
        source (str): Source of the document (e.g., file path or URL).
        llama_doc (LlamaIndexDocument): Underlying LlamaIndex Document object.
    """

    def __init__(
        self,
        text: str,
        metadata: Optional[Union[Dict, DocumentMetadata]] = None,
        source: Optional[str] = None,
        doc_id: Optional[str] = None,
    ):
        self.text = text.strip()
        self.doc_id = doc_id or str(uuid4())
        self.metadata = (
            DocumentMetadata.model_validate(metadata) if isinstance(metadata, dict) else metadata or DocumentMetadata()
        )
        self.source = source or self.metadata.file_path or ""   # Fallback to file_path or empty string
        if self.source and not self.metadata.file_path:
            self.metadata.file_path = self.source
        self.metadata.word_count = len(self.text.split())

    def to_llama_document(self) -> LlamaIndexDocument:
        """Convert to LlamaIndex Document."""
        return LlamaIndexDocument(
            text=self.text,
            metadata=self.metadata.model_dump(),
            id_=self.doc_id,
        )

    @classmethod
    def from_llama_document(cls, llama_doc: LlamaIndexDocument) -> "Document":
        """Create Document from LlamaIndex Document."""
        metadata = DocumentMetadata.model_validate(llama_doc.metadata)
        return cls(
            text=llama_doc.text,
            metadata=metadata,
            source=llama_doc.metadata.get("file_path", ""),
            doc_id=llama_doc.id_,
        )

    def compute_hash(self) -> str:
        """Compute a hash of the document text for deduplication."""
        return hashlib.sha256(self.text.encode()).hexdigest()

    def get_fragment(self, max_length: int = 100) -> str:
        """Return a fragment of the document text."""
        return (self.text[:max_length] + "...") if len(self.text) > max_length else self.text

    def to_dict(self) -> Dict:
        """Convert document to dictionary for serialization."""
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "source": self.source,
            "metadata": self.metadata.model_dump(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert document to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def __str__(self) -> str:
        return (
            f"Document(id={self.doc_id}, source={self.source}, "
            f"file_type={self.metadata.file_type}, word_count={self.metadata.word_count}, "
            f"fragment={self.get_fragment(max_length=50)})"
        )

    def __repr__(self) -> str:
        return (
            f"Document(doc_id={self.doc_id}, source={self.source}, metadata={self.metadata.model_dump()},"
            f"fragment={self.get_fragment(max_length=50)})"
        )


class Chunk:
    """A single chunk of a document for RAG processing.

    Attributes:
        text (str): The content of the chunk.
        doc_id (str): ID of the parent document.
        chunk_id (str): Unique identifier for the chunk.
        metadata (ChunkMetadata): Metadata including chunk size, embedding, etc.
        llama_node (BaseNode): Underlying LlamaIndex Node object.
    """

    def __init__(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Union[Dict, ChunkMetadata]] = None,
        chunk_id: Optional[str] = None,
    ):
        self.text = text.strip()
        self.doc_id = doc_id
        self.chunk_id = chunk_id or str(uuid4())
        self.metadata = (
            ChunkMetadata.model_validate(metadata) if isinstance(metadata, dict) else metadata or  ChunkMetadata(doc_id=doc_id)
        )
        self.metadata.word_count = len(self.text.split())

    def to_llama_node(self) -> BaseNode:
        """Convert to LlamaIndex Node."""
        relationships = {}
        if self.metadata.source_id:
            relationships["SOURCE"] = self.metadata.source_id
        if self.metadata.previous_id:
            relationships["PREVIOUS"] = self.metadata.previous_id
        if self.metadata.next_id:
            relationships["NEXT"] = self.metadata.next_id
        if self.metadata.parent_id:
            relationships["PARENT"] = self.metadata.parent_id
        if self.metadata.child_ids:
            relationships["CHILD"] = self.metadata.child_ids

        return BaseNode(
            text=self.text,
            metadata=self.metadata.model_dump(),
            id_=self.chunk_id,
            relationships=relationships,
        )

    @classmethod
    def from_llama_node(cls, node: BaseNode, doc_id: str) -> "Chunk":
        """Create Chunk from LlamaIndex Node."""
        metadata = {k: v for k, v in node.metadata.items() if k != "doc_id"}
        metadata.update({
            "source_id": node.relationships.get("SOURCE", None),
            "previous_id": node.relationships.get("PREVIOUS", None),
            "next_id": node.relationships.get("NEXT", None),
            "parent_id": node.relationships.get("PARENT", None),
            "child_ids": node.relationships.get("CHILD", None),
        })
        return cls(
            text=node.text,
            doc_id=doc_id,
            metadata=ChunkMetadata(doc_id=doc_id, **metadata),
            chunk_id=node.id_,
        )

    def set_embedding(self, embedding: List[float]):
        """Set the embedding vector for the chunk."""
        self.metadata.embedding = embedding
        self.llama_node.metadata["embedding"] = embedding

    def set_similarity_score(self, score: float):
        """Set the similarity score for retrieval."""
        self.metadata.similarity_score = score
        self.llama_node.metadata["similarity_score"] = score

    def get_fragment(self, max_length: int = 100) -> str:
        """Return a fragment of the chunk text."""
        return (self.text[:max_length] + "...") if len(self.text) > max_length else self.text

    def to_dict(self) -> Dict:
        """Convert chunk to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "metadata": self.metadata.model_dump(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert chunk to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def __str__(self) -> str:
        return (
            f"Chunk(id={self.chunk_id}, doc_id={self.doc_id}, "
            f"chunking_strategy={self.metadata.chunking_strategy}, "
            f"word_count={self.metadata.word_count}, "
            f"similarity_score={self.metadata.similarity_score}, "
            f"fragment={self.get_fragment(max_length=50)})"
        )

    def __repr__(self) -> str:
        return f"Chunk(chunk_id={self.chunk_id}, doc_id={self.doc_id}, metadata={self.metadata.model_dump()})"


class Corpus:
    """A collection of document chunks for RAG processing.

    Attributes:
        chunks (List[Chunk]): List of chunks in the corpus.
        chunk_index (Dict[str, Chunk]): Index of chunks by chunk_id for fast lookup.
    """

    def __init__(self, chunks: List[Chunk] = None):
        self.chunks = chunks or []
        self.chunk_index = {chunk.chunk_id: chunk for chunk in self.chunks}

    def to_llama_nodes(self) -> List[BaseNode]:
        """Convert to list of LlamaIndex Nodes."""
        return [chunk.to_llama_node() for chunk in self.chunks]

    @classmethod
    def from_llama_nodes(cls, nodes: List[BaseNode], doc_id: str) -> "Corpus":
        """Create a Corpus from a list of LlamaIndex Nodes.

        Args:
            nodes (List[BaseNode]): The LlamaIndex Nodes to convert.
            doc_id (str): The ID of the parent document.

        Returns:
            Corpus: A new Corpus instance.
        """
        return cls([Chunk.from_llama_node(node, doc_id) for node in nodes])

    def add_chunk(self, chunk: Chunk):
        """Add a chunk to the corpus and update index."""
        self.chunks.append(chunk)
        self.chunk_index[chunk.chunk_id] = chunk

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Retrieve a chunk by its ID."""
        return self.chunk_index.get(chunk_id)

    def remove_chunk(self, chunk_id: str):
        """Remove a chunk by its ID."""
        self.chunks = [chunk for chunk in self.chunks if chunk.chunk_id != chunk_id]
        self.chunk_index.pop(chunk_id, None)

    def filter_by_doc_id(self, doc_id: str) -> List[Chunk]:
        """Filter chunks by parent document ID."""
        return [chunk for chunk in self.chunks if chunk.doc_id == doc_id]

    def filter_by_similarity(self, threshold: float) -> List[Chunk]:
        """Filter chunks by similarity score."""
        return [chunk for chunk in self.chunks if chunk.metadata.similarity_score and chunk.metadata.similarity_score >= threshold]

    def sort_by_similarity(self, reverse: bool = True) -> List[Chunk]:
        """Sort chunks by similarity score (descending by default)."""
        return sorted(
            [chunk for chunk in self.chunks if chunk.metadata.similarity_score is not None],
            key=lambda x: x.metadata.similarity_score,
            reverse=reverse
        )

    def get_stats(self) -> Dict:
        """Return statistics about the corpus."""
        return {
            "chunk_count": len(self.chunks),
            "unique_docs": len(set(chunk.doc_id for chunk in self.chunks)),
            "avg_word_count": sum(chunk.metadata.word_count or 0 for chunk in self.chunks) / len(self.chunks) if self.chunks else 0,
            "strategies": list(set(chunk.metadata.chunking_strategy for chunk in self.chunks if chunk.metadata.chunking_strategy))
        }

    def to_dict(self) -> Dict:
        """Convert corpus to dictionary for serialization."""
        return {
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "stats": self.get_stats(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert corpus to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def __str__(self) -> str:
        stats = self.get_stats()
        return (
            f"Corpus(chunks={stats['chunk_count']}, unique_docs={stats['unique_docs']}, "
            f"avg_word_count={stats['avg_word_count']:.1f}, strategies={stats['strategies']})"
        )

    def __repr__(self) -> str:
        return f"Corpus(chunks={len(self.chunks)}, chunk_index_keys={list(self.chunk_index.keys())})"

    def __len__(self) -> int:
        return len(self.chunks)