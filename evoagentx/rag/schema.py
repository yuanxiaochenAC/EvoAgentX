import logging
from uuid import uuid4
from typing import List, Dict, Optional

from llama_index.core.schema import BaseNode
from llama_index.core import Document as LlamaIndexDocument


class Document:
    """A custom document class wrapping LlamaIndex's Document for user-friendly access.

    Attributes:
        text (str): The content of the document.
        metadata (Dict): Metadata such as title, file_path, file_name, etc.
        source (str): Source of the document (e.g., file path or URL).
        llama_doc (LlamaIndexDocument): The underlying LlamaIndex Document object.
    """

    def __init__(
        self,
        text: str,
        metadata: Optional[Dict] = None,
        source: Optional[str] = None,
        doc_id: Optional[str] = None,
    ):
        self.text = text
        self.metadata = metadata or {}
        self.source = source
        self.doc_id = doc_id or str(uuid4())
        self.llama_doc = LlamaIndexDocument(
            text=self.text,
            metadata=self.metadata,
            id_=self.doc_id,
        )
        self.logger = logging.getLogger(__name__)

    def to_llama_document(self) -> LlamaIndexDocument:
        """Convert to LlamaIndex Document."""
        return self.llama_doc

    @classmethod
    def from_llama_document(cls, llama_doc: LlamaIndexDocument) -> "Document":
        """Create Document from LlamaIndex Document."""
        return cls(
            text=llama_doc.text,
            metadata=llama_doc.metadata,
            source=llama_doc.metadata.get("file_path", ""),
            doc_id=llama_doc.id_,
        )


class Chunk:
    """A single chunk of a document.

    Attributes:
        text (str): The content of the chunk.
        metadata (Dict): Metadata such as chunk_id, parent_id, etc.
        doc_id (str): ID of the parent document.
        llama_node (BaseNode): The underlying LlamaIndex Node object.
    """

    def __init__(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict] = None,
        chunk_id: Optional[str] = None,
    ):
        self.text = text
        self.doc_id = doc_id
        self.metadata = metadata or {}
        self.chunk_id = chunk_id or str(uuid4())
        self.llama_node = BaseNode(
            text=self.text,
            metadata={**self.metadata, "doc_id": self.doc_id},
            id_=self.chunk_id,
        )
        self.logger = logging.getLogger(__name__)

    def to_llama_node(self) -> BaseNode:
        """Convert to LlamaIndex Node."""
        return self.llama_node

    @classmethod
    def from_llama_node(cls, node: BaseNode, doc_id: str) -> "Chunk":
        """Create Chunk from LlamaIndex Node."""
        return cls(
            text=node.text,
            doc_id=doc_id,
            metadata={k: v for k, v in node.metadata.items() if k != "doc_id"},
            chunk_id=node.id_,
        )


class Corpus:
    """A collection of document chunks.

    Attributes:
        chunks (List[Chunk]): List of chunks.
    """

    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.logger = logging.getLogger(__name__)

    def to_llama_nodes(self) -> List[BaseNode]:
        """Convert to list of LlamaIndex Nodes."""
        return [chunk.to_llama_node() for chunk in self.chunks]

    @classmethod
    def from_llama_nodes(cls, nodes: List[BaseNode], doc_id: str) -> "Corpus":
        """Create CustomCorpus from LlamaIndex Nodes."""
        return cls([Chunk.from_llama_node(node, doc_id) for node in nodes])

    def add_chunk(self, chunk: Chunk):
        """Add a chunk to the corpus."""
        self.chunks.append(chunk)
        self.logger.info(f"Added chunk {chunk.chunk_id} to corpus")

