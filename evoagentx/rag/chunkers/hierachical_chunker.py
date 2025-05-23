from typing import List

from llama_index.core.node_parser import HierarchicalNodeParser

from .base import BaseChunker
from evoagentx.rag.schema import Corpus, Document


class HierarchicalChunker(BaseChunker):
    def __init__(self) -> None:
        self.parser = HierarchicalNodeParser()

    def chunk(self, documents: List[Document], **kwargs) -> Corpus:
        return 