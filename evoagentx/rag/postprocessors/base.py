import logging
from enum import Enum
from typing import List
from abc import ABC, abstractmethod

from ..schema import Query, SchemaResult


class RerankerType(Enum):
    SIMPLE = "simple"
    BGE = "bge"


class BasePostprocessor(ABC):
    """Base interface for post-processors."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def postprocess(self, query: Query, results: List[SchemaResult]) -> SchemaResult:
        """Post-process retrieval results."""
        pass