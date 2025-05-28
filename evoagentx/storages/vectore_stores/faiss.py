import logging
from typing import List, Any, Union, Literal

import faiss
import numpy as np
from llama_index.core.vector_stores import FaissVectorStore
from llama_index.core.vector_stores import VectorStoreQuery, VectorStoreQueryResult

from .base import VectorStoreBase


class FaissVectorStoreWrapper(VectorStoreBase):
    """Wrapper for FAISS vector store."""
    
    def __init__(self, dimension: int = 1536, 
                 index_type: Union[Literal["flat_l2", "ivf_flat"]] = "flat_l2", **kwargs):
        self.dimension = dimension
        self.index_type = index_type
        self.faiss_index = self._create_index()
        self.vector_store = FaissVectorStore(faiss_index=self.faiss_index, **kwargs)
        self.logger = logging.getLogger(__name__)

    def _create_index(self) -> faiss.Index:
        if self.index_type == "flat_l2":
            return faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "ivf_flat":
            quantizer = faiss.IndexFlatL2(self.dimension)
            return faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        else:
            raise ValueError(f"Unsupported FAISS index type: {self.index_type}")
        
    def get_vector_store(self) -> FaissVectorStore:
        """Return the FAISS vector store."""
        return self.vector_store
    
    def add_vectors(self, vectors: List[List[float]], node_ids: List[str]):
        if not vectors or len(vectors[0]) != self.dimension:
            raise ValueError(f"Invalid vector dimension: expected {self.dimension}")
        try:
            vectors_np = np.array(vectors, dtype=np.float32)
            self.faiss_index.add(vectors_np)
            self.vector_store._id_to_node.update({node_id: node_id for node_id in node_ids})
            self.logger.info(f"Added {len(vectors)} vectors to FAISS store")
        except Exception as e:
            self.logger.error(f"Failed to add vectors: {str(e)}")
            raise
    
    def query(self, query: VectorStoreQuery, **kwargs) -> VectorStoreQueryResult:
        try:
            query_vector = np.array([query.query_embedding], dtype=np.float32)
            distances, indices = self.faiss_index.search(query_vector, query.similarity_top_k)
            node_ids = [self.vector_store._id_to_node.get(str(idx)) for idx in indices[0] if str(idx) in self.vector_store._id_to_node]
            scores = [float(dist) for dist in distances[0]]
            result = VectorStoreQueryResult(
                nodes=None,  # Nodes populated by LlamaIndex
                similarities=scores,
                ids=node_ids
            )
            self.logger.info(f"Queried FAISS store, retrieved {len(node_ids)} results")
            return result
        except Exception as e:
            self.logger.error(f"Query failed: {str(e)}")
            raise