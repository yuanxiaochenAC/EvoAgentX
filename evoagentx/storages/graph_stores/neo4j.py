from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.graph_stores.types import PropertyGraphStore

from .base import GraphStoreBase
from evoagentx.core.logging import logger


class Neo4jGraphStoreWrapper(GraphStoreBase):
    """Wrapper for Neo4j graph store."""
    
    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        **kwargs
    ):
        try:
            self.graph_store = Neo4jPropertyGraphStore(
                url=uri,
                username=username,
                password=password,
                database=database,
            )
        except Exception as e:
            raise ValueError(f"Failed to connect to Neo4j: {str(e)}")

        self.verify_version()
    
    def get_graph_store(self) -> PropertyGraphStore:
        return self.graph_store
    
    # borrow from llama_index
    def verify_version(self):
        """
        Check if the connected Neo4j database version supports vector indexing
        without specifying embedding dimension.

        Queries the Neo4j database to retrieve its version and compares it
        against a target version (5.23.0) that is known to support vector
        indexing. 
        """
        db_data = self.graph_store.structured_query("CALL dbms.components()")
        version = db_data[0]["versions"][0]
        if "aura" in version:
            version_tuple = (*map(int, version.split("-")[0].split(".")), 0)
        else:
            version_tuple = tuple(map(int, version.split(".")))

        target_version = (5, 23, 0)

        if version_tuple >= target_version:
            self._supports_vector_index = True
        else:
            self._supports_vector_index = False
            logger.warning(f"The version of Neo4j server is {version_tuple}, which is less than {target_version}. Disable the vector indexing.")