from typing import Dict, Any

from .base import StorageHandler


class MemoryStorageHandler(StorageHandler):
    """The MemoryStorageHandler is responsible for handling the operation with entity database.

    Attributes:
        messages: List of stored Message objects.
        memory_id: Unique identifier for this memory instance.
        timestamp: Creation timestamp of this memory instance.
        capacity: Maximum number of messages that can be stored, or None for unlimited.
    """

    

    def load_memory(self, memory_id: str, **kwargs) -> Dict[str, Any]:
        """
        Load a single long term memory data. 

        Args:
            memory_id (str): the id of the long term memory. 
        
        Returns:
            Dict[str, Any]: the data that can be used to create an LongTermMemory instance. 
        """
        pass


    def save_memory(self, memory_data: Dict[str, Any], **kwargs):
        """
        Save or update a single memory. 

        Args:
            memory_data (Dict[str, Any]): the long term memory's data. 
        """
        pass 
