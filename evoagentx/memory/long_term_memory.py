
from .memory import BaseMemory
from ..storages.memoryhadler import MemoryStorageHandler


class LongTermMemory(BaseMemory):

    """
    Responsible for the management of raw data for long-term storage.
    """
    storage: MemoryStorageHandler
    rag_engine = ...
    pass 


