from enum import Enum
from typing import List, Optional, Dict

from pydantic import BaseModel, Field

# Enum for defining default table names used in the database
class TableType(str, Enum):
    """
    Enum representing the default table types for the database.
    Each value corresponds to a specific table name used for storing different types of data.
    """
    store_agent = "agent"      # Table for agent data
    store_workflow = "workflow" # Table for workflow data
    store_memory = "memory"    # Table for memory data
    store_history = "history"  # Table for history data (note: typo 'hisotry' corrected in comments)

# Pydantic model for memory data storage
class MemoryStore(BaseModel):
    """
    Pydantic model representing a memory entry in the database.
    Stores memory-related metadata with optional fields for keywords, entities, and embeddings.
    """
    memory_id: str                      # Unique identifier for the memory
    content: str                       # Main content of the memory
    date: str                          # Date associated with the memory
    key_words: Optional[List] = Field([], description="Optional list of keywords associated with the memory")
    entity_content: Optional[Dict] = Field({}, description="Optional dictionary of entity-related content")
    embedding: Optional[List] = Field([], description="Optional list of embedding vectors")

# Pydantic model for workflow data storage
class WorkflowStore(BaseModel):
    """
    Pydantic model representing a workflow entry in the database.
    Stores workflow metadata with a unique name and content dictionary.
    """
    name: str                           # Unique workflow identifier
    content: Dict                       # Dictionary containing workflow details
    date: str                          # Date associated with the workflow

# Pydantic model for agent data storage
class AgentStore(BaseModel):
    """
    Pydantic model representing an agent entry in the database.
    Stores agent metadata with a unique name and content dictionary.
    """
    name: str                          # Unique agent identifier
    content: Dict                      # Dictionary containing agent details
    date: str                          # Date associated with the agent

# Pydantic model for history data storage
class HistoryStore(BaseModel):
    """
    Pydantic model representing a history entry in the database.
    Stores changes to memory with event details and timestamps.
    """
    memory_id: str                     # Identifier of the memory being modified
    old_memory: str                    # Original memory content before change
    new_memory: str                    # Updated memory content after change
    event: str                         # Description of the event causing the change
    created_at: Optional[str] = Field("", description="Optional timestamp for creation")
    updated_at: Optional[str] = Field("", description="Optional timestamp for last update")