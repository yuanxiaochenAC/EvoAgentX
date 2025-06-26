"""
Database connection and models for EvoAgentX.
"""
import logging
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Protocol
from abc import ABC, abstractmethod
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, TEXT, ReturnDocument
from pydantic_core import core_schema
from bson import ObjectId
from pydantic import GetCoreSchemaHandler
from pydantic import Field, BaseModel
from evoagentx.app.config import settings

# Setup logger
logger = logging.getLogger(__name__)

# Custom PyObjectId for MongoDB ObjectId compatibility with Pydantic
class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler: GetCoreSchemaHandler):
        return core_schema.no_info_after_validator_function(cls.validate, core_schema.str_schema())

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

# Base model with ObjectId handling
class MongoBaseModel(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)

    model_config = {
        "protected_namespaces": (),
        "populate_by_name": True,  # Replace `allow_population_by_field_name`
        "arbitrary_types_allowed": True,  # Keep custom types like ObjectId
        "json_encoders": {
            ObjectId: str  # Ensure ObjectId is serialized as a string
        }
    }

# Status Enums
class AgentStatus(str, Enum):
    CREATED = "created"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"

class WorkflowStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

# Database Models
class Agent(MongoBaseModel):
    id: str = Field(..., alias="_id")
    name: str
    description: Optional[str] = None
    config: Dict[str, Any]
    state: Dict[str, Any] = Field(default_factory=dict)
    runtime_params: Dict[str, Any] = Field(default_factory=dict)
    status: AgentStatus = AgentStatus.CREATED
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

class Workflow(MongoBaseModel):
    id: str = Field(..., alias="_id")
    name: str
    description: Optional[str] = None
    definition: Dict[str, Any]
    agent_ids: List[str] = Field(default_factory=list)
    status: WorkflowStatus = WorkflowStatus.CREATED
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    version: int = 1

class ExecutionLog(MongoBaseModel):
    workflow_id: str
    execution_id: str
    step_id: Optional[str] = None
    agent_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    level: str = "INFO"
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)

class WorkflowExecution(MongoBaseModel):
    workflow_id: str
    status: ExecutionStatus = ExecutionStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    input_params: Dict[str, Any] = Field(default_factory=dict)
    results: Dict[str, Any] = Field(default_factory=dict)
    created_by: Optional[str] = None
    step_results: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    current_step: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

# Abstract Database Interface
class Database(ABC):
    """Abstract base class for database implementations."""
    
    @abstractmethod
    async def connect(self) -> None:
        """Connect to the database."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the database."""
        pass
    
    @abstractmethod
    async def ping(self) -> bool:
        """Check if database connection is alive."""
        pass
    
    @abstractmethod
    async def search(self, collection: str, query: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:
        """Search for documents in a collection."""
        pass
    
    @abstractmethod
    async def write(self, collection: str, data: Dict[str, Any]) -> str:
        """Write a document to a collection."""
        pass
    
    @abstractmethod
    async def update(self, collection: str, filter_query: Dict[str, Any], update_data: Dict[str, Any], **kwargs) -> bool:
        """Update documents in a collection."""
        pass
    
    @abstractmethod
    async def delete(self, collection: str, query: Dict[str, Any]) -> int:
        """Delete documents from a collection."""
        pass
    
    @abstractmethod
    async def count(self, collection: str, query: Dict[str, Any] = None) -> int:
        """Count documents in a collection."""
        pass
    
    @abstractmethod
    async def find_one(self, collection: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find a single document in a collection."""
        pass

# MongoDB Implementation
class MongoDatabase(Database):
    """MongoDB implementation of the Database interface."""
    
    def __init__(self, url: str, db_name: str):
        self.url = url
        self.db_name = db_name
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
    
    # Collections
        self.agents = None
        self.workflows = None
        self.executions = None
        self.logs = None
        self.users = None
    
    async def connect(self) -> None:
        """Connect to MongoDB"""
        logger.info(f"Connecting to MongoDB at {self.url}...")
        self.client = AsyncIOMotorClient(self.url)
        self.db = self.client[self.db_name]
        
        # Set up collections
        self.agents = self.db.agents
        self.workflows = self.db.workflows
        self.executions = self.db.workflow_executions
        self.logs = self.db.execution_logs
        self.users = self.db.users
        
        # Create indexes
        await self._create_indexes()
        
        logger.info("Connected to MongoDB successfully")
    
    async def disconnect(self) -> None:
        """Disconnect from MongoDB"""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")
    
    async def ping(self) -> bool:
        """Check if MongoDB connection is alive."""
        try:
            await self.db.command('ping')
            return True
        except Exception as e:
            logger.error(f"Database ping failed: {e}")
            return False
    
    async def search(self, collection: str, query: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:
        """Search for documents in a MongoDB collection."""
        collection_obj = getattr(self, collection, None)
        if collection_obj is None:
            collection_obj = self.db[collection]
        
        limit = kwargs.get('limit', 100)
        skip = kwargs.get('skip', 0)
        sort = kwargs.get('sort', [])
        
        cursor = collection_obj.find(query)
        
        if sort:
            cursor = cursor.sort(sort)
        
        cursor = cursor.skip(skip).limit(limit)
        
        return await cursor.to_list(length=limit)
    
    async def write(self, collection: str, data: Dict[str, Any]) -> str:
        """Write a document to a MongoDB collection."""
        collection_obj = getattr(self, collection, None)
        if collection_obj is None:
            collection_obj = self.db[collection]
        result = await collection_obj.insert_one(data)
        return str(result.inserted_id)
    
    async def update(self, collection: str, filter_query: Dict[str, Any], update_data: Dict[str, Any], **kwargs) -> bool:
        """Update documents in a MongoDB collection."""
        collection_obj = getattr(self, collection, None)
        if collection_obj is None:
            collection_obj = self.db[collection]
        
        upsert = kwargs.get('upsert', False)
        many = kwargs.get('many', False)
        
        if many:
            result = await collection_obj.update_many(filter_query, update_data, upsert=upsert)
        else:
            result = await collection_obj.update_one(filter_query, update_data, upsert=upsert)
        
        return result.modified_count > 0 or result.upserted_id is not None
    
    async def delete(self, collection: str, query: Dict[str, Any]) -> int:
        """Delete documents from a MongoDB collection."""
        collection_obj = getattr(self, collection, None)
        if collection_obj is None:
            collection_obj = self.db[collection]
        result = await collection_obj.delete_many(query)
        return result.deleted_count
    
    async def count(self, collection: str, query: Dict[str, Any] = None) -> int:
        """Count documents in a MongoDB collection."""
        collection_obj = getattr(self, collection, None)
        if collection_obj is None:
            collection_obj = self.db[collection]
        query = query or {}
        return await collection_obj.count_documents(query)
    
    async def find_one(self, collection: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find a single document in a MongoDB collection."""
        collection_obj = getattr(self, collection, None)
        if collection_obj is None:
            collection_obj = self.db[collection]
        return await collection_obj.find_one(query)
    
    async def find_one_and_update(self, collection: str, filter_query: Dict[str, Any], update_data: Dict[str, Any], **kwargs) -> Optional[Dict[str, Any]]:
        """Find and update a single document in a MongoDB collection."""
        collection_obj = getattr(self, collection, None)
        if collection_obj is None:
            collection_obj = self.db[collection]
        
        # Default to returning the updated document
        if 'return_document' in kwargs and kwargs['return_document'] is True:
            kwargs['return_document'] = ReturnDocument.AFTER
        elif 'return_document' not in kwargs:
            kwargs['return_document'] = ReturnDocument.AFTER
            
        return await collection_obj.find_one_and_update(filter_query, update_data, **kwargs)
    
    async def create_index(self, collection: str, index_spec, **kwargs):
        """Create an index on a MongoDB collection."""
        collection_obj = getattr(self, collection, None)
        if collection_obj is None:
            collection_obj = self.db[collection]
        return await collection_obj.create_index(index_spec, **kwargs)
    
    async def _create_indexes(self):
        """Create indexes for collections"""
        # Agent indexes
        await self.agents.create_index([("name", ASCENDING)], unique=True)
        await self.agents.create_index([("name", TEXT), ("description", TEXT)])
        await self.agents.create_index([("created_at", ASCENDING)])
        await self.agents.create_index([("tags", ASCENDING)])
        
        # Workflow indexes
        await self.workflows.create_index([("name", ASCENDING)])
        await self.workflows.create_index([("name", TEXT), ("description", TEXT)])
        await self.workflows.create_index([("created_at", ASCENDING)])
        await self.workflows.create_index([("agent_ids", ASCENDING)])
        await self.workflows.create_index([("tags", ASCENDING)])
        
        # Execution indexes
        await self.executions.create_index([("workflow_id", ASCENDING)])
        await self.executions.create_index([("created_at", ASCENDING)])
        await self.executions.create_index([("status", ASCENDING)])
        
        # Log indexes
        await self.logs.create_index([("execution_id", ASCENDING)])
        await self.logs.create_index([("timestamp", ASCENDING)])
        await self.logs.create_index([("workflow_id", ASCENDING), ("execution_id", ASCENDING)])

# NoSQL/In-Memory Database Implementation
class NoDatabase(Database):
    """No-database implementation for testing or simple use cases."""
    
    def __init__(self):
        self.connected = False
        self.collections: Dict[str, List[Dict[str, Any]]] = {}
        self._id_counter = 0
    
    async def connect(self) -> None:
        """Connect to the no-database (just set flag)."""
        self.connected = True
        logger.info("No-database connected (in-memory storage)")
    
    async def disconnect(self) -> None:
        """Disconnect from the no-database."""
        self.connected = False
        self.collections.clear()
        logger.info("No-database disconnected")
    
    async def ping(self) -> bool:
        """Check if no-database connection is alive."""
        return self.connected
    
    async def search(self, collection: str, query: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:
        """Search for documents in the in-memory collection."""
        if collection not in self.collections:
            return []
        
        results = []
        for doc in self.collections[collection]:
            if self._matches_query(doc, query):
                results.append(doc.copy())
        
        # Apply limit and skip
        skip = kwargs.get('skip', 0)
        limit = kwargs.get('limit', 100)
        
        return results[skip:skip + limit]
    
    async def write(self, collection: str, data: Dict[str, Any]) -> str:
        """Write a document to the in-memory collection."""
        if collection not in self.collections:
            self.collections[collection] = []
        
        # Generate ID if not present
        if '_id' not in data:
            # Generate a valid ObjectId string instead of simple counter
            data['_id'] = str(ObjectId())
        
        self.collections[collection].append(data.copy())
        return str(data['_id'])
    
    async def update(self, collection: str, filter_query: Dict[str, Any], update_data: Dict[str, Any], **kwargs) -> bool:
        """Update documents in the in-memory collection."""
        if collection not in self.collections:
            return False
        
        updated = False
        for doc in self.collections[collection]:
            if self._matches_query(doc, filter_query):
                # Simple update (assumes $set operation)
                if '$set' in update_data:
                    doc.update(update_data['$set'])
                else:
                    doc.update(update_data)
                updated = True
                if not kwargs.get('many', False):
                    break
        
        return updated
    
    async def delete(self, collection: str, query: Dict[str, Any]) -> int:
        """Delete documents from the in-memory collection."""
        if collection not in self.collections:
            return 0
        
        original_count = len(self.collections[collection])
        self.collections[collection] = [
            doc for doc in self.collections[collection]
            if not self._matches_query(doc, query)
        ]
        
        return original_count - len(self.collections[collection])
    
    async def count(self, collection: str, query: Dict[str, Any] = None) -> int:
        """Count documents in the in-memory collection."""
        if collection not in self.collections:
            return 0
        
        if query is None:
            return len(self.collections[collection])
        
        return len([doc for doc in self.collections[collection] if self._matches_query(doc, query)])
    
    async def find_one(self, collection: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find a single document in the in-memory collection."""
        if collection not in self.collections:
            return None
        
        for doc in self.collections[collection]:
            if self._matches_query(doc, query):
                return doc.copy()
        
        return None
    
    async def find_one_and_update(self, collection: str, filter_query: Dict[str, Any], update_data: Dict[str, Any], **kwargs) -> Optional[Dict[str, Any]]:
        """Find and update a single document in the in-memory collection."""
        if collection not in self.collections:
            return None
        
        for doc in self.collections[collection]:
            if self._matches_query(doc, filter_query):
                # Apply the update
                if '$set' in update_data:
                    doc.update(update_data['$set'])
                else:
                    doc.update(update_data)
                return doc.copy()
        
        return None

    def _matches_query(self, doc: Dict[str, Any], query: Dict[str, Any]) -> bool:
        """Simple query matching for in-memory storage."""
        for key, value in query.items():
            if key not in doc:
                return False
            
            # Handle special MongoDB operators
            if isinstance(value, dict):
                if '$ne' in value:
                    # Handle $ne (not equal) operator
                    if key == "_id":
                        if str(doc[key]) == str(value['$ne']):
                            return False
                    elif doc[key] == value['$ne']:
                        return False
                    continue
                # Add more operators as needed
            
            # Handle ObjectId queries
            if isinstance(value, ObjectId):
                # Convert ObjectId to string for comparison
                if str(value) != str(doc[key]):
                    return False
            elif key == "_id":
                # Handle _id queries - always compare as strings
                if str(value) != str(doc[key]):
                    return False
            elif doc[key] != value:
                return False
        return True

# PostgreSQL Database Implementation (placeholder for future)
class PostgreSQLDatabase(Database):
    """PostgreSQL implementation placeholder."""
    
    def __init__(self, url: str):
        self.url = url
        self.connection = None
        raise NotImplementedError("PostgreSQL implementation not yet available")
    
    async def connect(self) -> None:
        raise NotImplementedError("PostgreSQL implementation not yet available")
    
    async def disconnect(self) -> None:
        raise NotImplementedError("PostgreSQL implementation not yet available")
    
    async def ping(self) -> bool:
        raise NotImplementedError("PostgreSQL implementation not yet available")
    
    async def search(self, collection: str, query: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:
        raise NotImplementedError("PostgreSQL implementation not yet available")
    
    async def write(self, collection: str, data: Dict[str, Any]) -> str:
        raise NotImplementedError("PostgreSQL implementation not yet available")
    
    async def update(self, collection: str, filter_query: Dict[str, Any], update_data: Dict[str, Any], **kwargs) -> bool:
        raise NotImplementedError("PostgreSQL implementation not yet available")
    
    async def delete(self, collection: str, query: Dict[str, Any]) -> int:
        raise NotImplementedError("PostgreSQL implementation not yet available")
    
    async def count(self, collection: str, query: Dict[str, Any] = None) -> int:
        raise NotImplementedError("PostgreSQL implementation not yet available")
    
    async def find_one(self, collection: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raise NotImplementedError("PostgreSQL implementation not yet available")

# Database Factory
def create_database(db_type: str = "", **kwargs) -> Database:
    """Factory function to create database instances."""
    if db_type.lower() == "mongodb":
        url = kwargs.get('url', settings.MONGODB_URL)
        db_name = kwargs.get('db_name', settings.MONGODB_DB_NAME)
        return MongoDatabase(url, db_name)
    elif db_type.lower() == "postgresql":
        url = kwargs.get('url', '')
        return PostgreSQLDatabase(url)
    else:
        return NoDatabase()

# Global database instance
database: Database = create_database("mongodb")