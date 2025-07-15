import json
import asyncio
from uuid import uuid4
from typing import Union, List, Dict, Optional, Any

from pydantic import Field

from .memory import BaseMemory
from evoagentx.rag import RAGConfig, RAGEngine
from evoagentx.rag.schema import Corpus, Chunk, ChunkMetadata, Query, RagResult
from evoagentx.storages.base import StorageHandler
from evoagentx.storages.schema import MemoryStore
from evoagentx.core.message import Message
from evoagentx.core.logging import logger
from evoagentx.models.base_model import BaseLLM
from evoagentx.rag.indexings.base import IndexType
from evoagentx.prompts.memory.manager import MANAGER_PROMPT


class LongTermMemory(BaseMemory):
    """
    Manages long-term storage and retrieval of memories, integrating with RAGEngine for indexing
    and StorageHandler for persistence. Uses an LLM to decide memory operations (add, update, delete).
    """
    storage_handler: StorageHandler = Field(..., description="Handler for persistent storage")
    rag_config: RAGConfig = Field(..., description="Configuration for RAG engine")
    rag_engine: RAGEngine = Field(default=None, description="RAG engine for indexing and retrieval")
    llm: BaseLLM = Field(..., description="LLM for deciding memory operations")
    memory_table: str = Field(default="memory", description="Database table for storing memories")
    default_corpus_id: str = Field(default_factory=lambda: str(uuid4()), description="Default corpus ID for memory indexing")

    def init_module(self):
        """Initialize the RAG engine and memory indices."""
        if self.rag_engine is None:
            self.rag_engine = RAGEngine(config=self.rag_config, storage_handler=self.storage_handler, llm=self.llm)
        logger.info(f"Initialized LongTermMemory with corpus_id {self.default_corpus_id}")

    def _create_memory_chunk(self, message: Message, memory_id: str) -> Chunk:
        """Convert a Message to a Chunk for RAG indexing."""
        metadata = ChunkMetadata(
            corpus_id=self.default_corpus_id,
            memory_id=memory_id,
            timestamp=message.timestamp,
            action=message.action,
            wf_goal=message.wf_goal
        )
        return Chunk(
            chunk_id=memory_id,
            text=message.content,
            metadata=metadata,
            start_char_idx=0,
            end_char_idx=len(message.content),
        )

    def _chunk_to_message(self, chunk: Chunk) -> Message:
        """Convert a Chunk to a Message object."""
        return Message(
            content=chunk.text,
            action=chunk.metadata.action,
            wf_goal=chunk.metadata.wf_goal,
            timestamp=chunk.metadata.timestamp
        )

    def _prompt_llm_for_memory_operation(self, input_data: Dict[str, Any], relevant_data: List[Chunk]) -> Dict[str, Any]:
        """Prompt the LLM to decide memory operation (add, update, delete) and return structured JSON."""

        prompt = MANAGER_PROMPT.format_map({"<<INPUT_DATA>>": json.dumps(input_data, ensure_ascii=False), 
                                            "<<RELEVANT_DATA>>": '\n'.join(relevant_data)})
        try:
            response = self.llm.generate(prompt=prompt)
            result = json.loads(response)
            if result["action"] not in ["add", "update", "delete"]:
                raise ValueError(f"Invalid action: {result['action']}")
            if result["action"] in ["update", "delete"] and not result.get("memory_id"):
                raise ValueError(f"memory_id required for {result['action']}")
            if result["action"] in ["add", "update"] and not result.get("message"):
                raise ValueError(f"message required for {result['action']}")
            return result
        except Exception as e:
            logger.error(f"LLM failed to generate valid memory operation: {str(e)}")
            raise

    def add_message(self, message: Message, save_to_db: bool = True):
        """Store a single message in memory and index it in RAGEngine."""
        if not message:
            return
        if message in self.messages:
            logger.info(f"Message already exists in memory: {message.content[:50]}...")
            return

        memory_id = str(uuid4())
        # Use LLM to confirm adding the memory
        input_data = {
            "operation": "add",
            "message": {
                "content": message.content,
                "action": message.action,
                "wf_goal": message.wf_goal,
                "timestamp": message.timestamp
            }
        }
        llm_decision = self._prompt_llm_for_memory_operation(input_data)
        if llm_decision["action"] != "add":
            logger.info(f"LLM rejected adding memory: {llm_decision}")
            return

        # Add to in-memory messages
        super().add_message(message)

        # Convert to Chunk and index in RAGEngine
        chunk = self._create_memory_chunk(message, memory_id)
        corpus = Corpus(chunks=[chunk], corpus_id=self.default_corpus_id)
        self.rag_engine.add(index_type=IndexType.VECTOR, nodes=corpus, corpus_id=self.default_corpus_id)

        # Save to database
        if save_to_db:
            memory_data = {
                "memory_id": memory_id,
                "content": message.content,
                "action": message.action,
                "wf_goal": message.wf_goal,
                "timestamp": message.timestamp,
                "corpus_id": self.default_corpus_id
            }
            self.storage_handler.save_memory(memory_data, table=self.memory_table)
            logger.info(f"Added memory {memory_id} to database and RAG index")

    def add_messages(self, messages: Union[Message, List[Message]], save_to_db: bool = True):
        """Store multiple messages in memory and index them."""
        if not isinstance(messages, list):
            messages = [messages]
        for message in messages:
            self.add_message(message, save_to_db)

    def remove_message(self, message: Message, delete_from_db: bool = True):
        """Remove a message from memory and RAG index."""
        if not message or message not in self.messages:
            logger.warning(f"Message not found in memory: {message.content[:50] if message else 'None'}...")
            return

        # Find memory_id from database
        memory_data = self.storage_handler.load_memory(memory_id=None, table=self.memory_table)
        memory_id = None
        for record in memory_data.get(self.memory_table, []):
            parsed = self.storage_handler.parse_result(record, MemoryStore)
            if parsed["content"] == message.content and parsed["timestamp"] == message.timestamp:
                memory_id = parsed["memory_id"]
                break

        if not memory_id:
            logger.warning(f"No memory_id found for message: {message.content[:50]}...")
            return

        # Use LLM to confirm deletion
        input_data = {"operation": "delete", "memory_id": memory_id}
        llm_decision = self._prompt_llm_for_memory_operation(input_data)
        if llm_decision["action"] != "delete":
            logger.info(f"LLM rejected deleting memory: {llm_decision}")
            return

        # Remove from in-memory messages
        super().remove_message(message)

        # Remove from RAG index
        self.rag_engine.delete(
            corpus_id=self.default_corpus_id,
            index_type=IndexType.VECTOR,
            node_ids=[memory_id],
            metadata_filters={"memory_id": memory_id}
        )

        # Remove from database
        if delete_from_db:
            self.storage_handler.storageDB.delete(memory_id, store_type="memory", table=self.memory_table)
            logger.info(f"Deleted memory {memory_id} from database and RAG index")

    def update_message(self, memory_id: str, new_message: Message, save_to_db: bool = True):
        """Update an existing memory with new content."""
        # Find existing message
        existing_message = None
        for msg in self.messages:
            if msg.content == new_message.content and msg.timestamp == new_message.timestamp:
                existing_message = msg
                break

        # Use LLM to confirm update
        input_data = {
            "operation": "update",
            "memory_id": memory_id,
            "message": {
                "content": new_message.content,
                "action": new_message.action,
                "wf_goal": new_message.wf_goal,
                "timestamp": new_message.timestamp
            }
        }
        llm_decision = self._prompt_llm_for_memory_operation(input_data)
        if llm_decision["action"] != "update":
            logger.info(f"LLM rejected updating memory: {llm_decision}")
            return

        # Remove old message if exists
        if existing_message:
            self.remove_message(existing_message, delete_from_db=False)

        # Add updated message
        self.add_message(new_message, save_to_db=False)

        # Update RAG index
        self.rag_engine.delete(
            corpus_id=self.default_corpus_id,
            index_type=IndexType.VECTOR,
            node_ids=[memory_id],
            metadata_filters={"memory_id": memory_id}
        )
        chunk = self._create_memory_chunk(new_message, memory_id)
        corpus = Corpus(chunks=[chunk], corpus_id=self.default_corpus_id)
        self.rag_engine.add(index_type=IndexType.VECTOR, nodes=corpus, corpus_id=self.default_corpus_id)

        # Update database
        if save_to_db:
            memory_data = {
                "memory_id": memory_id,
                "content": new_message.content,
                "action": new_message.action,
                "wf_goal": new_message.wf_goal,
                "timestamp": new_message.timestamp,
                "corpus_id": self.default_corpus_id
            }
            self.storage_handler.save_memory(memory_data, table=self.memory_table)
            logger.info(f"Updated memory {memory_id} in database and RAG index")

    async def get_async(self, query: Union[str, Query], n: Optional[int] = None, **kwargs) -> List[Message]:
        """Retrieve messages from RAG index asynchronously based on a query."""
        if isinstance(query, str):
            query = Query(query_str=query, top_k=n or self.rag_config.retrieval.top_k)

        try:
            result: RagResult = await self.rag_engine.query_async(query, corpus_id=self.default_corpus_id)
            messages = [self._chunk_to_message(chunk) for chunk in result.corpus.chunks]
            logger.info(f"Retrieved {len(messages)} messages for query: {query.query_str}")
            return messages[:n] if n else messages
        except Exception as e:
            logger.error(f"Failed to retrieve messages: {str(e)}")
            return []

    def get(self, query: Union[str, Query], n: Optional[int] = None, **kwargs) -> List[Message]:
        """Synchronous wrapper for retrieving messages."""
        return asyncio.run(self.get_async(query, n, **kwargs))

    def get_by_action(self, actions: Union[str, List[str]], n: Optional[int] = None, **kwargs) -> List[Message]:
        """Retrieve messages by action using in-memory indices."""
        messages = super().get_by_action(actions, n, **kwargs)
        if not messages and kwargs.get("use_rag", False):
            query = Query(
                query_str="",
                metadata_filters={"action": actions} if isinstance(actions, str) else {"action": actions[0]},
                top_k=n or self.rag_config.retrieval.top_k
            )
            messages = self.get(query, n)
        return messages

    def get_by_wf_goal(self, wf_goals: Union[str, List[str]], n: Optional[int] = None, **kwargs) -> List[Message]:
        """Retrieve messages by workflow goal using in-memory indices."""
        messages = super().get_by_wf_goal(wf_goals, n, **kwargs)
        if not messages and kwargs.get("use_rag", False):
            query = Query(
                query_str="",
                metadata_filters={"wf_goal": wf_goals} if isinstance(wf_goals, str) else {"wf_goal": wf_goals[0]},
                top_k=n or self.rag_config.retrieval.top_k
            )
            messages = self.get(query, n)
        return messages

    def clear(self, delete_from_db: bool = True):
        """Clear all messages and indices."""
        super().clear()
        self.rag_engine.clear(corpus_id=self.default_corpus_id)
        if delete_from_db:
            self.storage_handler.storageDB.clear_table(self.memory_table)
        logger.info(f"Cleared LongTermMemory with corpus_id {self.default_corpus_id}")

    def save(self):
        """Save all indices and memory data to database."""
        self.rag_engine.save(corpus_id=self.default_corpus_id, table=self.memory_table)
        for message in self.messages:
            memory_id = str(uuid4())
            memory_data = {
                "memory_id": memory_id,
                "content": message.content,
                "action": message.action,
                "wf_goal": message.wf_goal,
                "timestamp": message.timestamp,
                "corpus_id": self.default_corpus_id
            }
            self.storage_handler.save_memory(memory_data, table=self.memory_table)
        logger.info(f"Saved LongTermMemory data to database table {self.memory_table}")

    def load(self):
        """Load memory data from database and reconstruct indices."""
        memory_data = self.storage_handler.load(tables=[self.memory_table]).get(self.memory_table, [])
        self.clear(delete_from_db=False)
        for record in memory_data:
            parsed = self.storage_handler.parse_result(record, MemoryStore)
            message = Message(
                content=parsed["content"],
                action=parsed.get("action"),
                wf_goal=parsed.get("wf_goal"),
                timestamp=parsed["timestamp"]
            )
            self.add_message(message, save_to_db=False)
        self.rag_engine.load(corpus_id=self.default_corpus_id, table=self.memory_table)
        logger.info(f"Loaded LongTermMemory data from database table {self.memory_table}")