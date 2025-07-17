import json
import hashlib
import asyncio
import datetime
from uuid import uuid4
from typing import Union, List, Dict, Optional, Any, Tuple

from pydantic import Field

from .memory import BaseMemory
from evoagentx.rag import RAGConfig, RAGEngine
from evoagentx.rag.schema import Corpus, Chunk, ChunkMetadata, Query, RagResult
from evoagentx.storages.base import StorageHandler
from evoagentx.storages.schema import MemoryStore
from evoagentx.core.message import Message, MessageType
from evoagentx.core.logging import logger
from evoagentx.models.base_model import BaseLLM
from evoagentx.prompts.memory.manager import MANAGER_PROMPT


class LongTermMemory(BaseMemory):
    """
    Manages long-term storage and retrieval of memories, integrating with RAGEngine for indexing
    and StorageHandler for persistence. Uses an optional LLM to manage memory operations.
    """
    storage_handler: StorageHandler = Field(..., description="Handler for persistent storage")
    rag_config: RAGConfig = Field(..., description="Configuration for RAG engine")
    rag_engine: RAGEngine = Field(default=None, description="RAG engine for indexing and retrieval")
    llm: Optional[BaseLLM] = Field(default=None, description="LLM for deciding memory operations")
    memory_table: str = Field(default="memory", description="Database table for storing memories")
    default_corpus_id: str = Field(default_factory=lambda: str(uuid4()), description="Default corpus ID for memory indexing")
    use_llm_management: bool = Field(default=True, description="Toggle LLM-based memory management")

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
            wf_goal=message.wf_goal,
            agent=message.agent,
            msg_type=message.msg_type.value if message.msg_type else None,
            prompt=message.prompt,
            next_actions=message.next_actions,
            wf_task=message.wf_task,
            wf_task_desc=message.wf_task_desc,
            message_id=message.message_id,
            content=json.dumps(message.content) # keep the whole message content.
        )
        return Chunk(
            chunk_id=memory_id,
            text=str(message.content),  # stroe message content from __str__.
            metadata=metadata,
            start_char_idx=0,
            end_char_idx=len(message.content),
        )

    def _chunk_to_message(self, chunk: Chunk) -> Message:
        """Convert a Chunk to a Message object."""
        return Message(
            content=chunk.metadata.content,
            action=chunk.metadata.action,
            wf_goal=chunk.metadata.wf_goal,
            timestamp=chunk.metadata.timestamp,
            agent=chunk.metadata.agent,
            msg_type=chunk.metadata.msg_type,
            prompt=chunk.metadata.prompt,
            next_actions=chunk.metadata.next_actions,
            wf_task=chunk.metadata.wf_task,
            wf_task_desc=chunk.metadata.wf_task_desc,
            message_id=chunk.metadata.message_id
        )
    
    def _prompt_llm_for_memory_operation(self, input_data: Dict[str, Any], relevant_data: List[Chunk] = None) -> Dict[str, Any]:
        """Prompt the LLM to decide memory operation (add, update, delete) and return structured JSON."""
        if not self.llm or not self.use_llm_management:
            return input_data  # Bypass LLM if disabled or no LLM provided

        relevant_data_str = '\n'.join([chunk.to_json() for chunk in relevant_data or []])
        prompt = MANAGER_PROMPT.replace("<<INPUT_DATA>>", json.dumps(input_data, ensure_ascii=False)).replace("<<RELEVANT_DATA>>", relevant_data_str)

        logger.info(f"The Memory Manager Prompts: \n\n {prompt}")
        try:
            response = self.llm.generate(prompt=prompt)
            parsed = response.content.replace("```json", "").replace("```", "").strip()
            result = json.loads(parsed)
            if result["action"] not in ["add", "update", "delete"]:
                raise ValueError(f"Invalid action: {result['action']}")
            if result["action"] in ["update", "delete"] and not result.get("memory_id"):
                raise ValueError(f"memory_id required for {result['action']}")
            if result["action"] in ["add", "update"] and not result.get("message"):
                raise ValueError(f"message required for {result['action']}")
            return result
        except Exception as e:
            logger.error(f"LLM failed to generate valid memory operation: {str(e)}")
            return input_data  # Fallback to input data on error

    def add(self, messages: Union[Message, str, List[Union[Message, str]]], save_to_db: bool = True) -> List[str]:
        """Store messages in memory and index them in RAGEngine, returning memory_ids."""
        if not isinstance(messages, list):
            messages = [messages]
        messages = [Message(content=msg) if isinstance(msg, str) else msg for msg in messages]
        messages = [msg for msg in messages if msg.content]  # Filter out empty messages

        if not messages:
            logger.warning("No valid messages to add")
            return []

        # Prepare input data for LLM
        memory_ids = [str(uuid4()) for _ in messages]
        input_data = [
            {
                "action": "add",
                "memory_id": memory_id,
                "message": {
                    "content": msg.content,
                    "action": msg.action,
                    "wf_goal": msg.wf_goal,
                    "timestamp": msg.timestamp or datetime.now().isoformat(),
                    "agent": msg.agent or "user",
                    "msg_type": msg.msg_type.value if msg.msg_type else "request",
                    "prompt": msg.prompt,
                    "next_actions": msg.next_actions or [],
                    "wf_task": msg.wf_task,
                    "wf_task_desc": msg.wf_task_desc,
                    "message_id": msg.message_id
                }
            }
            for msg, memory_id in zip(messages, memory_ids)
        ]

        # Hash-based deduplication when LLM is disabled
        if not self.use_llm_management:
            existing_hashes = {
                record["content_hash"]
                for record in self.storage_handler.load(tables=[self.memory_table]).get(self.memory_table, [])
                if "content_hash" in record
            }
            final_messages = []
            final_memory_ids = []
            final_input_data = []
            for msg, memory_id, data in zip(messages, memory_ids, input_data):
                content_hash = hashlib.sha256(msg.content.encode()).hexdigest()
                if content_hash in existing_hashes:
                    logger.info(f"Duplicate message found (hash): {msg.content[:50]}...")
                    existing_id = next(
                        (r["memory_id"] for r in self.storage_handler.load(tables=[self.memory_table]).get(self.memory_table, [])
                         if r.get("content_hash") == content_hash), None
                    )
                    if existing_id:
                        final_memory_ids.append(existing_id)
                        continue
                final_messages.append(msg)
                final_memory_ids.append(memory_id)
                final_input_data.append({**data, "content_hash": content_hash})
            messages = final_messages
            memory_ids = final_memory_ids
            input_data = final_input_data
        else:
            final_input_data = input_data

        # LLM decision for batch
        llm_decisions = asyncio.run(self._prompt_llm_for_memory_operation(final_input_data))
        final_messages = []
        final_memory_ids = []
        final_chunks = []

        for decision, msg, memory_id in zip(llm_decisions, messages, memory_ids):
            if not decision or decision.get("action") != "add":
                logger.info(f"LLM rejected adding memory {memory_id}: {decision}")
                continue
            final_messages.append(msg)
            final_memory_ids.append(decision.get("memory_id", memory_id))
            final_chunks.append(self._create_memory_chunk(msg, decision.get("memory_id", memory_id)))

        if not final_chunks:
            logger.info("No messages added after LLM filtering")
            return final_memory_ids

        # Add to in-memory messages
        for msg in final_messages:
            super().add_message(msg)

        # Index in RAGEngine
        corpus = Corpus(chunks=final_chunks, corpus_id=self.default_corpus_id)
        chunk_ids = self.rag_engine.add(index_type=self.rag_config.index.index_type, nodes=corpus, corpus_id=self.default_corpus_id)
        if not chunk_ids:
            logger.error("Failed to index memories")
            return final_memory_ids

        # Save to database
        if save_to_db:
            memory_data_batch = [
                {
                    "memory_id": memory_id,
                    "content": msg.content,
                    "content_hash": hashlib.sha256(msg.content.encode()).hexdigest(),
                    "action": msg.action,
                    "wf_goal": msg.wf_goal,
                    "timestamp": msg.timestamp,
                    "agent": msg.agent,
                    "msg_type": msg.msg_type.value if msg.msg_type else None,
                    "prompt": msg.prompt,
                    "next_actions": msg.next_actions,
                    "wf_task": msg.wf_task,
                    "wf_task_desc": msg.wf_task_desc,
                    "message_id": msg.message_id,
                    "corpus_id": self.default_corpus_id
                }
                for msg, memory_id in zip(final_messages, final_memory_ids)
            ]
            for memory_data in memory_data_batch:
                self.storage_handler.save_memory(memory_data, table=self.memory_table)
            logger.info(f"Added {len(final_memory_ids)} memories to database and RAG index")

        return final_memory_ids

    def delete(self, memory_ids: Union[str, List[str]], delete_from_db: bool = True) -> List[bool]:
        """Delete memories by memory_ids, returning success status for each."""
        if not isinstance(memory_ids, list):
            memory_ids = [memory_ids]

        if not memory_ids:
            logger.warning("No memory_ids provided for deletion")
            return []

        # Search for existing memories
        input_data = [{"action": "delete", "memory_id": mid} for mid in memory_ids]
        existing_chunks = []
        for mid in memory_ids:
            chunks = self.search("", metadata_filters={"memory_id": mid}, n=1)
            existing_chunks.append(chunks[0][0] if chunks else None)

        # LLM decision for batch
        llm_decisions = self._prompt_llm_for_memory_operation(input_data, relevant_data=[c for c in existing_chunks if c])
        successes = []
        valid_memory_ids = []
        valid_messages = []

        for decision, chunk, mid in zip(llm_decisions, existing_chunks, memory_ids):
            if not chunk:
                logger.warning(f"No memory found with memory_id {mid}")
                successes.append(False)
                continue
            if decision["action"] != "delete":
                logger.info(f"LLM rejected deleting memory {mid}: {decision}")
                successes.append(False)
                continue
            valid_memory_ids.append(mid)
            valid_messages.append(self._chunk_to_message(chunk))
            successes.append(True)

        if not valid_memory_ids:
            logger.info("No memories deleted after LLM filtering")
            return successes

        # Remove from in-memory messages
        for msg in valid_messages:
            super().remove_message(msg)

        # Remove from RAG index
        self.rag_engine.delete(
            corpus_id=self.default_corpus_id,
            index_type=self.rag_config.index.index_type,
            node_ids=valid_memory_ids,
            metadata_filters={"memory_id": valid_memory_ids}
        )

        # Remove from database
        if delete_from_db:
            for mid in valid_memory_ids:
                self.storage_handler.storageDB.delete(mid, store_type="memory", table=self.memory_table)
            logger.info(f"Deleted {len(valid_memory_ids)} memories from database and RAG index")

        return successes

    def update(self, updates: Union[Tuple[str, Union[Message, str]], List[Tuple[str, Union[Message, str]]]], 
               save_to_db: bool = True) -> List[bool]:
        """Update memories with new content, returning success status for each."""
        if not isinstance(updates, list):
            updates = [updates]
        updates = [(mid, Message(content=msg) if isinstance(msg, str) else msg) for mid, msg in updates]
        updates = [(mid, msg) for mid, msg in updates if msg.content]

        if not updates:
            logger.warning("No valid updates provided")
            return []

        # Search for existing memories
        input_data = []
        existing_chunks = []
        for mid, msg in updates:
            chunks = self.search("", metadata_filters={"memory_id": mid}, n=1)
            existing_chunks.append(chunks[0][0] if chunks else None)
            input_data.append({
                "action": "update",
                "memory_id": mid,
                "message": {
                    "content": msg.content,
                    "action": msg.action,
                    "wf_goal": msg.wf_goal,
                    "timestamp": msg.timestamp,
                    "agent": msg.agent,
                    "msg_type": msg.msg_type.value if msg.msg_type else None,
                    "prompt": msg.prompt,
                    "next_actions": msg.next_actions,
                    "wf_task": msg.wf_task,
                    "wf_task_desc": msg.wf_task_desc,
                    "message_id": msg.message_id
                }
            })

        # LLM decision for batch
        llm_decisions = self._prompt_llm_for_memory_operation(input_data, relevant_data=[c for c in existing_chunks if c])
        successes = []
        valid_updates = []
        valid_memory_ids = []

        for decision, chunk, (mid, msg) in zip(llm_decisions, existing_chunks, updates):
            if not chunk:
                logger.warning(f"No memory found with memory_id {mid}")
                successes.append(False)
                continue
            if decision["action"] != "update":
                logger.info(f"LLM rejected updating memory {mid}: {decision}")
                successes.append(False)
                continue
            valid_updates.append((mid, msg))
            valid_memory_ids.append(mid)
            successes.append(True)

        if not valid_updates:
            logger.info("No memories updated after LLM filtering")
            return successes

        # Remove old messages
        for mid in valid_memory_ids:
            chunks = self.search("", metadata_filters={"memory_id": mid}, n=1)
            if chunks:
                super().remove_message(self._chunk_to_message(chunks[0][0]))

        # Add updated messages
        chunks = [self._create_memory_chunk(msg, mid) for mid, msg in valid_updates]
        for msg in [msg for _, msg in valid_updates]:
            super().add_message(msg)

        # Update RAG index
        self.rag_engine.delete(
            corpus_id=self.default_corpus_id,
            index_type=self.rag_config.index.index_type,
            node_ids=valid_memory_ids,
            metadata_filters={"memory_id": valid_memory_ids}
        )
        corpus = Corpus(chunks=chunks, corpus_id=self.default_corpus_id)
        chunk_ids = self.rag_engine.add(index_type=self.rag_config.index.index_type, nodes=corpus, corpus_id=self.default_corpus_id)
        if not chunk_ids:
            logger.error(f"Failed to update memories in RAG index: {valid_memory_ids}")
            return [False] * len(updates)

        # Update database
        if save_to_db:
            memory_data_batch = [
                {
                    "memory_id": mid,
                    "content": msg.content,
                    "action": msg.action,
                    "wf_goal": msg.wf_goal,
                    "timestamp": msg.timestamp,
                    "agent": msg.agent,
                    "msg_type": msg.msg_type.value if msg.msg_type else None,
                    "prompt": msg.prompt,
                    "next_actions": msg.next_actions,
                    "wf_task": msg.wf_task,
                    "wf_task_desc": msg.wf_task_desc,
                    "message_id": msg.message_id,
                    "corpus_id": self.default_corpus_id
                }
                for mid, msg in valid_updates
            ]
            for memory_data in memory_data_batch:
                self.storage_handler.save_memory(memory_data, table=self.memory_table)
            logger.info(f"Updated {len(valid_memory_ids)} memories in database and RAG index")

        return successes

    async def search_async(self, query: Union[str, Query], n: Optional[int] = None, 
                          metadata_filters: Optional[Dict[str, Any]] = None) -> List[Tuple[Message, str]]:
        """Retrieve messages from RAG index asynchronously based on a query, returning messages and memory_ids."""
        if isinstance(query, str):
            query_obj = Query(
                query_str=query,
                top_k=n or self.rag_config.retrieval.top_k,
                metadata_filters=metadata_filters or {}
            )
        else:
            query_obj = query
            query_obj.top_k = n or self.rag_config.retrieval.top_k
            if metadata_filters:
                query_obj.metadata_filters = {**query_obj.metadata_filters, **metadata_filters} if query_obj.metadata_filters else metadata_filters

        try:
            result: RagResult = await self.rag_engine.query_async(query_obj, corpus_id=self.default_corpus_id)
            messages = [(self._chunk_to_message(chunk), chunk.metadata.memory_id) for chunk in result.corpus.chunks]
            logger.info(f"Retrieved {len(messages)} memories for query: {query_obj.query_str}")
            return messages[:n] if n else messages
        except Exception as e:
            logger.error(f"Failed to search memories: {str(e)}")
            return []

    def search(self, query: Union[str, Query], n: Optional[int] = None, 
               metadata_filters: Optional[Dict[str, Any]] = None) -> List[Tuple[Message, str]]:
        """Synchronous wrapper for searching memories."""
        return asyncio.run(self.search_async(query, n, metadata_filters))

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
                "corpus_id": self.default_corpus_id,
                "user_id": message.agent,
                "tags": [message.msg_type.value] if message.msg_type else [],
                "memory_type": "message"
            }
            self.storage_handler.save_memory(memory_data, table=self.memory_table)
        logger.info(f"Saved LongTermMemory data to database table {self.memory_table}")

    def load(self) -> List[str]:
        """Load memory data from database and reconstruct indices, returning memory_ids."""
        memory_data = self.storage_handler.load(tables=[self.memory_table]).get(self.memory_table, [])
        self.clear(delete_from_db=False)
        memory_ids = []
        for record in memory_data:
            parsed = self.storage_handler.parse_result(record, MemoryStore)
            memory_id = parsed["memory_id"]
            message = Message(
                content=parsed["content"],
                action=parsed.get("action"),
                wf_goal=parsed.get("wf_goal"),
                timestamp=parsed["timestamp"],
                agent=parsed.get("user_id"),
                msg_type=parsed.get("tag", MessageType.UNKNOWN.value)
            )
            chunk = self._create_memory_chunk(message, memory_id)
            corpus = Corpus(chunks=[chunk], corpus_id=self.default_corpus_id)
            chunk_ids = self.rag_engine.add(index_type=self.rag_config.index.index_type, nodes=corpus, corpus_id=self.default_corpus_id)
            if chunk_ids:
                super().add_message(message)
                memory_ids.append(memory_id)
        chunk_ids = self.rag_engine.load(corpus_id=self.default_corpus_id, table=self.memory_table)
        memory_ids.extend(chunk_ids)
        logger.info(f"Loaded {len(memory_ids)} memories from database table {self.memory_table}")
        return memory_ids