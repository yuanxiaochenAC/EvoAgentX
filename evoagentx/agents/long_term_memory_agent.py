import json
import asyncio
from uuid import uuid4
from datetime import datetime
from typing import Optional, List, Dict, Union, Type, Tuple, Callable

from pydantic import create_model, Field

from evoagentx.models import BaseLLM
from evoagentx.agents import CustomizeAgent
from evoagentx.core.logging import logger
from evoagentx.core.message import Message, MessageType
from evoagentx.models.model_configs import LLMConfig
from evoagentx.prompts.template import PromptTemplate
from evoagentx.actions.action import Action, ActionInput, ActionOutput
from evoagentx.memory.memory_manager import MemoryManager
from evoagentx.rag.schema import Query
from evoagentx.utils.utils import generate_dynamic_class_name, make_parent_folder
from evoagentx.actions.customize_action import CustomizeAction


class ConversationActionInput(ActionInput):
    message: str = Field(..., description="User's input message for the conversation")
    top_k: Optional[int] = Field(default=5, description="Number of memory results to retrieve")
    metadata_filters: Optional[Dict] = Field(default=None, description="Metadata filters for memory search")

class ConversationActionOutput(ActionOutput):
    response: str = Field(..., description="Agent's response to the user")
    memory_ids: List[str] = Field(default_factory=list, description="IDs of memories used as context")
    conversation_id: Optional[str] = Field(default=None, description="ID of the conversation thread")

class LongTermMemoryAgent(CustomizeAgent):
    """
    An agent for conversational tasks, using MemoryManager to manage LongTermMemory for storing
    and retrieving conversation history and relevant context for coherent dialogue.

    Attributes:
        memory_manager (MemoryManager): Instance for managing memory operations.
        conv_prompt_template (PromptTemplate, optional): Template for conversation action prompt.
        conv_system_prompt (str, optional): System prompt for conversational tasks.
        conversation_id (str): Unique ID for the current conversation thread.
    """
    memory_manager: MemoryManager = Field(..., description="Memory manager instance")
    conv_prompt_template: Optional[PromptTemplate] = Field(default=None, description="Prompt template for conversation action")
    conv_system_prompt: Optional[str] = Field(default=None, description="System prompt for conversation tasks")
    conversation_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique ID for the conversation thread")

    def __init__(
        self,
        name: str,
        description: str,
        memory_manager: MemoryManager,
        llm_config: Optional[LLMConfig] = None,
        conv_prompt: Optional[str] = None,
        conv_prompt_template: Optional[PromptTemplate] = None,
        conv_system_prompt: Optional[str] = None,
        parse_mode: str = "json",
        parse_func: Optional[Callable] = None,
        title_format: Optional[str] = None,
        custom_output_format: Optional[str] = None,
        **kwargs
    ):
        if conv_prompt is None and conv_prompt_template is None:
            conv_prompt = (
                "You are engaged in a conversation with a user. Use the provided conversation history and relevant memories to respond.\n"
                "{combined_content}\n"
                "Provide a coherent response in JSON format:\n"
                "```json\n"
                "{\"response\": \"your_response\", \"memory_ids\": [], \"conversation_id\": \"{conversation_id}\"}\n"
                "```"
            )

        inputs = [
            {"name": "message", "type": "str", "description": "User's input message for the conversation", "required": True},
            {"name": "top_k", "type": "int", "description": "Number of memory results to retrieve", "required": False},
            {"name": "metadata_filters", "type": "Dict", "description": "Metadata filters for memory search", "required": False}
        ]
        outputs = [
            {"name": "response", "type": "str", "description": "Agent's response to the user", "required": True},
            {"name": "memory_ids", "type": "List[str]", "description": "IDs of memories used as context", "required": False},
            {"name": "conversation_id", "type": "str", "description": "ID of the conversation thread", "required": False}
        ]

        self.memory_manager = memory_manager
        self.conv_prompt_template = conv_prompt_template
        self.conv_system_prompt = conv_system_prompt or (
            "You are a conversational AI assistant with access to a memory management system. Use the provided history and memories to provide coherent, context-aware responses."
        )

        super().__init__(
            name=name,
            description=description,
            prompt=conv_prompt,
            prompt_template=conv_prompt_template,
            llm_config=llm_config,
            inputs=inputs,
            outputs=outputs,
            system_prompt=self.conv_system_prompt,
            parse_mode=parse_mode,
            parse_func=parse_func,
            title_format=title_format,
            custom_output_format=custom_output_format,
            **kwargs
        )

    def create_customize_action(
        self,
        name: str,
        desc: str,
        prompt: str,
        prompt_template: PromptTemplate,
        inputs: List[dict],
        outputs: List[dict],
        parse_mode: str,
        parse_func: Optional[Callable] = None,
        output_parser: Optional[Type[ActionOutput]] = None,
        title_format: Optional[str] = None,
        custom_output_format: Optional[str] = None,
        tools: Optional[List] = None,
        max_tool_calls: Optional[int] = 5
    ) -> Action:
        """Override to create a conversation-specific action using MemoryManager."""
        action_input_fields = {
            field["name"]: (eval(field["type"]), Field(description=field["description"], default=None if not field.get("required", True) else ...))
            for field in inputs
        }
        action_input_type = create_model(
            self._get_unique_class_name(generate_dynamic_class_name(name + "_conv_input")),
            **action_input_fields,
            __base__=ConversationActionInput
        )

        action_output_fields = {
            field["name"]: (eval(field["type"]), Field(description=field["description"], default_factory=list if field["type"] == "List[str]" else None))
            for field in outputs
        }
        action_output_type = create_model(
            self._get_unique_class_name(generate_dynamic_class_name(name + "_conv_output")),
            **action_output_fields,
            __base__=ConversationActionOutput
        )

        class ConversationAction(CustomizeAction):
            async def execute_async(
                self,
                llm: Optional["BaseLLM"] = None,
                inputs: Optional[Dict] = None,
                sys_msg: Optional[str] = None,
                return_prompt: bool = False,
                memory_manager: Optional[MemoryManager] = None,
                conversation_id: Optional[str] = None,
                **kwargs
            ) -> Union[ActionOutput, tuple[ActionOutput, str]]:
                if memory_manager is None or llm is None or conversation_id is None:
                    raise ValueError("MemoryManager, LLM, and conversation_id required")
                inputs = inputs or {}
                message = inputs.get("message")
                top_k = inputs.get("top_k")
                metadata_filters = inputs.get("metadata_filters", {})

                # Create combined message with history and context
                combined_message = await memory_manager.create_conversation_message(
                    user_prompt=message, conversation_id=conversation_id, top_k=top_k, metadata_filters=metadata_filters
                )

                # Format prompt
                prompt_str = self.prompt.format(combined_content=combined_message.content, conversation_id=conversation_id)
                if self.prompt_template:
                    prompt_str = self.prompt_template.format(combined_content=combined_message.content, conversation_id=conversation_id)

                # Generate response
                try:
                    response = llm.generate(prompt=prompt_str, system_prompt=sys_msg or self.system_prompt)
                    parsed = json.loads(response.content.replace("```json", "").replace("```", "").strip())
                    output = action_output_type(
                        response=parsed["response"],
                        memory_ids=parsed.get("memory_ids", combined_message.memory_ids or []),
                        conversation_id=conversation_id
                    )
                except Exception as e:
                    logger.error(f"Failed to generate or parse response: {str(e)}")
                    output = action_output_type(
                        response="Sorry, I encountered an error while responding.",
                        memory_ids=combined_message.memory_ids or [],
                        conversation_id=conversation_id
                    )

                # Store user message and response in memory
                user_message = Message(
                    content=message,
                    msg_type=MessageType.REQUEST,
                    timestamp=datetime.now().isoformat(),
                    agent="user",
                    conversation_id=conversation_id,
                    message_id=str(uuid4())
                )
                response_message = Message(
                    content=output.response,
                    msg_type=MessageType.RESPONSE,
                    timestamp=datetime.now().isoformat(),
                    agent="assistant",
                    conversation_id=conversation_id,
                    message_id=str(uuid4())
                )
                await memory_manager.handle_memory(action="add", data=[user_message, response_message], conversation_id=conversation_id)

                if return_prompt:
                    return output, prompt_str
                return output

        action_cls_name = self._get_unique_class_name(generate_dynamic_class_name(name + "_conv_action"))
        conv_action_cls = create_model(action_cls_name, __base__=ConversationAction)
        return conv_action_cls(
            name=action_cls_name,
            description=desc,
            prompt=prompt,
            prompt_template=prompt_template,
            inputs_format=action_input_type,
            outputs_format=action_output_type,
            parse_mode=parse_mode,
            parse_func=parse_func,
            title_format=title_format,
            custom_output_format=custom_output_format,
            max_tool_try=max_tool_calls,
            tools=tools
        )

    async def add_memories(
        self,
        messages: Union[Message, str, List[Union[Message, str]]],
        conversation_id: Optional[str] = None
    ) -> List[str]:
        """Add memories using MemoryManager."""
        return await self.memory_manager.handle_memory(
            action="add", data=messages, conversation_id=conversation_id or self.conversation_id
        )

    async def get_memories(self, memory_ids: Union[str, List[str]]) -> List[Tuple[Message, str]]:
        """Retrieve memories by IDs using MemoryManager."""
        return await self.memory_manager.handle_memory(action="get", data=memory_ids)

    async def delete_memories(self, memory_ids: Union[str, List[str]]) -> List[bool]:
        """Delete memories by IDs using MemoryManager."""
        return await self.memory_manager.handle_memory(action="delete", data=memory_ids)

    async def update_memories(
        self,
        updates: Union[Tuple[str, Union[Message, str]], List[Tuple[str, Union[Message, str]]]],
        conversation_id: Optional[str] = None
    ) -> List[bool]:
        """Update memories using MemoryManager."""
        return await self.memory_manager.handle_memory(
            action="update", data=updates, conversation_id=conversation_id or self.conversation_id
        )

    async def search_memories(
        self,
        query: Union[str, Query],
        top_k: Optional[int] = None,
        metadata_filters: Optional[Dict] = None
    ) -> List[Tuple[Message, str]]:
        """Search memories using MemoryManager."""
        return await self.memory_manager.handle_memory(
            action="search", user_prompt=query, top_k=top_k, metadata_filters=metadata_filters
        )

    def clear_memories(self) -> None:
        """Clear all memories using MemoryManager."""
        self.memory_manager.handle_memory(action="clear")

    def save_memories(self, save_path: Optional[str] = None) -> None:
        """Save memories using MemoryManager."""
        self.memory_manager.handle_memory(action="save", data=save_path)

    def load_memories(self, save_path: Optional[str] = None) -> List[str]:
        """Load memories using MemoryManager."""
        return self.memory_manager.handle_memory(action="load", data=save_path)

    def save_module(self, path: str, ignore: List[str] = [], **kwargs) -> str:
        """Save agent configuration, including memory state."""
        config = self.get_customize_agent_info()
        config["memory_config"] = {
            "corpus_id": self.memory_manager.memory.default_corpus_id,
            "memory_table": self.memory_manager.memory.memory_table,
            "use_llm_management": self.memory_manager.use_llm_management,
            "conversation_id": self.conversation_id
        }
        make_parent_folder(path)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        self.memory_manager.handle_memory(action="save", data=path)
        return path

    @classmethod
    def load_module(cls, path: str, llm_config: LLMConfig = None, memory_manager: MemoryManager = None, **kwargs) -> "LongTermMemoryAgent":
        """Load agent from file, including memory state."""
        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        if memory_manager is None:
            raise ValueError("MemoryManager instance required for loading")
        agent = cls(
            name=config["name"],
            description=config["description"],
            memory_manager=memory_manager,
            llm_config=llm_config,
            conv_prompt=config.get("prompt"),
            conv_system_prompt=config.get("system_prompt"),
            parse_mode=config.get("parse_mode", "json"),
            title_format=config.get("title_format"),
            custom_output_format=config.get("custom_output_format"),
            **kwargs
        )
        agent.conversation_id = config.get("memory_config", {}).get("conversation_id", str(uuid4()))
        memory_manager.handle_memory(action="load", data=path)
        return agent


# Example Usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    from storages.base import StorageHandler
    from evoagentx.memory.long_term_memory import LongTermMemory
    from evoagentx.models import OpenRouterConfig, OpenRouterLLM
    from evoagentx.storages.storages_config import StoreConfig, VectorStoreConfig, DBConfig
    from evoagentx.rag.rag_config import RAGConfig, ReaderConfig, IndexConfig, ChunkerConfig, RetrievalConfig, EmbeddingConfig

    load_dotenv()

    # Initialize LLM
    OPEN_ROUNTER_API_KEY = os.environ.get("OPEN_ROUNTER_API_KEY")
    if not OPEN_ROUNTER_API_KEY:
        raise ValueError("OPEN_ROUNTER_API_KEY not set")

    llm_config = OpenRouterConfig(
        openrouter_key=OPEN_ROUNTER_API_KEY,
        temperature=0.3,
        model="google/gemini-2.5-flash-lite-preview-06-17",
    )
    llm = OpenRouterLLM(config=llm_config)

    # Initialize StorageHandler
    store_config = StoreConfig(
        dbConfig=DBConfig(
            db_name="sqlite",
            path="./debug/data/conversation/cache/test_conversation.sql"
        ),
        vectorConfig=VectorStoreConfig(
            vector_name="faiss",
            dimensions=384,
            index_type="flat_l2",
        ),
        graphConfig=None,
        path="./debug/data/conversation/cache/indexing"
    )
    storage_handler = StorageHandler(storageConfig=store_config)

    # Initialize RAGConfig
    embedding = EmbeddingConfig(
        provider="huggingface",
        model_name="debug/weights/bge-small-en-v1.5",
        device="cpu"
    )
    rag_config = RAGConfig(
        reader=ReaderConfig(
            recursive=False, exclude_hidden=True,
            num_files_limit=None, custom_metadata_function=None,
            extern_file_extractor=None, errors="ignore", encoding="utf-8"
        ),
        chunker=ChunkerConfig(
            strategy="simple",
            chunk_size=512,
            chunk_overlap=0,
            max_chunks=None
        ),
        embedding=embedding,
        index=IndexConfig(index_type="vector"),
        retrieval=RetrievalConfig(
            retrivel_type="vector",
            postprocessor_type="simple",
            top_k=5,
            similarity_cutoff=0.3,
            keyword_filters=None,
            metadata_filters=None
        )
    )

    # Initialize LongTermMemory
    memory = LongTermMemory(
        storage_handler=storage_handler,
        rag_config=rag_config
    )
    memory.init_module()

    # Initialize MemoryManager
    memory_manager = MemoryManager(memory=memory, llm=llm, use_llm_management=True)

    # Initialize LongTermMemoryAgent
    agent = LongTermMemoryAgent(
        name="ConversationalAgent",
        description="An agent for conversational tasks with long-term memory support",
        memory_manager=memory_manager,
        llm_config=llm_config,
        parse_mode="json"
    )

    # Test 1: Add initial memories (background knowledge)
    background_memories = [
        Message(content="Paris is the capital of France.", msg_type=MessageType.FACT, message_id="fact_001"),
        Message(content="The Eiffel Tower is a famous landmark in Paris.", msg_type=MessageType.FACT, message_id="fact_002"),
        Message(content="France won the FIFA World Cup in 2018.", msg_type=MessageType.FACT, message_id="fact_003")
    ]
    memory_ids = asyncio.run(agent.add_memories(background_memories))
    print("Test 1: Added background memories")
    print(f"Memory IDs: {memory_ids}")

    # Test 2: Start conversation
    conversation_id = str(uuid4())
    response = agent(
        inputs={"message": "Hi, tell me about Paris.", "top_k": 3, "metadata_filters": {"msg_type": MessageType.FACT.value}},
        return_msg_type=MessageType.RESPONSE,
        conversation_id=conversation_id
    )
    print("\nTest 2: Conversation start")
    print(f"User: Hi, tell me about Paris.")
    print(f"Assistant: {response.content['response']}")
    print(f"Memory IDs: {response.content['memory_ids']}")

    # Test 3: Continue conversation
    response = agent(
        inputs={"message": "What is the capital city?", "top_k": 3, "metadata_filters": {"msg_type": MessageType.FACT.value}},
        return_msg_type=MessageType.RESPONSE,
        conversation_id=conversation_id
    )
    print("\nTest 3: Continue conversation")
    print(f"User: What is the capital city?")
    print(f"Assistant: {response.content['response']}")
    print(f"Memory IDs: {response.content['memory_ids']}")

    # Test 4: Retrieve conversation history
    history = asyncio.run(agent.search_memories(
        query="Paris", top_k=5, metadata_filters={"conversation_id": conversation_id}
    ))
    print("\nTest 4: Retrieve conversation history")
    for msg, mid in history:
        role = "User" if msg.msg_type == MessageType.REQUEST else "Assistant"
        print(f"- {role}: {msg.content} (Memory ID: {mid})")

    # Test 5: Update a memory
    updates = [(memory_ids[0], Message(
        content="Paris is the capital of France and a cultural hub.", msg_type=MessageType.FACT, message_id="fact_001_updated"
    ))]
    successes = asyncio.run(agent.update_memories(updates))
    print("\nTest 5: Update memory")
    print(f"Successes: {successes}")

    # Test 6: Get specific memories
    retrieved = asyncio.run(agent.get_memories(memory_ids[:1]))
    print("\nTest 6: Get memories")
    for msg, mid in retrieved:
        print(f"- Memory ID: {mid}, Content: {msg.content}")

    # Test 7: Save memories
    agent.save_memories("./debug/conversation_memory.json")
    print("\nTest 7: Saved memories")

    # Test 8: Clear memories (in-memory only)
    agent.clear_memories()
    search_results = asyncio.run(agent.search_memories("Paris", top_k=1))
    print("\nTest 8: Search after clear (in-memory)")
    print(f"Results: {len(search_results)} memories found")

    # Test 9: Load memories
    loaded_ids = agent.load_memories("./debug/conversation_memory.json")
    print("\nTest 9: Loaded memories")
    print(f"Loaded {len(loaded_ids)} memory IDs: {loaded_ids}")

    # Test 10: Delete memories
    successes = asyncio.run(agent.delete_memories(memory_ids))
    print("\nTest 10: Delete memories")
    print(f"Successes: {successes}")

    # Test 11: Save and load agent
    agent.save_module("./debug/conversation_agent.json")
    loaded_agent = LongTermMemoryAgent.load_module(
        path="./debug/conversation_agent.json",
        llm_config=llm_config,
        memory_manager=memory_manager
    )
    print("\nTest 11: Saved and loaded agent")
    print(f"Agent name: {loaded_agent.name}")

    # Test 12: Verify conversation continuity after loading
    response = loaded_agent(
        inputs={"message": "Is Paris still the capital?", "top_k": 3, "metadata_filters": {"msg_type": MessageType.FACT.value}},
        return_msg_type=MessageType.RESPONSE,
        conversation_id=conversation_id
    )
    print("\nTest 12: Conversation after loading agent")
    print(f"User: Is Paris still the capital?")
    print(f"Assistant: {response.content['response']}")
    print(f"Memory IDs: {response.content['memory_ids']}")

