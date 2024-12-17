from pydantic import Field
from typing import Optional, List

from ..core.module import BaseModule
from ..core.module_utils import generate_id
from ..core.message import Message
from ..models.model_configs import LLMConfig
from ..models.base_model import BaseLLM
from ..memory.memory import ShortTermMemory
from ..memory.long_term_memory import LongTermMemory
from ..memory.memory_manager import MemoryManager
from ..storages.base import StorageHandler
from ..actions.action import Action


class Agent(BaseModule):

    name: str # should be unique
    description: str
    llm_config: Optional[LLMConfig] = None
    llm: Optional[BaseLLM] = None
    agent_id: Optional[str] = Field(default_factory=generate_id)
    system_prompt: Optional[str] = None
    short_term_memory: Optional[ShortTermMemory] = Field(default_factory=ShortTermMemory) # store short term memory for a single workflow.
    use_long_term_memory: Optional[bool] = False
    storage_handler: Optional[StorageHandler] = None
    long_term_memory: Optional[LongTermMemory] = None
    long_term_memory_manager: Optional[MemoryManager] = None
    actions: List[Action] = Field(default=None)
    is_human: bool = Field(default=False)
    version: int = 0 

    def init_module(self):
        """
        Step 1: If self.is_human is False, llm_config or llm must be provided. 
        Step 2: Initialize self._llm: 
            - If self.llm_config is provided, create a BaseLLM instance based on the config
            - If self.llm is provided, directly use the provided llm and set the self.llm_config to self.llm.config 
        Step 3: If self.use_long_term_memory is True:
            - If self.storage_handler is None, raise an error, since is required in this case. 
            - If self.long_term_memory is None, create a new LongTermMemory instance. 
                Otherwise, use self.long_term_memory directly (load from saved memory).
            - If self.long_term_memory_manager is None, create a new MemoryManager using 
                both self.storage_handler and self.long_term_memory as input.
        """
        pass

    def execute(self, action_name: str, msgs: List[Message], **kwargs) -> Message:
        """
        Execute an action.

        Args:
            action_name (str): the name of the action to execute. 
            msgs (List[Message]): the context for the current task. 
        
        Returns:
            Message: a message that contains the execution results. 
        """
        pass 

    def clear_short_term_memory(self):
        """
        remove all the content in the short term memory.
        """
        pass 
        
    def __eq__(self, other: "Agent"):
        return self.agent_id == other.agent_id

    def __hash__(self):
        return self.agent_id

